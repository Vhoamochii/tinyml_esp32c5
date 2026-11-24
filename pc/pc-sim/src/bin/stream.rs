// Bước 5e: phát luồng Paderborn THẬT xuống ESP32-C5 qua UART (cổng /dev/ttyUSB0) và đối chiếu từng sự kiện
// với bản tham chiếu chạy song song trên Ubuntu (cùng mã tinyml-core). Giao thức: tinyml-core/src/protocol.rs.
//
// Cách chạy:
//   1. cd embedded/firmware && cargo run --release   -> đợi dòng "[STREAM] Cho PC ...", rồi Ctrl+C
//   2. cd pc/pc-sim && cargo run --release --bin stream -- [so_buoc] [C1|C2|C3] [rai|dot] [hat_giong] [lr] [luong] [cong]
//      Mặc định: 300 C2 rai 11 0.1 cauhinh /dev/ttyUSB0. Luồng: cauhinh (K004, KI14, KA16), A hoặc B (6 ổ kiểm thử).
// Cấu hình đã khóa giống pc-sim: K = 25, r = 0%, ưu tiên Q, Q tối đa 9, 2 lượt học mỗi bước.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::process::Command;
use std::time::{Duration, Instant};

use pc_sim::paderborn;
use tinyml_core::event::Event;
use tinyml_core::features::{extract, normalize, WINDOW_LEN};
use tinyml_core::learner::{Config, Learner};
use tinyml_core::policy::Retention;
use tinyml_core::pretrained;
use tinyml_core::protocol;
use tinyml_core::schedule::Schedule;
use tinyml_core::NUM_CLASSES;

const K: usize = 25; // Phải bằng LOCKED_K của firmware (bo báo lại trong dòng CONFIG OK)
const Q_MAX: u32 = 9;
const UPDATES_PER_STEP: u32 = 2;
const LABEL_GROUP: u32 = 9;
/// Hạn mỗi cửa sổ theo đề cương: T = bước trượt / 64000 giây = 16384 / 64000 s
const WINDOW_BUDGET_US: u32 = 256_000;
/// Luồng kiểm thử, giống TEST_STREAMS trong main.rs
const STREAM_A: [&str; NUM_CLASSES] = ["K005", "KI18", "KA22"];
const STREAM_B: [&str; NUM_CLASSES] = ["K006", "KI21", "KA30"];

fn fail(msg: &str) -> ! {
    eprintln!("[STREAM][ERROR] {msg}");
    std::process::exit(1);
}

/// Đặt tốc độ và chế độ thô cho cổng; `timeout_ds` = thời gian chờ mỗi lần đọc (phần mười giây)
fn stty(port: &str, baud: u32, timeout_ds: u8) {
    let ok = Command::new("stty")
        .args(["-F", port, &baud.to_string(), "raw", "-echo", "-hupcl", "min", "0", "time", &timeout_ds.to_string()])
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        fail(&format!("stty khong dat duoc {baud} baud cho {port}"));
    }
}

/// Một dòng từ bo (bỏ ký tự cuối dòng); None nếu hết thời gian chờ
fn read_line(reader: &mut BufReader<File>) -> Option<String> {
    let mut buf = Vec::new();
    match reader.read_until(b'\n', &mut buf) {
        Ok(0) => None,
        Ok(_) => Some(String::from_utf8_lossy(&buf).trim_end().to_string()),
        Err(e) => fail(&format!("doc cong loi: {e}")),
    }
}

/// Gửi `cmd` lặp lại cho tới khi nhận dòng bắt đầu bằng `want`
fn handshake(reader: &mut BufReader<File>, writer: &mut File, cmd: u8, want: &str, tries: u32) -> bool {
    for _ in 0..tries {
        if writer.write_all(&[cmd]).and_then(|_| writer.flush()).is_err() {
            fail("ghi cong loi");
        }
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(1) {
            match read_line(reader) {
                Some(line) if line.starts_with(want) => return true,
                Some(line) if !line.is_empty() => println!("[BO] {line}"),
                Some(_) => {}
                None => break,
            }
        }
    }
    false
}

/// Độ trễ nhãn theo lịch, giống LabelSchedule trong main.rs: rải = j, theo đợt = 8 - j
fn label_delay(burst: bool, id: u32) -> u32 {
    let j = id % LABEL_GROUP;
    if burst {
        LABEL_GROUP - 1 - j
    } else {
        j
    }
}

/// Các cửa sổ (lớp, tín hiệu) theo đúng thứ tự luồng của pc-sim: N15 lần đo 1-16 rồi N09 lần đo 1-20,
/// luân phiên bản ghi giữa ba lớp; dừng khi đủ `steps` cửa sổ
fn load_stream(bearings: &[&'static str], steps: usize) -> Vec<(u8, Vec<f32>)> {
    let root = paderborn::data_dir();
    let mut out = Vec::new();
    for (condition, (first, last)) in [(paderborn::OLD_SPEED, (1, 16)), (paderborn::NEW_SPEED, (1, 20))] {
        if out.len() >= steps {
            break;
        }
        let mut records = paderborn::load_raw_records(&root, condition, bearings, first..=last).unwrap_or_else(|e| fail(&e));
        records.sort_by_key(|r| r.measurement);
        for r in records {
            out.extend(r.windows.into_iter().map(|w| (r.class, w)));
        }
    }
    out.truncate(steps);
    out
}

/// (trung vị, P95 theo hạng gần nhất, lớn nhất)
fn stats(v: &[u32]) -> (u32, u32, u32) {
    if v.is_empty() {
        return (0, 0, 0);
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    (s[s.len() / 2], s[(s.len() * 95).div_ceil(100) - 1], s[s.len() - 1])
}

/// Giá trị của khóa `key=` trong dòng [STEP]
fn field(line: &str, key: &str) -> Option<u32> {
    line.split_whitespace().find_map(|t| t.strip_prefix(key)?.strip_prefix('=')?.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: &str| args.get(i).cloned().unwrap_or_else(|| default.to_string());
    let steps: usize = arg(0, "300").parse().unwrap_or_else(|_| fail("so buoc khong hop le"));
    let case = arg(1, "C2");
    let burst = match arg(2, "rai").as_str() {
        "rai" => false,
        "dot" => true,
        other => fail(&format!("lich nhan '{other}' khong hop le (rai|dot)")),
    };
    let seed: u32 = arg(3, "11").parse().unwrap_or_else(|_| fail("hat giong khong hop le"));
    let lr: f32 = arg(4, "0.1").parse().unwrap_or_else(|_| fail("lr khong hop le"));
    let stream_name = arg(5, "cauhinh");
    let port = arg(6, "/dev/ttyUSB0");

    let (replay_enabled, retention) = match case.as_str() {
        "C1" => (false, Retention::Fifo),
        "C2" => (true, Retention::Fifo),
        "C3" => (true, Retention::Reservoir),
        other => fail(&format!("case '{other}' khong hop le (C1|C2|C3)")),
    };
    let bearings: &[&'static str] = match stream_name.as_str() {
        "cauhinh" => &paderborn::CONFIG_BEARINGS,
        "A" => &STREAM_A,
        "B" => &STREAM_B,
        other => fail(&format!("luong '{other}' khong hop le (cauhinh|A|B)")),
    };
    if !pretrained::AVAILABLE {
        fail("chua co trong so pretrain: chay `cargo run --release --bin pretrain` truoc");
    }
    let cfg = Config { r_protected: 0, q_max: Q_MAX, schedule: Schedule::QPriority, replay_enabled, lr, seed, retention };
    println!(
        "[STREAM] {steps} buoc | {case} | lich nhan {} | hat giong {seed} | lr = {lr} | luong {stream_name} {bearings:?} | cong {port}",
        if burst { "theo dot" } else { "rai" }
    );

    let data = load_stream(bearings, steps);
    if data.len() < steps {
        fail(&format!("luong chi co {} cua so", data.len()));
    }

    // --- Mở cổng và bắt tay ---
    stty(&port, protocol::BOOT_BAUD, 10);
    let file = OpenOptions::new().read(true).write(true).open(&port).unwrap_or_else(|e| fail(&format!("mo {port}: {e}")));
    let mut writer = file.try_clone().unwrap_or_else(|e| fail(&format!("{e}")));
    let mut reader = BufReader::new(file);
    println!("[STREAM] Bat tay o {} baud (bo co the dang khoi dong lai)...", protocol::BOOT_BAUD);
    if !handshake(&mut reader, &mut writer, protocol::CMD_GO, "[STREAM] BAUD", 30) {
        fail("bo khong tra loi: da nap firmware moi va thoat espflash monitor chua?");
    }
    std::thread::sleep(Duration::from_millis(100));
    stty(&port, protocol::FAST_BAUD, 50);
    if !handshake(&mut reader, &mut writer, protocol::CMD_PING, "[STREAM] OK", 10) {
        fail(&format!("khong lien lac duoc o {} baud", protocol::FAST_BAUD));
    }
    let mut frame = vec![protocol::CMD_CONFIG];
    frame.extend_from_slice(&protocol::encode_config(&cfg, UPDATES_PER_STEP));
    writer.write_all(&frame).unwrap_or_else(|e| fail(&format!("{e}")));
    match read_line(&mut reader) {
        Some(line) if line.starts_with("[STREAM] CONFIG OK") && line.contains(&format!("K={K} ")) => println!("{line}"),
        other => fail(&format!("bo khong nhan cau hinh: {other:?}")),
    }

    // --- Bản tham chiếu trên Ubuntu, chạy song song từng bước ---
    let mut reference: Learner<K, NUM_CLASSES> = Learner::new(&cfg);
    reference.head = pretrained::head();
    let mut pending: Vec<(u32, u32, u8)> = Vec::new(); // (bước nhãn về, ID, nhãn)
    let (mut ext_us, mut learn_us, mut loop_us, mut wall_us) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut matched_steps, mut correct) = (0usize, 0usize);
    let mut mismatches_shown = 0;

    for (step, (class, window)) in data.iter().enumerate() {
        let step = step as u32;
        pending.push((step + label_delay(burst, step), step, *class));
        let (mut due, rest): (Vec<_>, Vec<_>) = pending.into_iter().partition(|&(t, _, _)| t <= step);
        pending = rest;
        due.sort_by_key(|&(_, id, _)| id);
        if due.len() > protocol::MAX_LABELS_PER_STEP {
            fail("qua nhieu nhan trong mot buoc");
        }

        // Tham chiếu: cùng thứ tự gọi như pc-sim (dự đoán, nhãn đến hạn theo ID tăng dần, học)
        let mut expected: Vec<String> = Vec::new();
        let mut log = |e: Event| expected.push(e.to_string());
        let mut z = extract(window.as_slice().try_into().expect("cua so du WINDOW_LEN mau"));
        normalize(&mut z, &pretrained::FEATURE_MEAN, &pretrained::FEATURE_STD);
        let ref_pred = reference.on_window(step, step, z, &mut log);
        for &(_, id, label) in &due {
            reference.on_label(step, id, label, &mut log);
        }
        reference.train(step, UPDATES_PER_STEP, &mut log);

        // Khung gửi xuống bo
        let mut frame = Vec::with_capacity(6 + due.len() * 5 + protocol::SAMPLE_BYTES);
        frame.push(protocol::CMD_STEP);
        frame.extend_from_slice(&step.to_le_bytes());
        frame.push(due.len() as u8);
        for &(_, id, label) in &due {
            frame.extend_from_slice(&id.to_le_bytes());
            frame.push(label);
        }
        debug_assert_eq!(window.len(), WINDOW_LEN);
        for v in window {
            frame.extend_from_slice(&v.to_le_bytes());
        }
        let sent = Instant::now();
        writer.write_all(&frame).and_then(|_| writer.flush()).unwrap_or_else(|e| fail(&format!("{e}")));

        // Kết quả của bo: các dòng [EVT], kết thúc bằng [STEP]
        let mut got: Vec<String> = Vec::new();
        let step_line = loop {
            match read_line(&mut reader) {
                Some(line) if line.starts_with("[STEP]") => break line,
                Some(line) if line.starts_with("[EVT]") => got.push(line),
                Some(line) if !line.is_empty() => println!("[BO] {line}"),
                Some(_) => {}
                None => fail(&format!("bo khong tra loi o buoc {step}")),
            }
        };
        wall_us.push(sent.elapsed().as_micros() as u32);
        let board_pred = field(&step_line, "pred");
        let (e_us, l_us) = (field(&step_line, "ext_us").unwrap_or(0), field(&step_line, "learn_us").unwrap_or(0));
        ext_us.push(e_us);
        learn_us.push(l_us);
        loop_us.push(e_us + l_us);

        let same = got == expected && board_pred == Some(ref_pred as u32);
        matched_steps += same as usize;
        correct += (ref_pred == *class) as usize;
        if !same && mismatches_shown < 3 {
            mismatches_shown += 1;
            println!("[STREAM][LECH] buoc {step}: bo {got:?} pred {board_pred:?} | Ubuntu {expected:?} pred {ref_pred}");
        }
        if (step + 1) % 50 == 0 {
            println!("[STREAM] buoc {}/{steps} | khop {matched_steps}/{}", step + 1, step + 1);
        }
    }

    let n = steps as f32;
    let over = loop_us.iter().filter(|&&t| t > WINDOW_BUDGET_US).count();
    let (e50, e95, emax) = stats(&ext_us);
    let (l50, l95, lmax) = stats(&learn_us);
    let (t50, t95, tmax) = stats(&loop_us);
    let (w50, w95, _) = stats(&wall_us);
    println!("\n[STREAM] ===== KET QUA {steps} BUOC =====");
    println!("[STREAM] Khop tung su kien va du doan voi Ubuntu: {matched_steps}/{steps} buoc");
    println!("[STREAM] Du doan dung (truoc khi co nhan): {:.2}%", correct as f32 / n * 100.0);
    println!("[STREAM] Trich dac trung tren bo: trung vi {e50} us | P95 {e95} us | max {emax} us");
    println!("[STREAM] Buoc hoc tren bo:        trung vi {l50} us | P95 {l95} us | max {lmax} us");
    println!("[STREAM] Toan vong tren bo:       trung vi {t50} us | P95 {t95} us | max {tmax} us");
    println!("[STREAM] Qua han (> {WINDOW_BUDGET_US} us): {over}/{steps} ({:.2}%)", over as f32 / n * 100.0);
    println!("[STREAM] Ca truyen UART (PC do, gui khung -> nhan [STEP]): trung vi {w50} us | P95 {w95} us");
    if matched_steps != steps {
        std::process::exit(1);
    }
}
