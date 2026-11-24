// Điểm vào firmware ESP32-C5 (no_std, không heap).
// 1. Chạy hai phép thử bắt buộc của đề cương (XOR và ghép nhãn qua ranh giới epoch) bằng tinyml-core,
//    cùng mã với `cargo run --bin selftest` trên Ubuntu. Log in qua UART phải khớp từng dòng với Ubuntu.
// 2. Bước 5, đo thời gian trên bo: bộ trích xuất 16 đặc trưng trên tín hiệu giả dùng chung với Ubuntu,
//    và một bước học đầy đủ với cấu hình đã khóa (K = 25, r = 0%, ưu tiên Q; Case 3 reservoir là case
//    nặng nhất vì phải băm ID khi thu hồi). Ghi log UART nằm ngoài khoảng đo.
// 3. Bước 5e, nhận luồng thật từ PC qua UART0 (GPIO11 TX, GPIO12 RX, cổng /dev/ttyUSB0) theo giao thức
//    tinyml-core/src/protocol.rs; PC chạy `cargo run --release --bin stream` trong pc/pc-sim.
#![no_std]
#![no_main]

use esp_hal::clock::CpuClock;
use esp_hal::main;
use core::fmt::Write as _;
use esp_hal::time::Instant;
use esp_hal::uart::{Config as UartConfig, Uart};
use esp_hal::Blocking;
use esp_println::println;
use tinyml_core::event::{Event, EventKind};
use tinyml_core::features::{normalize, WindowExtractor, BLOCKS_PER_WINDOW, BLOCK_LEN, SAMPLE_RATE_HZ};
use tinyml_core::learner::{Config, Learner, MANAGEMENT_BYTES};
use tinyml_core::memsim::{slots_for_budget, SLOT_SIZE_BYTES};
use tinyml_core::policy::Retention;
use tinyml_core::pretrained;
use tinyml_core::protocol;
use tinyml_core::schedule::Schedule;
use tinyml_core::testsignal::{FakeSignal, CHECK_WINDOWS};
use tinyml_core::{selftest, NUM_CLASSES, NUM_FEATURES};

// Bắt buộc với esp-hal 1.x: espflash cần app descriptor để nạp firmware
esp_bootloader_esp_idf::esp_app_desc!();

const MEMORY_BYTES: usize = 2000; // M, giống pc-sim
/// Cấu hình đã khóa trên tập chọn cấu hình (pc-sim): K = 25 ô
const LOCKED_K: usize = 25;
/// Số bước học dùng để đo
const BENCH_STEPS: usize = 200;
/// Số sự kiện tối đa giữ lại trong một bước của luồng (gửi sau khi đo thời gian)
const MAX_EVENTS_PER_STEP: usize = 64;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("[FW][PANIC] {}", info);
    loop {}
}

#[main]
fn main() -> ! {
    // Chạy CPU ở xung nhịp cao nhất; đề cương yêu cầu ghi xung nhịp bo kèm số đo thời gian
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    println!(
        "[FW][BOOT] s = {} byte/o, M = {} byte, H = {} byte -> K = {} o | CPU = CpuClock::max()",
        SLOT_SIZE_BYTES,
        MEMORY_BYTES,
        MANAGEMENT_BYTES,
        slots_for_budget(MEMORY_BYTES, MANAGEMENT_BYTES)
    );

    let mut out = |a: core::fmt::Arguments<'_>| println!("{}", a);
    let xor_ok = selftest::run_xor(&mut out);
    let epoch_ok = selftest::run_epoch_matching(&mut out);
    println!(
        "[SELFTEST] XOR={} EPOCH={}",
        if xor_ok { "PASS" } else { "FAIL" },
        if epoch_ok { "PASS" } else { "FAIL" }
    );

    println!(
        "[BENCH][RAM] WindowExtractor = {} byte | Learner<{}, {}> = {} byte | pretrain co san = {}",
        core::mem::size_of::<WindowExtractor>(),
        LOCKED_K,
        NUM_CLASSES,
        core::mem::size_of::<Learner<LOCKED_K, NUM_CLASSES>>(),
        pretrained::AVAILABLE
    );
    let z = bench_extractor();
    bench_learner(z);
    println!("[BENCH] XONG");

    // Bước 5e: nhận luồng từ PC. Đợi chữ cuối kịp ra khỏi UART trước khi driver Uart lấy UART0
    println!("[STREAM] Cho PC: thoat espflash monitor roi chay `cargo run --release --bin stream` trong pc/pc-sim");
    busy_wait_us(50_000);
    let uart = Uart::new(peripherals.UART0, UartConfig::default().with_baudrate(protocol::BOOT_BAUD))
        .unwrap()
        .with_rx(peripherals.GPIO12)
        .with_tx(peripherals.GPIO11);
    stream_mode(uart)
}

/// Số µs từ `t0` đến giờ
fn micros_since(t0: Instant) -> u32 {
    t0.elapsed().as_micros() as u32
}

/// (trung vị, P95 theo hạng gần nhất, lớn nhất) của các số đo µs; sắp xếp tại chỗ
fn stats(v: &mut [u32]) -> (u32, u32, u32) {
    v.sort_unstable();
    let p95 = v[(v.len() * 95).div_ceil(100) - 1];
    (v[v.len() / 2], p95, v[v.len() - 1])
}

/// Đo bộ trích xuất theo từng khối như khi nhận luồng thật. Hạn: mỗi khối 1024 mẫu đến cách nhau 16 ms,
/// mỗi cửa sổ 16 khối cách nhau 256 ms (đề cương: T = bước trượt / 64000 giây). Trả về z của cửa sổ cuối,
/// in ra để so với dòng cùng tên của `cargo run --bin selftest` trên Ubuntu (tín hiệu giả dùng chung).
fn bench_extractor() -> [f32; NUM_FEATURES] {
    let mut extractor = WindowExtractor::new();
    let mut signal = FakeSignal::new();
    let mut block = [0.0f32; BLOCK_LEN];
    let mut normal = [0u32; CHECK_WINDOWS * (BLOCKS_PER_WINDOW - 1)];
    let mut last = [0u32; CHECK_WINDOWS];
    let mut window_total = [0u32; CHECK_WINDOWS];
    let mut z = [0.0f32; NUM_FEATURES];

    for w in 0..CHECK_WINDOWS {
        for b in 0..BLOCKS_PER_WINDOW {
            signal.fill(&mut block);
            let t0 = Instant::now();
            let result = extractor.push_block(&block);
            let dt = micros_since(t0);
            window_total[w] += dt;
            match result {
                Some(features) => {
                    last[w] = dt;
                    z = features;
                }
                None => normal[w * (BLOCKS_PER_WINDOW - 1) + b] = dt,
            }
        }
    }

    let block_budget = (BLOCK_LEN as f32 / SAMPLE_RATE_HZ * 1e6) as u32;
    let window_budget = block_budget * BLOCKS_PER_WINDOW as u32;
    let (med, p95, max) = stats(&mut normal);
    println!(
        "[BENCH][EXTRACT] khoi thuong ({} khoi): trung vi {} us | P95 {} us | max {} us | han moi khoi {} us",
        normal.len(),
        med,
        p95,
        max,
        block_budget
    );
    let (med, _, max) = stats(&mut last);
    println!("[BENCH][EXTRACT] khoi cuoi (them FFT pho bao): trung vi {} us | max {} us", med, max);
    let (med, _, max) = stats(&mut window_total);
    println!(
        "[BENCH][EXTRACT] ca cua so ({} khoi): trung vi {} us | max {} us | han {} us -> {}",
        BLOCKS_PER_WINDOW,
        med,
        max,
        window_budget,
        if max <= window_budget { "KIP" } else { "KHONG KIP" }
    );
    println!("[BENCH][EXTRACT] z = {:?}", z);
    z
}

/// Đo một bước học đầy đủ: chuẩn hóa + on_window (dự đoán, cấp ô) + on_label + train(2 lượt).
/// Nhãn trễ 4 bước nên sau ~25 bước mảng đầy và mỗi bước đều phải thu hồi ô (reservoir băm ID).
fn bench_learner(z: [f32; NUM_FEATURES]) {
    let cfg = Config {
        r_protected: 0,
        q_max: 9,
        schedule: Schedule::QPriority,
        replay_enabled: true,
        lr: 0.1,
        seed: 11,
        retention: Retention::Reservoir,
    };
    let mut learner: Learner<LOCKED_K, NUM_CLASSES> = Learner::new(&cfg);
    learner.head = pretrained::head();
    let mut ignore = |_e: Event| {};
    let mut times = [0u32; BENCH_STEPS];

    for step in 0..BENCH_STEPS as u32 {
        let mut features = z;
        features[0] += step as f32 * 0.001; // mỗi bước một vectơ hơi khác
        let t0 = Instant::now();
        normalize(&mut features, &pretrained::FEATURE_MEAN, &pretrained::FEATURE_STD);
        learner.on_window(step, step, features, &mut ignore);
        if step >= 4 {
            learner.on_label(step, step - 4, ((step - 4) % NUM_CLASSES as u32) as u8, &mut ignore);
        }
        learner.train(step, 2, &mut ignore);
        times[step as usize] = micros_since(t0);
    }

    let (med, p95, max) = stats(&mut times);
    println!(
        "[BENCH][LEARN] 1 buoc (chuan hoa + du doan + cap o + nhan + 2 luot hoc), {} buoc: trung vi {} us | P95 {} us | max {} us",
        BENCH_STEPS, med, p95, max
    );
}

/// Chờ bận `us` micro giây
fn busy_wait_us(us: u64) {
    let t0 = Instant::now();
    while t0.elapsed().as_micros() < us {}
}

/// Ghi hết dữ liệu ra UART (`write` có thể ghi ít hơn độ dài được đưa)
fn uart_write_all(uart: &mut Uart<'_, Blocking>, data: &[u8]) {
    let mut sent = 0;
    while sent < data.len() {
        match uart.write(&data[sent..]) {
            Ok(n) => sent += n,
            Err(_) => return,
        }
    }
}

/// Đọc đủ `buf.len()` byte (chặn tới khi có dữ liệu); false nếu UART báo lỗi
fn uart_read_exact(uart: &mut Uart<'_, Blocking>, buf: &mut [u8]) -> bool {
    let mut filled = 0;
    while filled < buf.len() {
        match uart.read(&mut buf[filled..]) {
            Ok(n) => filled += n,
            Err(_) => return false,
        }
    }
    true
}

/// Cho phép dùng writeln! trên UART
struct UartText<'a, 'd>(&'a mut Uart<'d, Blocking>);

impl core::fmt::Write for UartText<'_, '_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        uart_write_all(self.0, s.as_bytes());
        Ok(())
    }
}

/// Nhận luồng từ PC theo protocol.rs: bắt tay, đổi tốc độ, rồi xử lý lệnh CONFIG / STEP / PING mãi mãi
fn stream_mode(mut uart: Uart<'_, Blocking>) -> ! {
    // Bắt tay ở BOOT_BAUD: chờ CMD_GO, trả lời, rồi đổi sang FAST_BAUD.
    // Dòng xác nhận + nhịp "đang chờ" mỗi 2 giây cho biết chiều bo -> PC hoạt động; byte lạ được in ra
    // để biết chiều PC -> bo hoạt động (có thể gõ G thẳng trong espflash monitor để thử)
    uart_write_all(&mut uart, b"[STREAM] UART0 san sang o 115200 baud, cho lenh G tu PC
");
    let mut byte = [0u8; 1];
    let mut last_beat = Instant::now();
    loop {
        if uart.read_ready() {
            if uart_read_exact(&mut uart, &mut byte) {
                if byte[0] == protocol::CMD_GO {
                    break;
                }
                let _ = writeln!(UartText(&mut uart), "[STREAM] nhan byte 0x{:02x}, chua phai G", byte[0]);
            }
        } else if last_beat.elapsed().as_micros() > 2_000_000 {
            uart_write_all(&mut uart, b"[STREAM] van dang cho lenh G...
");
            last_beat = Instant::now();
        }
    }
    let _ = writeln!(UartText(&mut uart), "[STREAM] BAUD {}", protocol::FAST_BAUD);
    let _ = uart.flush();
    busy_wait_us(20_000);
    uart.apply_config(&UartConfig::default().with_baudrate(protocol::FAST_BAUD)).unwrap();

    let mut extractor = WindowExtractor::new();
    let mut learner: Option<(Learner<LOCKED_K, NUM_CLASSES>, u32)> = None;
    let mut samples = [0u8; protocol::SAMPLE_BYTES];
    let mut label_bytes = [0u8; protocol::MAX_LABELS_PER_STEP * 5];
    let mut events = [Event { step: 0, id: 0, kind: EventKind::Admitted }; MAX_EVENTS_PER_STEP];

    loop {
        if !uart_read_exact(&mut uart, &mut byte) {
            continue;
        }
        match byte[0] {
            protocol::CMD_PING => uart_write_all(&mut uart, b"[STREAM] OK\n"),
            protocol::CMD_CONFIG => {
                let mut raw = [0u8; protocol::CONFIG_LEN];
                if !uart_read_exact(&mut uart, &mut raw) {
                    continue;
                }
                match protocol::decode_config(&raw) {
                    Some((cfg, updates)) => {
                        let mut l = Learner::new(&cfg);
                        l.head = pretrained::head();
                        learner = Some((l, updates));
                        let _ = writeln!(UartText(&mut uart), "[STREAM] CONFIG OK K={} H={} ", LOCKED_K, MANAGEMENT_BYTES);
                    }
                    None => uart_write_all(&mut uart, b"[STREAM] CONFIG LOI\n"),
                }
            }
            protocol::CMD_STEP => {
                // Đầu khung: bước (u32) + số nhãn (u8), rồi các nhãn, rồi mẫu
                let mut head = [0u8; 5];
                if !uart_read_exact(&mut uart, &mut head) {
                    continue;
                }
                let step = u32::from_le_bytes([head[0], head[1], head[2], head[3]]);
                let n_labels = head[4] as usize;
                if n_labels > protocol::MAX_LABELS_PER_STEP {
                    uart_write_all(&mut uart, b"[STREAM] QUA NHIEU NHAN\n");
                    continue;
                }
                if !uart_read_exact(&mut uart, &mut label_bytes[..n_labels * 5])
                    || !uart_read_exact(&mut uart, &mut samples)
                {
                    continue;
                }
                let Some((learner, updates)) = learner.as_mut() else {
                    uart_write_all(&mut uart, b"[STREAM] CHUA CO CONFIG\n");
                    continue;
                };

                // Trích đặc trưng theo khối; chỉ tính thời gian push_block (đổi byte -> f32 nằm ngoài khoảng đo)
                let mut ext_us = 0u32;
                let mut z = [0.0f32; NUM_FEATURES];
                for chunk in samples.chunks_exact(BLOCK_LEN * 4) {
                    let mut block = [0.0f32; BLOCK_LEN];
                    for (v, b) in block.iter_mut().zip(chunk.chunks_exact(4)) {
                        *v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    }
                    let t0 = Instant::now();
                    let result = extractor.push_block(&block);
                    ext_us += micros_since(t0);
                    if let Some(features) = result {
                        z = features;
                    }
                }

                // Học: chuẩn hóa + dự đoán + cấp ô + nhãn đến hạn + học; sự kiện gom vào bộ đệm
                let mut count = 0usize;
                let mut overflow = false;
                let t1 = Instant::now();
                let pred = {
                    let mut log = |e: Event| {
                        if count < events.len() {
                            events[count] = e;
                            count += 1;
                        } else {
                            overflow = true;
                        }
                    };
                    normalize(&mut z, &pretrained::FEATURE_MEAN, &pretrained::FEATURE_STD);
                    let pred = learner.on_window(step, step, z, &mut log);
                    for lb in label_bytes[..n_labels * 5].chunks_exact(5) {
                        let id = u32::from_le_bytes([lb[0], lb[1], lb[2], lb[3]]);
                        learner.on_label(step, id, lb[4], &mut log);
                    }
                    learner.train(step, *updates, &mut log);
                    pred
                };
                let learn_us = micros_since(t1);

                // Gửi kết quả sau khi đã đo xong
                for e in &events[..count] {
                    let _ = writeln!(UartText(&mut uart), "{}", e);
                }
                if overflow {
                    uart_write_all(&mut uart, b"[STREAM][WARN] tran bo dem su kien\n");
                }
                let _ = writeln!(
                    UartText(&mut uart),
                    "[STEP] step={} pred={} ext_us={} learn_us={}",
                    step,
                    pred,
                    ext_us,
                    learn_us
                );
            }
            _ => {} // Bỏ qua byte lạ (ví dụ CMD_GO thừa từ lúc bắt tay)
        }
    }
}
