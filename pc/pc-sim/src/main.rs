// Bộ phát luồng + chấm điểm trên Ubuntu (phía máy tính trong đề cương phương pháp).
// Máy tính giữ nhãn thật, chỉ trả nhãn theo lịch đã định; phía thiết bị dùng tinyml-core, cùng mã với firmware.
//
// BƯỚC 4, DỮ LIỆU THẬT: luồng Paderborn trên 3 ổ chọn cấu hình (K004, KI14, KA16).
//   - Đoạn tốc độ cũ N15_M07_F10, lần đo 1-16, rồi đoạn tốc độ mới N09_M07_F10, lần đo 1-20.
//   - Trong mỗi đoạn, luân phiên bản ghi giữa ba lớp, giữ thứ tự 15 cửa sổ trong từng bản ghi.
//   - Lần đo 17-20 ở tốc độ cũ để riêng đo mức quên, không vào luồng học.
//   - Đặc trưng chuẩn hóa và lớp cuối khởi đầu bằng kết quả pretrain (tinyml-core/src/pretrained.rs).
// Mỗi lớp chỉ có một ổ nên BA theo lớp cũng là BA theo ổ.
// Mặc định: cấu hình đã khóa (K = 25, r = 0%, ưu tiên Q, Q tối đa 9) với 3 phương án nhóm chốt:
//   Case 1 = tắt học lại, Case 2 = R vào trước ra trước (luật đề cương), Case 3 = R reservoir.
//   In (1) bảng chọn tốc độ học lr, tự chọn lr theo luật ở choose_learning_rate, và (2) bảng chốt trên tập
//   chọn cấu hình với lr đã chọn; có độ lệch chuẩn giữa 5 hạt giống.
// --sweep: thí nghiệm chọn cấu hình đầy đủ: quét K = 12, 15, 20, 25; r = 0%, 25%, 50% với cả hai luật R;
//   hai lịch học (xen kẽ, ưu tiên Q). Cả hai chế độ đều chạy hai lịch trả nhãn (rải, theo đợt).
// --test: GIAI ĐOẠN KIỂM THỬ trên 6 ổ kiểm thử. Trước tiên áp luật chọn lr đã khóa trên 3 ổ chọn cấu hình
//   (khóa LUẬT, không khóa con số: nếu bộ trích xuất đổi thì lr được chọn lại đúng luật, chỉ trên tập cấu hình),
//   rồi chạy 6 ổ kiểm thử với lr đó; không chọn lại gì trên dữ liệu kiểm thử. Hai luồng A, B, mỗi luồng một ổ mỗi lớp; báo theo từng ổ,
//   rồi tổng hợp 6 ổ kèm độ phân tán giữa các ổ.
//   cargo run --release --bin pc-sim
//   cargo run --release --bin pc-sim -- --sweep
//   cargo run --release --bin pc-sim -- --test

use std::collections::VecDeque;

use pc_sim::paderborn::{self, Record};
use tinyml_core::event::{Event, EventKind};
use tinyml_core::features::normalize;
use tinyml_core::head::Head;
use tinyml_core::learner::{Config, Learner, MANAGEMENT_BYTES};
use tinyml_core::memsim::{slots_for_budget, SlotState, SLOT_SIZE_BYTES};
use tinyml_core::policy::Retention;
use tinyml_core::pretrained;
use tinyml_core::schedule::Schedule;
use tinyml_core::{NUM_CLASSES, NUM_FEATURES};

type Features = [f32; NUM_FEATURES];

// Giới hạn bộ nhớ: K = floor((M - H) / s). Quét K = 12, 15, 20, 25 trong main() để tìm vùng bộ nhớ
// đủ chật cho r có tác dụng; M tương ứng = K·s + H.
const Q_MAX: u32 = 9; // Giới hạn Q (tạm): bằng một đợt nhãn, để đợt nhãn không tự mất 1 nhãn
const UPDATES_PER_STEP: u32 = 2; // Số lượt học mỗi bước, thay cho giới hạn thời gian mỗi bước (tạm)
const LEARNING_RATE: f32 = 0.1; // Chỉ dùng cho --sweep (chế độ mặc định tự chọn lr trong LR_CANDIDATES)

// Cấu hình đã khóa sau thí nghiệm chọn cấu hình (--sweep): r = 0% vì r lớn hơn không giảm quên mà chỉ
// làm mất nhãn; ưu tiên Q vì chờ ngắn hơn, BA cao hơn ở lịch theo đợt; BA gần như không đổi theo K
const LOCKED_K: usize = 25;
const LOCKED_SCHEDULE: Schedule = Schedule::QPriority;
const LR_CANDIDATES: [f32; 5] = [0.05, 0.1, 0.2, 0.3, 0.4];
/// Ngưỡng chọn lr: cỡ độ lệch chuẩn lớn nhất của BA mới giữa các hạt giống (điểm %)
const LR_BA_TOLERANCE: f32 = 0.5;
/// Giai đoạn kiểm thử: hai luồng, mỗi luồng một ổ khỏe, một ổ vòng trong, một ổ vòng ngoài (đúng thứ tự lớp 0, 1, 2)
const TEST_STREAMS: [(&str, [&str; NUM_CLASSES]); 2] = [("A", ["K005", "KI18", "KA22"]), ("B", ["K006", "KI21", "KA30"])];

// Luồng Paderborn
const OLD_STREAM_MEASUREMENTS: (u32, u32) = (1, 16); // Tốc độ cũ, vào luồng học
const OLD_HELDOUT_MEASUREMENTS: (u32, u32) = (17, 20); // Tốc độ cũ, để riêng đo mức quên
const NEW_STREAM_MEASUREMENTS: (u32, u32) = (1, 20); // Tốc độ mới
const LABEL_GROUP: u32 = 9; // Trả nhãn theo từng nhóm chín yêu cầu
const SLIDING_WINDOWS: usize = 96; // Cửa sổ trượt đo phục hồi (đủ ba lớp)
const SEEDS: [u32; 5] = [11, 22, 33, 44, 55];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LabelSchedule {
    Spread, // Lịch rải: vị trí j trong nhóm trễ j bước
    Burst,  // Lịch theo đợt: trễ 8 - j bước, cả nhóm về cùng một bước
}

impl LabelSchedule {
    fn delay(self, id: u32) -> u32 {
        let j = id % LABEL_GROUP;
        match self {
            Self::Spread => j,
            Self::Burst => LABEL_GROUP - 1 - j,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Spread => "RAI",
            Self::Burst => "THEO DOT",
        }
    }
}

struct Window {
    class: u8,
    features: Features,
    new_speed: bool,
}

struct Dataset {
    stream: Vec<Window>,
    heldout_old: Vec<(u8, Features)>,
}

fn load(condition: &str, bearings: &[&'static str], range: (u32, u32)) -> Vec<Record> {
    match paderborn::load_records(&paderborn::data_dir(), condition, bearings, range.0..=range.1) {
        Ok(mut records) => {
            // load_records trả theo từng ổ; xếp lại theo lần đo (sắp xếp ổn định giữ thứ tự ổ, ví dụ K004, KI14, KA16)
            // để luân phiên bản ghi giữa ba lớp
            records.sort_by_key(|r| r.measurement);
            records
        }
        Err(e) => {
            eprintln!("[DEBUG][MAIN][ERROR] {e}");
            std::process::exit(1);
        }
    }
}

/// Các cửa sổ của bản ghi, theo thứ tự thời gian, đã chuẩn hóa bằng hệ số pretrain
fn normalized_windows(records: &[Record]) -> impl Iterator<Item = (u8, Features)> + '_ {
    records.iter().flat_map(|r| {
        r.features.iter().map(move |z| {
            let mut z = *z;
            normalize(&mut z, &pretrained::FEATURE_MEAN, &pretrained::FEATURE_STD);
            (r.class, z)
        })
    })
}

/// Luồng của các ổ `bearings` (mỗi lớp một ổ): N15 lần đo 1-16, rồi N09 lần đo 1-20; N15 lần đo 17-20 để riêng
fn load_dataset(bearings: &[&'static str]) -> Dataset {
    let old = load(paderborn::OLD_SPEED, bearings, OLD_STREAM_MEASUREMENTS);
    let heldout = load(paderborn::OLD_SPEED, bearings, OLD_HELDOUT_MEASUREMENTS);
    let new = load(paderborn::NEW_SPEED, bearings, NEW_STREAM_MEASUREMENTS);

    // Tốc độ cũ trước, tốc độ mới sau
    let mut stream = Vec::new();
    for (new_speed, records) in [(false, &old), (true, &new)] {
        for (class, features) in normalized_windows(records) {
            stream.push(Window { class, features, new_speed });
        }
    }
    Dataset { stream, heldout_old: normalized_windows(&heldout).collect() }
}

/// BA (%) = trung bình đều độ nhạy của ba lớp
fn balanced_accuracy<'a>(results: impl Iterator<Item = &'a (u8, bool)>) -> f32 {
    let mut hit = [0u32; NUM_CLASSES];
    let mut total = [0u32; NUM_CLASSES];
    for &(class, ok) in results {
        total[class as usize] += 1;
        hit[class as usize] += ok as u32;
    }
    let recalls: Vec<f32> = (0..NUM_CLASSES)
        .filter(|&c| total[c] > 0)
        .map(|c| hit[c] as f32 / total[c] as f32)
        .collect();
    if recalls.is_empty() {
        0.0
    } else {
        recalls.iter().sum::<f32>() / recalls.len() as f32 * 100.0
    }
}

/// Độ nhạy (%) của từng lớp; lớp không có cửa sổ nào cho 0
fn class_recalls<'a>(results: impl Iterator<Item = &'a (u8, bool)>) -> [f32; NUM_CLASSES] {
    let mut hit = [0u32; NUM_CLASSES];
    let mut total = [0u32; NUM_CLASSES];
    for &(class, ok) in results {
        total[class as usize] += 1;
        hit[class as usize] += ok as u32;
    }
    core::array::from_fn(|c| hit[c] as f32 / total[c].max(1) as f32 * 100.0)
}

/// Kết quả dự đoán trên các bản ghi tốc độ cũ để riêng, với cập nhật đã tắt
fn heldout_results(head: &Head<NUM_CLASSES>, heldout: &[(u8, Features)]) -> Vec<(u8, bool)> {
    heldout.iter().map(|(c, z)| (*c, head.predict(z) == *c as usize)).collect()
}

#[derive(Default, Clone, Copy)]
struct RunStats {
    ba_new: f32,
    ba_old_before: f32, // BA trên các cửa sổ N15 để riêng, ngay trước khi đổi tốc độ
    forgetting: f32,
    recovery: Option<u32>,
    labels_arrived: u32,
    labels_matched: u32,
    evicted_p: u32,
    dropped_q_full: u32,
    dropped_no_slot: u32,
    first_learns: u32,
    replays: u32,
    wait_sum: u64, // Tổng số bước từ khi nhãn về đến lần học đầu
    left_in_q: u32,
    recall_new: [f32; NUM_CLASSES],       // Độ nhạy từng lớp trên các cửa sổ N09 (mỗi lớp một ổ trong luồng)
    forgetting_class: [f32; NUM_CLASSES], // Độ nhạy từng lớp trên N15 để riêng: trước khi đổi trừ sau thích nghi
}

fn run<const N: usize>(data: &Dataset, labels: LabelSchedule, cfg: Config) -> RunStats {
    let mut learner: Learner<N, NUM_CLASSES> = Learner::new(&cfg);
    learner.head = pretrained::head(); // Trọng số ban đầu chung cho mọi lần chạy
    let n = data.stream.len() as u32;

    let mut st = RunStats::default();
    let mut label_step = vec![0u32; n as usize];
    let mut log = |e: Event| match e.kind {
        EventKind::LabelMatched => {
            st.labels_matched += 1;
            label_step[e.id as usize] = e.step;
        }
        EventKind::EvictedP => st.evicted_p += 1,
        EventKind::DroppedQFull => st.dropped_q_full += 1,
        EventKind::DroppedNoSlot => st.dropped_no_slot += 1,
        EventKind::FirstLearn => {
            st.first_learns += 1;
            st.wait_sum += (e.step - label_step[e.id as usize]) as u64;
        }
        EventKind::Replay => st.replays += 1,
        _ => {}
    };

    let mut pending: Vec<(u32, u32, u8)> = Vec::new(); // (bước nhãn về, ID, nhãn thật)
    let mut labels_arrived = 0u32;
    let mut new_speed_results: Vec<(u8, bool)> = Vec::new();
    let mut recent: VecDeque<(u8, bool)> = VecDeque::with_capacity(SLIDING_WINDOWS + 1);
    // (BA trượt, BA tốc độ cũ để riêng, độ nhạy từng lớp trên tốc độ cũ để riêng)
    let mut before_switch: Option<(f32, f32, [f32; NUM_CLASSES])> = None;
    let mut recovery = None;

    let mut step = 0u32;
    while step < n || !pending.is_empty() {
        if step < n {
            let w = &data.stream[step as usize];
            if w.new_speed && before_switch.is_none() {
                let old = heldout_results(&learner.head, &data.heldout_old);
                before_switch = Some((balanced_accuracy(recent.iter()), balanced_accuracy(old.iter()), class_recalls(old.iter())));
            }
            // Dự đoán được chấm ngay, kể cả khi mẫu bị bỏ sau đó
            let pred = learner.on_window(step, step, w.features, &mut log);
            let result = (w.class, pred == w.class);
            recent.push_back(result);
            if recent.len() > SLIDING_WINDOWS {
                recent.pop_front();
            }
            if w.new_speed {
                new_speed_results.push(result);
                let seen = new_speed_results.len();
                if let Some((target, _, _)) = before_switch {
                    if recovery.is_none() && seen >= SLIDING_WINDOWS && balanced_accuracy(recent.iter()) >= target {
                        recovery = Some(seen as u32);
                    }
                }
            }
            pending.push((step + labels.delay(step), step, w.class));
        }

        // Nhãn đến hạn, xử lý theo ID tăng dần; cuối luồng vẫn giao hết nhãn còn chờ
        let (mut due, rest): (Vec<_>, Vec<_>) = pending.into_iter().partition(|&(t, _, _)| t <= step);
        pending = rest;
        due.sort_by_key(|&(_, id, _)| id);
        for (_, id, label) in due {
            labels_arrived += 1;
            learner.on_label(step, id, label, &mut log);
        }

        learner.train(step, UPDATES_PER_STEP, &mut log);
        step += 1;
    }

    st.ba_new = balanced_accuracy(new_speed_results.iter());
    st.recall_new = class_recalls(new_speed_results.iter());
    let old_after = heldout_results(&learner.head, &data.heldout_old);
    let recalls_after = class_recalls(old_after.iter());
    if let Some((_, ba_old, recalls_before)) = before_switch {
        st.ba_old_before = ba_old;
        st.forgetting = ba_old - balanced_accuracy(old_after.iter());
        st.forgetting_class = core::array::from_fn(|c| recalls_before[c] - recalls_after[c]);
    }
    st.recovery = recovery;
    st.labels_arrived = labels_arrived;
    st.left_in_q = learner.pool.count(SlotState::Q) as u32;
    st
}

#[derive(Clone, Copy)]
struct Variant {
    name: &'static str,
    r_protected: u32,
    replay_enabled: bool,
    learn: bool, // false: tốc độ học 0, giữ nguyên trọng số pretrain (mốc so sánh)
    retention: Retention,
}

/// Mốc so sánh: giữ nguyên trọng số pretrain, không học trên thiết bị
const PRETRAIN_ONLY: Variant =
    Variant { name: "chi pretrain", r_protected: 0, replay_enabled: false, learn: false, retention: Retention::Fifo };

/// Ba phương án nhóm chốt (r = 0%)
const CASES: [Variant; 3] = [
    Variant { name: "C1 tat hoc lai", r_protected: 0, replay_enabled: false, learn: true, retention: Retention::Fifo },
    Variant { name: "C2 R vao truoc", r_protected: 0, replay_enabled: true, learn: true, retention: Retention::Fifo },
    Variant { name: "C3 reservoir", r_protected: 0, replay_enabled: true, learn: true, retention: Retention::Reservoir },
];

/// Chạy một biến thể với mọi hạt giống
fn run_seeds<const N: usize>(data: &Dataset, labels: LabelSchedule, schedule: Schedule, v: &Variant, lr: f32) -> Vec<RunStats> {
    SEEDS
        .iter()
        .map(|&seed| {
            let cfg = Config {
                r_protected: v.r_protected,
                q_max: Q_MAX,
                schedule,
                replay_enabled: v.replay_enabled,
                lr: if v.learn { lr } else { 0.0 },
                seed,
                retention: v.retention,
            };
            run::<N>(data, labels, cfg)
        })
        .collect()
}

fn main() {
    if !pretrained::AVAILABLE {
        eprintln!("[DEBUG][MAIN][ERROR] Chua co trong so pretrain: chay `cargo run --release --bin pretrain` truoc");
        std::process::exit(1);
    }
    if std::env::args().any(|a| a == "--test") {
        // Luật chọn lr áp trên 3 ổ chọn cấu hình, không dùng dữ liệu kiểm thử
        let config = load_dataset(&paderborn::CONFIG_BEARINGS);
        print_data_header("o cau hinh", &paderborn::CONFIG_BEARINGS, &config);
        let lr = choose_learning_rate(&config);
        test_phase(lr);
    } else {
        let data = load_dataset(&paderborn::CONFIG_BEARINGS);
        print_data_header("o cau hinh", &paderborn::CONFIG_BEARINGS, &data);
        if std::env::args().any(|a| a == "--sweep") {
            sweep::<12>(&data);
            sweep::<15>(&data);
            sweep::<20>(&data);
            sweep::<25>(&data);
        } else {
            let lr = choose_learning_rate(&data);
            locked_table(&data, lr);
        }
    }

    println!("\nC1 = hoc tren thiet bi, tat hoc lai | C2 = hoc lai, R vao truoc ra truoc (de cuong) | C3 = hoc lai, R reservoir");
    println!("a+-b = trung binh +- do lech chuan giua {} hat giong.", SEEDS.len());
    println!("BA moi = BA tren cac cua so N09, cham ngay khi cua so den (truoc khi co nhan).");
    println!("BA cu = BA tren cac cua so N15 de rieng, ngay truoc khi doi toc do.");
    println!("Quen = BA cu - BA tren cung cac cua so do sau thich nghi (diem %).");
    println!("Phuc hoi = so cua so toc do moi de BA truot ({SLIDING_WINDOWS} cua so) tro ve muc truoc khi doi.");
}

fn print_data_header(name: &str, bearings: &[&str], data: &Dataset) {
    let n_old = data.stream.iter().filter(|w| !w.new_speed).count();
    println!(
        "[DEBUG][MAIN] PADERBORN, {name} {bearings:?} | luong: {} cua so N15 + {} cua so N09 | {} cua so N15 de rieng | {} hat giong ghep cap",
        n_old,
        data.stream.len() - n_old,
        data.heldout_old.len(),
        SEEDS.len()
    );
}

fn locked_banner(title: &str) {
    println!(
        "\n################ {title} | K = {LOCKED_K} o, M = {} byte | r = 0% | {} | Q toi da {Q_MAX} | {UPDATES_PER_STEP} luot hoc/buoc",
        LOCKED_K * SLOT_SIZE_BYTES + MANAGEMENT_BYTES,
        schedule_name(LOCKED_SCHEDULE)
    );
}

/// Đề cương: chọn tốc độ học trên tập chọn cấu hình, giữ cố định các yếu tố còn lại.
/// Luật (cùng một lr cho cả 3 case để so sánh công bằng): tính BA mới trung bình trên 3 case, 2 lịch trả nhãn
/// và mọi hạt giống; chọn lr NHỎ NHẤT có BA mới cách BA tốt nhất không quá LR_BA_TOLERANCE điểm.
/// Lý do: BA bão hòa khi lr tăng (phần tăng thêm nhỏ hơn độ dao động giữa hạt giống) trong khi mức quên tăng
/// ngày càng nhanh; luật "BA cao nhất" thuần túy luôn đẩy lr ra mép lưới (đã thấy ở lần chạy 0,05-0,4).
fn choose_learning_rate(data: &Dataset) -> f32 {
    locked_banner("CHON TOC DO HOC");
    let mut summary: Vec<(f32, f32, f32)> = Vec::new(); // (lr, BA mới TB, quên TB)
    let mut all_runs: Vec<Vec<RunStats>> = vec![Vec::new(); LR_CANDIDATES.len()];
    for labels in [LabelSchedule::Spread, LabelSchedule::Burst] {
        println!("\n=== CHON LR | LICH TRA NHAN: {} ===", labels.name());
        print_header("Toc do hoc");
        for case in &CASES {
            for (i, lr) in LR_CANDIDATES.iter().enumerate() {
                let runs = run_seeds::<LOCKED_K>(data, labels, LOCKED_SCHEDULE, case, *lr);
                print_row(&format!("lr = {lr}"), case.name, &runs);
                all_runs[i].extend(runs);
            }
        }
    }

    println!("\n=== TONG HOP LR (trung binh 3 case x 2 lich tra nhan x {} hat giong) ===", SEEDS.len());
    for (lr, runs) in LR_CANDIDATES.iter().zip(&all_runs) {
        let n = runs.len() as f32;
        let ba = runs.iter().map(|r| r.ba_new).sum::<f32>() / n;
        let forgetting = runs.iter().map(|r| r.forgetting).sum::<f32>() / n;
        println!("[LR] lr = {lr:<5} | BA moi TB = {ba:6.2} | Quen TB = {forgetting:6.2}");
        summary.push((*lr, ba, forgetting));
    }
    // LR_CANDIDATES tăng dần nên phần tử đầu tiên đạt ngưỡng là lr nhỏ nhất
    let best_ba = summary.iter().map(|s| s.1).fold(f32::MIN, f32::max);
    let threshold = best_ba - LR_BA_TOLERANCE;
    let chosen = summary.iter().find(|s| s.1 >= threshold).map_or(LR_CANDIDATES[0], |s| s.0);
    let edge = if chosen == LR_CANDIDATES[LR_CANDIDATES.len() - 1] { " (NAM O MEP LUOI: nen thu lr lon hon)" } else { "" };
    println!(
        "[LR] BA moi tot nhat = {best_ba:.2} -> chon lr NHO NHAT co BA moi >= {threshold:.2}: lr = {chosen}{edge}"
    );
    chosen
}

/// Bảng chốt trên tập chọn cấu hình: mốc chỉ pretrain và 3 phương án với cấu hình đã khóa
fn locked_table(data: &Dataset, lr: f32) {
    locked_banner(&format!("BANG CHOT TREN TAP CHON CAU HINH, lr = {lr}"));
    for labels in [LabelSchedule::Spread, LabelSchedule::Burst] {
        println!("\n=== BANG CHOT | LICH TRA NHAN: {} ===", labels.name());
        print_header("Lich nhan");
        for v in std::iter::once(&PRETRAIN_ONLY).chain(CASES.iter()) {
            print_row(labels.name(), v.name, &run_seeds::<LOCKED_K>(data, labels, LOCKED_SCHEDULE, v, lr));
        }
    }
}

/// Kết quả một ổ kiểm thử: (ổ, lớp, độ nhạy N09, quên), đã lấy trung bình theo hạt giống
type BearingResult = (&'static str, u8, f32, f32);

/// Giai đoạn kiểm thử (--test): 6 ổ kiểm thử, cấu hình đã khóa, lr do luật chọn trên tập cấu hình.
/// Đề cương: báo kết quả từng ổ và độ phân tán giữa các ổ; tổng hợp các lần chạy (hạt giống) trên cùng ổ
/// trước khi so độ phân tán giữa ổ. BA = trung bình các ổ trong từng lớp rồi trung bình đều ba lớp.
fn test_phase(lr: f32) {
    locked_banner(&format!("KIEM THU 6 O, lr = {lr} (chon tren tap cau hinh)"));
    let variants: Vec<&Variant> = std::iter::once(&PRETRAIN_ONLY).chain(CASES.iter()).collect();
    let label_schedules = [LabelSchedule::Spread, LabelSchedule::Burst];
    // per_bearing[lịch nhãn][phương án] = các (ổ, lớp, độ nhạy N09, quên), đã lấy trung bình theo hạt giống
    let mut per_bearing: Vec<Vec<Vec<BearingResult>>> = vec![vec![Vec::new(); variants.len()]; label_schedules.len()];

    for (stream_name, bearings) in &TEST_STREAMS {
        for (c, b) in bearings.iter().enumerate() {
            assert_eq!(paderborn::class_of(b), Some(c as u8), "luong {stream_name}: o {b} sai thu tu lop");
        }
        let data = load_dataset(bearings);
        print_data_header(&format!("luong kiem thu {stream_name}"), bearings, &data);
        for (li, labels) in label_schedules.iter().enumerate() {
            println!("\n=== LUONG {stream_name} {bearings:?} | LICH TRA NHAN: {} ===", labels.name());
            print_header("Lich nhan");
            let mut details = Vec::new();
            for (vi, v) in variants.iter().enumerate() {
                let runs = run_seeds::<LOCKED_K>(&data, *labels, LOCKED_SCHEDULE, v, lr);
                print_row(labels.name(), v.name, &runs);
                let n = runs.len() as f32;
                let mut cols = Vec::new();
                for (c, bearing) in bearings.iter().enumerate() {
                    let recall = runs.iter().map(|r| r.recall_new[c]).sum::<f32>() / n;
                    let forgetting = runs.iter().map(|r| r.forgetting_class[c]).sum::<f32>() / n;
                    per_bearing[li][vi].push((bearing, c as u8, recall, forgetting));
                    cols.push(format!("{bearing}: N09 {recall:6.2}, quen {forgetting:6.2}"));
                }
                details.push(format!("[O] {:<14} | {}", v.name, cols.join(" | ")));
            }
            println!("[O] Theo tung o (trung binh {} hat giong):", SEEDS.len());
            for line in details {
                println!("{line}");
            }
        }
    }

    println!("\n################ TONG HOP 6 O KIEM THU | K = {LOCKED_K} o, r = 0%, {}, lr = {lr}", schedule_name(LOCKED_SCHEDULE));
    for (li, labels) in label_schedules.iter().enumerate() {
        println!("\n=== TONG HOP | LICH TRA NHAN: {} ===", labels.name());
        println!(
            "{:<14} | {:>10} | {:>12} | {:>10} | {:>12} | {:>24}",
            "Phuong an", "BA moi", "SD giua o", "Quen", "SD giua o", "O thap nhat (N09)"
        );
        for (vi, v) in variants.iter().enumerate() {
            let rows = &per_bearing[li][vi];
            // Trung bình các ổ trong từng lớp, rồi trung bình đều ba lớp
            let by_class = |f: &dyn Fn(&BearingResult) -> f32| {
                (0..NUM_CLASSES as u8)
                    .map(|c| {
                        let vals: Vec<f32> = rows.iter().filter(|r| r.1 == c).map(f).collect();
                        vals.iter().sum::<f32>() / vals.len().max(1) as f32
                    })
                    .sum::<f32>()
                    / NUM_CLASSES as f32
            };
            let sd = |f: &dyn Fn(&BearingResult) -> f32| {
                let n = rows.len() as f32;
                let m = rows.iter().map(f).sum::<f32>() / n;
                (rows.iter().map(|r| (f(r) - m).powi(2)).sum::<f32>() / (n - 1.0).max(1.0)).sqrt()
            };
            let worst = rows.iter().min_by(|a, b| a.2.total_cmp(&b.2)).map_or(String::new(), |r| format!("{} {:.2}", r.0, r.2));
            println!(
                "{:<14} | {:>10.2} | {:>12.2} | {:>10.2} | {:>12.2} | {:>24}",
                v.name,
                by_class(&|r| r.2),
                sd(&|r| r.2),
                by_class(&|r| r.3),
                sd(&|r| r.3),
                worst
            );
        }
    }
    println!("\nDo nhay theo o (N09) = % cua so N09 cua o do duoc doan dung ngay khi den.");
    println!("Quen theo o = do nhay tren cac cua so N15 de rieng cua o do, truoc khi doi toc do tru sau thich nghi.");
    println!("SD giua o = do lech chuan giua 6 o (moi o da lay trung binh {} hat giong).", SEEDS.len());
}

/// Thí nghiệm chọn cấu hình với K = N ô (--sweep)
fn sweep<const N: usize>(data: &Dataset) {
    let memory = N * SLOT_SIZE_BYTES + MANAGEMENT_BYTES;
    assert_eq!(slots_for_budget(memory, MANAGEMENT_BYTES), N);
    println!(
        "\n################ K = {N} o | M = {memory} byte (s = {SLOT_SIZE_BYTES} byte/o, H = {MANAGEMENT_BYTES} byte) | Q toi da {Q_MAX} | {UPDATES_PER_STEP} luot hoc/buoc | lr = {LEARNING_RATE}"
    );

    let r25 = (N * 25 / 100) as u32;
    let r50 = (N * 50 / 100) as u32;
    let fifo = CASES[1];
    let reservoir = CASES[2];
    let variants = [
        PRETRAIN_ONLY,
        CASES[0],
        Variant { name: "r = 0%", ..fifo },
        Variant { name: "r = 25%", r_protected: r25, ..fifo },
        Variant { name: "r = 50%", r_protected: r50, ..fifo },
        Variant { name: "res r = 0%", ..reservoir },
        Variant { name: "res r = 25%", r_protected: r25, ..reservoir },
        Variant { name: "res r = 50%", r_protected: r50, ..reservoir },
    ];

    for labels in [LabelSchedule::Spread, LabelSchedule::Burst] {
        println!("\n=== K = {N} | LICH TRA NHAN: {} ===", labels.name());
        print_header("Lich hoc");
        for schedule in [Schedule::Interleaved, Schedule::QPriority] {
            for v in &variants {
                print_row(schedule_name(schedule), v.name, &run_seeds::<N>(data, labels, schedule, v, LEARNING_RATE));
            }
        }
    }
}

fn schedule_name(schedule: Schedule) -> &'static str {
    match schedule {
        Schedule::Interleaved => "xen ke",
        Schedule::QPriority => "uu tien Q",
    }
}

fn print_header(first: &str) {
    println!(
        "{:<10} | {:<14} | {:>12} | {:>6} | {:>12} | {:>12} | {:>6} | {:>5} | {:>5} | {:>7} | {:>7} | {:>7} | {:>6}",
        first, "Phuong an", "BA moi", "BA cu", "Quen", "Phuc hoi", "Khop", "Bo P", "Bo Q", "Khong o", "Hoc dau", "Hoc lai", "Cho TB"
    );
}

fn print_row(first: &str, variant: &str, runs: &[RunStats]) {
    let n = runs.len() as f32;
    let mean = |f: &dyn Fn(&RunStats) -> f32| runs.iter().map(f).sum::<f32>() / n;
    // Trung bình +- độ lệch chuẩn mẫu (chia n - 1) giữa các hạt giống
    let spread = |f: &dyn Fn(&RunStats) -> f32| {
        let m = mean(f);
        let var = runs.iter().map(|r| (f(r) - m).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
        format!("{m:.2}+-{:.2}", var.sqrt())
    };
    let recovered: Vec<u32> = runs.iter().filter_map(|r| r.recovery).collect();
    let recovery = if recovered.is_empty() {
        "chua phuc hoi".to_string()
    } else {
        let avg = recovered.iter().sum::<u32>() as f32 / recovered.len() as f32;
        format!("{:.0} ({}/{})", avg, recovered.len(), runs.len())
    };
    let match_rate = mean(&|r| r.labels_matched as f32 / r.labels_arrived.max(1) as f32 * 100.0);
    let wait = mean(&|r| r.wait_sum as f32 / r.first_learns.max(1) as f32);
    println!(
        "{:<10} | {:<14} | {:>12} | {:>6.2} | {:>12} | {:>12} | {:>5.1}% | {:>5.0} | {:>5.0} | {:>7.0} | {:>7.0} | {:>7.0} | {:>6.2}",
        first,
        variant,
        spread(&|r| r.ba_new),
        mean(&|r| r.ba_old_before),
        spread(&|r| r.forgetting),
        recovery,
        match_rate,
        mean(&|r| r.evicted_p as f32),
        mean(&|r| r.dropped_q_full as f32),
        mean(&|r| r.dropped_no_slot as f32),
        mean(&|r| r.first_learns as f32),
        mean(&|r| r.replays as f32),
        wait
    );
}
