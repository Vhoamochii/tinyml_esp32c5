// Bộ phát luồng + chấm điểm trên Ubuntu (phía máy tính trong đề cương phương pháp).
// Máy tính giữ nhãn thật, chỉ trả nhãn theo lịch đã định; phía thiết bị dùng tinyml-core, cùng mã với firmware.
//
// BƯỚC 4, DỮ LIỆU THẬT: luồng Paderborn trên 3 ổ chọn cấu hình (K004, KI14, KA16).
//   - Đoạn tốc độ cũ N15_M07_F10, lần đo 1-16, rồi đoạn tốc độ mới N09_M07_F10, lần đo 1-20.
//   - Trong mỗi đoạn, luân phiên bản ghi giữa ba lớp, giữ thứ tự 15 cửa sổ trong từng bản ghi.
//   - Lần đo 17-20 ở tốc độ cũ để riêng đo mức quên, không vào luồng học.
//   - Đặc trưng chuẩn hóa và lớp cuối khởi đầu bằng kết quả pretrain (tinyml-core/src/pretrained.rs).
// Mỗi lớp chỉ có một ổ nên BA theo lớp cũng là BA theo ổ.
// Thí nghiệm: quét số ô K (tức dung lượng M), với mỗi K: r = 0%, 25%, 50% số ô K, đối chứng tắt học lại,
// mốc "chỉ pretrain" (không cập nhật), và thí nghiệm phụ "res" = reservoir (R giữ mẫu ngẫu nhiên đều của cả
// lịch sử thay cho vào trước ra trước) với r = 0%, 25%, 50%; hai lịch trả nhãn (rải, theo đợt); hai lịch học (xen kẽ, ưu tiên Q);
// 5 hạt giống ghép cặp.
//   cargo run --release --bin pc-sim

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
const LEARNING_RATE: f32 = 0.1;

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

fn load(condition: &str, range: (u32, u32)) -> Vec<Record> {
    match paderborn::load_records(&paderborn::data_dir(), condition, &paderborn::CONFIG_BEARINGS, range.0..=range.1) {
        Ok(mut records) => {
            // load_records trả theo từng ổ; xếp lại theo lần đo (sắp xếp ổn định giữ thứ tự ổ K004, KI14, KA16)
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

fn load_dataset() -> Dataset {
    let old = load(paderborn::OLD_SPEED, OLD_STREAM_MEASUREMENTS);
    let heldout = load(paderborn::OLD_SPEED, OLD_HELDOUT_MEASUREMENTS);
    let new = load(paderborn::NEW_SPEED, NEW_STREAM_MEASUREMENTS);

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

/// BA (%) trên các bản ghi tốc độ cũ để riêng, với cập nhật đã tắt
fn heldout_ba(head: &Head<NUM_CLASSES>, heldout: &[(u8, Features)]) -> f32 {
    let results: Vec<(u8, bool)> = heldout.iter().map(|(c, z)| (*c, head.predict(z) == *c as usize)).collect();
    balanced_accuracy(results.iter())
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
    let mut before_switch: Option<(f32, f32)> = None; // (BA trượt, BA tốc độ cũ để riêng)
    let mut recovery = None;

    let mut step = 0u32;
    while step < n || !pending.is_empty() {
        if step < n {
            let w = &data.stream[step as usize];
            if w.new_speed && before_switch.is_none() {
                before_switch = Some((
                    balanced_accuracy(recent.iter()),
                    heldout_ba(&learner.head, &data.heldout_old),
                ));
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
                if let Some((target, _)) = before_switch {
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
    st.ba_old_before = before_switch.map_or(0.0, |(_, ba_old)| ba_old);
    st.forgetting = before_switch.map_or(0.0, |(_, ba_old)| ba_old - heldout_ba(&learner.head, &data.heldout_old));
    st.recovery = recovery;
    st.labels_arrived = labels_arrived;
    st.left_in_q = learner.pool.count(SlotState::Q) as u32;
    st
}

struct Variant {
    name: &'static str,
    r_protected: u32,
    replay_enabled: bool,
    learn: bool, // false: tốc độ học 0, giữ nguyên trọng số pretrain (mốc so sánh)
    retention: Retention,
}

fn main() {
    if !pretrained::AVAILABLE {
        eprintln!("[DEBUG][MAIN][ERROR] Chua co trong so pretrain: chay `cargo run --release --bin pretrain` truoc");
        std::process::exit(1);
    }
    let data = load_dataset();
    let n_old = data.stream.iter().filter(|w| !w.new_speed).count();
    println!(
        "[DEBUG][MAIN] PADERBORN, o cau hinh {:?} | luong: {} cua so N15 + {} cua so N09 | {} cua so N15 de rieng | {} hat giong ghep cap",
        paderborn::CONFIG_BEARINGS,
        n_old,
        data.stream.len() - n_old,
        data.heldout_old.len(),
        SEEDS.len()
    );

    sweep::<12>(&data);
    sweep::<15>(&data);
    sweep::<20>(&data);
    sweep::<25>(&data);

    println!("\nBA moi = BA tren cac cua so N09, cham ngay khi cua so den (truoc khi co nhan).");
    println!("BA cu = BA tren cac cua so N15 de rieng, ngay truoc khi doi toc do.");
    println!("Quen = BA cu - BA tren cung cac cua so do sau thich nghi (diem %).");
    println!("Phuc hoi = so cua so toc do moi de BA truot ({} cua so) tro ve muc truoc khi doi.", SLIDING_WINDOWS);
}

/// Toàn bộ bảng thí nghiệm với K = N ô
fn sweep<const N: usize>(data: &Dataset) {
    let memory = N * SLOT_SIZE_BYTES + MANAGEMENT_BYTES;
    assert_eq!(slots_for_budget(memory, MANAGEMENT_BYTES), N);
    println!(
        "\n################ K = {} o | M = {} byte (s = {} byte/o, H = {} byte) | Q toi da {} | {} luot hoc/buoc | lr = {}",
        N, memory, SLOT_SIZE_BYTES, MANAGEMENT_BYTES, Q_MAX, UPDATES_PER_STEP, LEARNING_RATE
    );

    let variants = [
        Variant { name: "chi pretrain", r_protected: 0, replay_enabled: false, learn: false, retention: Retention::Fifo },
        Variant { name: "tat hoc lai", r_protected: 0, replay_enabled: false, learn: true, retention: Retention::Fifo },
        Variant { name: "r = 0%", r_protected: 0, replay_enabled: true, learn: true, retention: Retention::Fifo },
        Variant {
            name: "r = 25%",
            r_protected: (N * 25 / 100) as u32,
            replay_enabled: true,
            learn: true,
            retention: Retention::Fifo,
        },
        Variant {
            name: "r = 50%",
            r_protected: (N * 50 / 100) as u32,
            replay_enabled: true,
            learn: true,
            retention: Retention::Fifo,
        },
        Variant { name: "res r = 0%", r_protected: 0, replay_enabled: true, learn: true, retention: Retention::Reservoir },
        Variant {
            name: "res r = 25%",
            r_protected: (N * 25 / 100) as u32,
            replay_enabled: true,
            learn: true,
            retention: Retention::Reservoir,
        },
        Variant {
            name: "res r = 50%",
            r_protected: (N * 50 / 100) as u32,
            replay_enabled: true,
            learn: true,
            retention: Retention::Reservoir,
        },
    ];

    for labels in [LabelSchedule::Spread, LabelSchedule::Burst] {
        println!("\n=== K = {} | LICH TRA NHAN: {} (trung binh {} hat giong) ===", N, labels.name(), SEEDS.len());
        println!(
            "{:<10} | {:<12} | {:>6} | {:>6} | {:>6} | {:>12} | {:>6} | {:>5} | {:>5} | {:>7} | {:>7} | {:>7} | {:>6}",
            "Lich hoc", "Bien the", "BA moi", "BA cu", "Quen", "Phuc hoi", "Khop", "Bo P", "Bo Q", "Khong o", "Hoc dau", "Hoc lai", "Cho TB"
        );
        for schedule in [Schedule::Interleaved, Schedule::QPriority] {
            for v in &variants {
                let runs: Vec<RunStats> = SEEDS
                    .iter()
                    .map(|&seed| {
                        let cfg = Config {
                            r_protected: v.r_protected,
                            q_max: Q_MAX,
                            schedule,
                            replay_enabled: v.replay_enabled,
                            lr: if v.learn { LEARNING_RATE } else { 0.0 },
                            seed,
                            retention: v.retention,
                        };
                        run::<N>(data, labels, cfg)
                    })
                    .collect();
                print_row(schedule, v.name, &runs);
            }
        }
    }
}

fn print_row(schedule: Schedule, variant: &str, runs: &[RunStats]) {
    let n = runs.len() as f32;
    let mean = |f: &dyn Fn(&RunStats) -> f32| runs.iter().map(f).sum::<f32>() / n;
    let recovered: Vec<u32> = runs.iter().filter_map(|r| r.recovery).collect();
    let recovery = if recovered.is_empty() {
        "chua phuc hoi".to_string()
    } else {
        let avg = recovered.iter().sum::<u32>() as f32 / recovered.len() as f32;
        format!("{:.0} ({}/{})", avg, recovered.len(), runs.len())
    };
    let match_rate = mean(&|r| r.labels_matched as f32 / r.labels_arrived.max(1) as f32 * 100.0);
    let wait = mean(&|r| r.wait_sum as f32 / r.first_learns.max(1) as f32);
    let schedule_name = match schedule {
        Schedule::Interleaved => "xen ke",
        Schedule::QPriority => "uu tien Q",
    };
    println!(
        "{:<10} | {:<12} | {:>6.2} | {:>6.2} | {:>6.2} | {:>12} | {:>5.1}% | {:>5.0} | {:>5.0} | {:>7.0} | {:>7.0} | {:>7.0} | {:>6.2}",
        schedule_name,
        variant,
        mean(&|r| r.ba_new),
        mean(&|r| r.ba_old_before),
        mean(&|r| r.forgetting),
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
