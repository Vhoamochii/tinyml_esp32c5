// Bộ phát luồng + chấm điểm trên Ubuntu (phía máy tính trong đề cương phương pháp).
// Máy tính giữ nhãn thật, chỉ trả nhãn theo lịch đã định; phía thiết bị dùng tinyml-core, cùng mã với firmware.
//
// DỮ LIỆU GIẢ LẬP: điều kiện "tốc độ cũ" (thay cho N15_M07_F10) rồi "tốc độ mới" (thay cho N09_M07_F10),
// mỗi lớp là một tâm cụm + nhiễu. Thay bằng bộ đọc Paderborn + bộ trích xuất INT8 khi đã có.
// Thí nghiệm: r = 0%, 25%, 50% số ô K và đối chứng tắt học lại; hai lịch trả nhãn (rải, theo đợt);
// hai lịch học (xen kẽ, ưu tiên Q); 5 hạt giống ghép cặp.

use std::collections::VecDeque;

use tinyml_core::event::{Event, EventKind};
use tinyml_core::head::Head;
use tinyml_core::learner::{Config, Learner, MANAGEMENT_BYTES};
use tinyml_core::memsim::{slots_for_budget, SlotState, SLOT_SIZE_BYTES};
use tinyml_core::rng::XorShift32;
use tinyml_core::schedule::Schedule;
use tinyml_core::{NUM_CLASSES, NUM_FEATURES};

type Features = [f32; NUM_FEATURES];

// Giới hạn bộ nhớ: K = floor((M - H) / s)
const MEMORY_BYTES: usize = 2000; // M (tạm, chờ đo RAM khả dụng trên bo)
const K: usize = slots_for_budget(MEMORY_BYTES, MANAGEMENT_BYTES);
const Q_MAX: u32 = 8; // Giới hạn Q (tạm)
const UPDATES_PER_STEP: u32 = 2; // Số lượt học mỗi bước, thay cho giới hạn thời gian mỗi bước (tạm)
const LEARNING_RATE: f32 = 0.1;

// Luồng giả lập
const WINDOWS_PER_RECORD: u32 = 32;
const RECORDS_PER_CLASS: u32 = 10; // Mỗi đoạn tốc độ, luân phiên bản ghi giữa ba lớp
const HELDOUT_RECORDS_PER_CLASS: u32 = 4; // Thay cho các lần đo 17–20 ở tốc độ cũ, không vào luồng học
const NOISE: f32 = 0.6;
const SPEED_SHIFT: f32 = 0.8; // Độ lệch tâm cụm giữa hai tốc độ
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

fn sample(rng: &mut XorShift32, centers: &[Features; NUM_CLASSES], class: usize) -> Features {
    let mut z = [0.0f32; NUM_FEATURES];
    for (zi, ci) in z.iter_mut().zip(centers[class].iter()) {
        *zi = ci + rng.range_f32(-NOISE, NOISE);
    }
    z
}

fn make_dataset(seed: u32) -> Dataset {
    let mut rng = XorShift32::new(seed);
    let mut old = [[0.0f32; NUM_FEATURES]; NUM_CLASSES];
    for center in old.iter_mut() {
        for v in center.iter_mut() {
            *v = rng.range_f32(-1.0, 1.0);
        }
    }
    let mut new = old;
    for center in new.iter_mut() {
        for v in center.iter_mut() {
            *v += rng.range_f32(-SPEED_SHIFT, SPEED_SHIFT);
        }
    }

    // Tốc độ cũ trước, tốc độ mới sau; giữ thứ tự cửa sổ trong từng bản ghi
    let mut stream = Vec::new();
    for (new_speed, centers) in [(false, &old), (true, &new)] {
        for record in 0..RECORDS_PER_CLASS * NUM_CLASSES as u32 {
            let class = (record % NUM_CLASSES as u32) as usize;
            for _ in 0..WINDOWS_PER_RECORD {
                stream.push(Window { class: class as u8, features: sample(&mut rng, centers, class), new_speed });
            }
        }
    }

    let mut heldout_old = Vec::new();
    for class in 0..NUM_CLASSES {
        for _ in 0..HELDOUT_RECORDS_PER_CLASS * WINDOWS_PER_RECORD {
            heldout_old.push((class as u8, sample(&mut rng, &old, class)));
        }
    }
    Dataset { stream, heldout_old }
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

fn run(data: &Dataset, labels: LabelSchedule, cfg: Config) -> RunStats {
    let mut learner: Learner<K, NUM_CLASSES> = Learner::new(&cfg);
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
}

fn main() {
    println!("[DEBUG][MAIN] DU LIEU GIA LAP (chua co Paderborn) | {} hat giong ghep cap", SEEDS.len());
    println!(
        "[DEBUG][MAIN] s = {} byte/o, M = {} byte, H = {} byte -> K = {} o | Q toi da {} | {} luot hoc/buoc | lr = {}",
        SLOT_SIZE_BYTES, MEMORY_BYTES, MANAGEMENT_BYTES, K, Q_MAX, UPDATES_PER_STEP, LEARNING_RATE
    );

    let datasets: Vec<Dataset> = SEEDS.iter().map(|&s| make_dataset(s)).collect();
    let variants = [
        Variant { name: "tat hoc lai", r_protected: 0, replay_enabled: false },
        Variant { name: "r = 0%", r_protected: 0, replay_enabled: true },
        Variant { name: "r = 25%", r_protected: (K * 25 / 100) as u32, replay_enabled: true },
        Variant { name: "r = 50%", r_protected: (K * 50 / 100) as u32, replay_enabled: true },
    ];

    for labels in [LabelSchedule::Spread, LabelSchedule::Burst] {
        println!("\n=== LICH TRA NHAN: {} (trung binh {} hat giong) ===", labels.name(), SEEDS.len());
        println!(
            "{:<10} | {:<11} | {:>6} | {:>6} | {:>12} | {:>6} | {:>5} | {:>5} | {:>7} | {:>7} | {:>7} | {:>6}",
            "Lich hoc", "Bien the", "BA moi", "Quen", "Phuc hoi", "Khop", "Bo P", "Bo Q", "Khong o", "Hoc dau", "Hoc lai", "Cho TB"
        );
        for schedule in [Schedule::Interleaved, Schedule::QPriority] {
            for v in &variants {
                let runs: Vec<RunStats> = SEEDS
                    .iter()
                    .zip(datasets.iter())
                    .map(|(&seed, data)| {
                        let cfg = Config {
                            r_protected: v.r_protected,
                            q_max: Q_MAX,
                            schedule,
                            replay_enabled: v.replay_enabled,
                            lr: LEARNING_RATE,
                            seed,
                        };
                        run(data, labels, cfg)
                    })
                    .collect();
                print_row(schedule, v.name, &runs);
            }
        }
    }
    println!("\nQuen = BA toc do cu ngay truoc khi doi toc do - BA sau thich nghi (diem %).");
    println!("Phuc hoi = so cua so toc do moi de BA truot ({} cua so) tro ve muc truoc khi doi.", SLIDING_WINDOWS);
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
        "{:<10} | {:<11} | {:>6.2} | {:>6.2} | {:>12} | {:>5.1}% | {:>5.0} | {:>5.0} | {:>7.0} | {:>7.0} | {:>7.0} | {:>6.2}",
        schedule_name,
        variant,
        mean(&|r| r.ba_new),
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
