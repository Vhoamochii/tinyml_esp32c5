// Kiểm thử tinyml-core theo đề cương phương pháp (src/De_cuong_phuong_phap_Rust_ESP32-C5.docx)
use tinyml_core::event::{Event, EventKind};
use tinyml_core::learner::{Config, Learner, MANAGEMENT_BYTES};
use tinyml_core::memsim::{slots_for_budget, SlotState, SLOT_SIZE_BYTES};
use tinyml_core::policy::Retention;
use tinyml_core::rng::XorShift32;
use tinyml_core::schedule::Schedule;
use tinyml_core::{selftest, NUM_CLASSES, NUM_FEATURES};

fn cfg(r_protected: u32, q_max: u32, schedule: Schedule, replay_enabled: bool) -> Config {
    Config { r_protected, q_max, schedule, replay_enabled, lr: 0.1, seed: 7, retention: Retention::Fifo }
}

fn z() -> [f32; NUM_FEATURES] {
    [0.0; NUM_FEATURES]
}

/// Các sự kiện (bỏ PRED và ADMIT_P cho gọn) dưới dạng (id, loại)
fn kinds(events: &[Event]) -> Vec<(u32, EventKind)> {
    events
        .iter()
        .filter(|e| !matches!(e.kind, EventKind::Predicted(_) | EventKind::Admitted))
        .map(|e| (e.id, e.kind))
        .collect()
}

#[test]
fn test_slot_size_and_budget() {
    println!("[TEST][MEMSIM][SlotSize] Bat dau...");
    // 16 f32 (64) + id, order (8) + label, state (2) = 74 byte, căn chỉnh 4 -> 76
    assert_eq!(SLOT_SIZE_BYTES, 76);
    // K = floor((M - H) / s)
    assert_eq!(slots_for_budget(2000, 0), 26);
    assert_eq!(slots_for_budget(2000, MANAGEMENT_BYTES), (2000 - MANAGEMENT_BYTES) / 76);
    assert_eq!(slots_for_budget(10, 100), 0);
    println!("[TEST][MEMSIM][SlotSize] H = {} byte -> PASS", MANAGEMENT_BYTES);
}

#[test]
fn test_reclaim_order_free_then_r_then_p() {
    println!("[TEST][POLICY][ReclaimOrder] Bat dau...");
    // 3 ô, r = 1
    let mut l: Learner<3, NUM_CLASSES> = Learner::new(&cfg(1, 3, Schedule::QPriority, false));
    let mut ev = Vec::new();
    let mut log = |e: Event| ev.push(e);

    // Ba ô trống được dùng trước
    for id in 1..=3 {
        l.on_window(id, id, z(), &mut log);
    }
    // ID 1 và 2 được học -> R = {1, 2}, P = {3}
    l.on_label(4, 1, 0, &mut log);
    l.on_label(4, 2, 0, &mut log);
    l.train(4, 10, &mut log);
    // Mảng đầy, R có 2 > r = 1 -> thu hồi R vào sớm nhất (ID 1)
    l.on_window(5, 4, z(), &mut log);
    // R còn 1 = r -> R được bảo vệ, thu hồi P chờ lâu nhất (ID 3)
    l.on_window(6, 5, z(), &mut log);
    // Nhãn của ID 3 về sau khi ô đã bị thu hồi -> ghép trượt
    l.on_label(7, 3, 0, &mut log);

    let k = kinds(&ev);
    assert_eq!(
        k,
        vec![
            (1, EventKind::LabelMatched),
            (2, EventKind::LabelMatched),
            (1, EventKind::FirstLearn),
            (2, EventKind::FirstLearn),
            (1, EventKind::EvictedR),
            (3, EventKind::EvictedP),
            (3, EventKind::LabelMissed),
        ]
    );
    println!("[TEST][POLICY][ReclaimOrder] PASS");
}

#[test]
fn test_q_protected_drop_no_slot() {
    println!("[TEST][POLICY][QProtected] Bat dau...");
    // 2 ô, r = 0; cả hai ô ở Q -> mẫu mới không có ô
    let mut l: Learner<2, NUM_CLASSES> = Learner::new(&cfg(0, 2, Schedule::QPriority, false));
    let mut ev = Vec::new();
    let mut log = |e: Event| ev.push(e);
    l.on_window(0, 1, z(), &mut log);
    l.on_window(0, 2, z(), &mut log);
    l.on_label(1, 1, 0, &mut log);
    l.on_label(1, 2, 0, &mut log);
    l.on_window(2, 3, z(), &mut log);

    assert_eq!(kinds(&ev).last(), Some(&(3, EventKind::DroppedNoSlot)));
    assert_eq!(l.pool.count(SlotState::Q), 2);
    println!("[TEST][POLICY][QProtected] PASS");
}

#[test]
fn test_q_full_drops_sample() {
    println!("[TEST][POLICY][QFull] Bat dau...");
    // Q tối đa 1 mẫu
    let mut l: Learner<4, NUM_CLASSES> = Learner::new(&cfg(0, 1, Schedule::QPriority, false));
    let mut ev = Vec::new();
    let mut log = |e: Event| ev.push(e);
    l.on_window(0, 1, z(), &mut log);
    l.on_window(0, 2, z(), &mut log);
    l.on_label(1, 1, 0, &mut log);
    l.on_label(1, 2, 0, &mut log);

    assert_eq!(
        kinds(&ev),
        vec![(1, EventKind::LabelMatched), (2, EventKind::LabelMatched), (2, EventKind::DroppedQFull)]
    );
    assert_eq!(l.pool.count(SlotState::Q), 1);
    assert_eq!(l.pool.count(SlotState::Free), 3);
    println!("[TEST][POLICY][QFull] PASS");
}

#[test]
fn test_duplicate_label_no_second_first_learn() {
    println!("[TEST][POLICY][DuplicateLabel] Bat dau...");
    let mut l: Learner<4, NUM_CLASSES> = Learner::new(&cfg(4, 4, Schedule::QPriority, false));
    let mut ev = Vec::new();
    let mut log = |e: Event| ev.push(e);
    l.on_window(0, 1, z(), &mut log);
    l.on_label(1, 1, 0, &mut log);
    l.on_label(1, 1, 0, &mut log); // Trùng khi mẫu đang ở Q
    l.train(1, 10, &mut log);
    l.on_label(2, 1, 0, &mut log); // Trùng khi mẫu đã ở R
    l.train(2, 10, &mut log);

    let first_learns = ev.iter().filter(|e| e.kind == EventKind::FirstLearn).count();
    let duplicates = ev.iter().filter(|e| e.kind == EventKind::LabelDuplicate).count();
    assert_eq!(first_learns, 1);
    assert_eq!(duplicates, 2);
    println!("[TEST][POLICY][DuplicateLabel] PASS");
}

#[test]
fn test_out_of_order_labels_learn_in_q_order() {
    println!("[TEST][POLICY][OutOfOrder] Bat dau...");
    let mut l: Learner<4, NUM_CLASSES> = Learner::new(&cfg(4, 4, Schedule::QPriority, false));
    let mut ev = Vec::new();
    let mut log = |e: Event| ev.push(e);
    for id in 1..=3 {
        l.on_window(0, id, z(), &mut log);
    }
    // Nhãn về khác thứ tự: 3, 1, 2 -> Q học đúng theo thứ tự vào hàng
    l.on_label(1, 3, 0, &mut log);
    l.on_label(2, 1, 0, &mut log);
    l.on_label(3, 2, 0, &mut log);
    l.train(3, 10, &mut log);

    let order: Vec<u32> = ev.iter().filter(|e| e.kind == EventKind::FirstLearn).map(|e| e.id).collect();
    assert_eq!(order, vec![3, 1, 2]);
    println!("[TEST][POLICY][OutOfOrder] PASS");
}

/// Loại lượt học (N = mới, R = học lại) của bước thứ hai trong kịch bản chung
fn schedule_pattern(schedule: Schedule) -> Vec<char> {
    let mut l: Learner<8, NUM_CLASSES> = Learner::new(&cfg(8, 8, schedule, true));
    let mut ignore = |_: Event| {};
    for id in 1..=4 {
        l.on_window(0, id, z(), &mut ignore);
    }
    // Bước 1: học ID 1 và 2 -> R = {1, 2}
    l.on_label(1, 1, 0, &mut ignore);
    l.on_label(1, 2, 0, &mut ignore);
    l.train(1, 2, &mut ignore);
    // Bước 2: nhãn của ID 3 và 4 về cùng lúc, đủ thời gian cho 4 lượt học
    let mut ev = Vec::new();
    let mut log = |e: Event| ev.push(e);
    l.on_label(2, 3, 0, &mut log);
    l.on_label(2, 4, 0, &mut log);
    l.train(2, 4, &mut log);
    ev.iter()
        .filter_map(|e| match e.kind {
            EventKind::FirstLearn => Some('N'),
            EventKind::Replay => Some('R'),
            _ => None,
        })
        .collect()
}

#[test]
fn test_schedule_interleaved_vs_q_priority() {
    println!("[TEST][SCHEDULE][Patterns] Bat dau...");
    // Xen kẽ: học lại một mẫu cũ sau mỗi lần học mẫu mới
    assert_eq!(schedule_pattern(Schedule::Interleaved), vec!['R', 'N', 'R', 'N']);
    // Ưu tiên Q: chỉ học lại khi Q rỗng
    assert_eq!(schedule_pattern(Schedule::QPriority), vec!['N', 'N', 'R', 'R']);
    println!("[TEST][SCHEDULE][Patterns] PASS");
}

#[test]
fn test_replay_skips_last_learned_and_can_be_disabled() {
    println!("[TEST][SCHEDULE][ReplayRules] Bat dau...");
    // R chỉ có đúng mẫu vừa học -> không được học lại ngay
    let mut l: Learner<4, NUM_CLASSES> = Learner::new(&cfg(4, 4, Schedule::QPriority, true));
    let mut ev = Vec::new();
    let mut log = |e: Event| ev.push(e);
    l.on_window(0, 1, z(), &mut log);
    l.on_label(1, 1, 0, &mut log);
    l.train(1, 5, &mut log);
    assert_eq!(ev.iter().filter(|e| e.kind == EventKind::Replay).count(), 0);

    // Đối chứng tắt học lại: không có lượt học lại nào
    let mut l: Learner<8, NUM_CLASSES> = Learner::new(&cfg(8, 8, Schedule::QPriority, false));
    let mut ev = Vec::new();
    let mut log = |e: Event| ev.push(e);
    for id in 1..=4 {
        l.on_window(0, id, z(), &mut log);
        l.on_label(0, id, 0, &mut log);
    }
    l.train(0, 20, &mut log);
    assert_eq!(ev.iter().filter(|e| e.kind == EventKind::Replay).count(), 0);
    println!("[TEST][SCHEDULE][ReplayRules] PASS");
}

/// Chạy 1000 cửa sổ qua K = 16 ô, nhãn về ngay, học 1 lượt mỗi bước; trả về các ID còn trong R
fn r_ids_after_stream(retention: Retention) -> Vec<u32> {
    let mut learner: Learner<16, NUM_CLASSES> =
        Learner::new(&Config { retention, ..cfg(0, 4, Schedule::QPriority, false) });
    let mut ignore = |_e: Event| {};
    for t in 0..1000 {
        learner.on_window(t, t, z(), &mut ignore);
        learner.on_label(t, t, 0, &mut ignore);
        learner.train(t, 1, &mut ignore);
    }
    let pool = &learner.pool;
    let mut ids: Vec<u32> = pool.indices(SlotState::R).map(|i| pool.slot(i).id).collect();
    ids.sort();
    ids
}

#[test]
fn test_retention_fifo_vs_reservoir() {
    println!("[TEST][POLICY][Retention] Bat dau...");
    // Vào trước ra trước: R chỉ còn các mẫu mới nhất
    let fifo = r_ids_after_stream(Retention::Fifo);
    assert_eq!(fifo, (984..1000).collect::<Vec<u32>>(), "fifo: {fifo:?}");
    // Reservoir: R rải khắp lịch sử (mô phỏng Python với hạt giống 7: 7/16 ID nhỏ hơn 500)
    let reservoir = r_ids_after_stream(Retention::Reservoir);
    assert_eq!(reservoir.len(), 16);
    let old = reservoir.iter().filter(|&&id| id < 500).count();
    assert!(old >= 5, "reservoir chi giu {old} mau cu: {reservoir:?}");
    println!("[TEST][POLICY][Retention] fifo {fifo:?} | reservoir {reservoir:?} -> PASS");
}

#[test]
fn test_selftest_xor() {
    println!("[TEST][SELFTEST][XOR] Bat dau...");
    let mut out = |a: std::fmt::Arguments<'_>| println!("{a}");
    assert!(selftest::run_xor(&mut out));
    println!("[TEST][SELFTEST][XOR] PASS");
}

#[test]
fn test_selftest_epoch_matching() {
    println!("[TEST][SELFTEST][Epoch] Bat dau...");
    let mut out = |a: std::fmt::Arguments<'_>| println!("{a}");
    assert!(selftest::run_epoch_matching(&mut out));
    println!("[TEST][SELFTEST][Epoch] PASS");
}

#[test]
fn test_rng_deterministic() {
    println!("[TEST][RNG][Deterministic] Bat dau...");
    let mut a = XorShift32::new(42);
    let mut b = XorShift32::new(42);
    for _ in 0..100 {
        assert_eq!(a.next_u32(), b.next_u32());
        let v = a.range_f32(-1.0, 1.0);
        assert!((-1.0..1.0).contains(&v));
        b.range_f32(-1.0, 1.0);
    }
    println!("[TEST][RNG][Deterministic] PASS");
}
