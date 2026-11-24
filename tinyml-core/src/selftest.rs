// Hai phép thử bắt buộc trước khi đo trên tín hiệu rung (đề cương, mục 4).
// Chạy giống hệt trên Ubuntu (pc-sim, bin `selftest`) và ESP32-C5 (firmware): mọi dòng in ra phải khớp.

use core::fmt;

use crate::event::{Event, EventKind};
use crate::learner::{Config, Learner};
use crate::memsim::SlotState;
use crate::policy::Retention;
use crate::schedule::Schedule;
use crate::{NUM_CLASSES, NUM_FEATURES};

/// Bốn mẫu XOR: (x1, x2, nhãn)
pub const XOR_DATA: [(f32, f32, u8); 4] = [(0.0, 0.0, 0), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)];
const XOR_LR: f32 = 0.1;
const XOR_EPOCHS: usize = 200;
const XOR_TOL: f32 = 1e-6;

/// z = (x1, x2, x1*x2), 13 phần tử còn lại bằng 0. Thành phần x1*x2 chỉ dùng trong phép thử XOR.
pub fn xor_features(x1: f32, x2: f32) -> [f32; NUM_FEATURES] {
    let mut z = [0.0f32; NUM_FEATURES];
    z[0] = x1;
    z[1] = x2;
    z[2] = x1 * x2;
    z
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

/// Phép thử XOR: dự đoán với điểm cố định, một bước SGD, rồi học 200 lượt. Trả về true nếu đạt cả ba.
pub fn run_xor<O: FnMut(fmt::Arguments<'_>)>(out: &mut O) -> bool {
    // 1. Điểm lớp 0 bằng 0, điểm lớp 1 bằng x1 + x2 - 2*x1*x2 - 0,5
    let mut head = crate::head::Head::<2>::zeros();
    head.weights[1][0] = 1.0;
    head.weights[1][1] = 1.0;
    head.weights[1][2] = -2.0;
    head.bias[1] = -0.5;
    let mut fixed_ok = true;
    let mut preds = [0usize; 4];
    for (k, &(x1, x2, y)) in XOR_DATA.iter().enumerate() {
        preds[k] = head.predict(&xor_features(x1, x2));
        fixed_ok &= preds[k] == y as usize;
    }
    out(format_args!(
        "[XOR][FIXED] du doan={} {} {} {} (mong doi 0 1 1 0) -> {}",
        preds[0],
        preds[1],
        preds[2],
        preds[3],
        verdict(fixed_ok)
    ));

    // 2. W = 0, b = 0, lr = 0,1, một bước SGD với (1,0) mang nhãn 1
    let mut head = crate::head::Head::<2>::zeros();
    head.sgd_step(&xor_features(1.0, 0.0), 1, XOR_LR);
    let near = |v: f32, want: f32| (v - want).abs() <= XOR_TOL;
    let mut others_zero = true;
    for c in 0..2 {
        for f in 1..NUM_FEATURES {
            others_zero &= head.weights[c][f] == 0.0;
        }
    }
    let step_ok = near(head.weights[1][0], 0.05)
        && near(head.bias[1], 0.05)
        && near(head.weights[0][0], -0.05)
        && near(head.bias[0], -0.05)
        && others_zero;
    out(format_args!(
        "[XOR][STEP] w1[0]={:.9} b1={:.9} w0[0]={:.9} b0={:.9} con_lai_bang_0={} -> {}",
        head.weights[1][0],
        head.bias[1],
        head.weights[0][0],
        head.bias[0],
        others_zero,
        verdict(step_ok)
    ));

    // 3. Đặt lại W, b = 0, học bốn mẫu theo thứ tự, 200 lượt; tắt cập nhật rồi dự đoán
    let mut head = crate::head::Head::<2>::zeros();
    for _ in 0..XOR_EPOCHS {
        for &(x1, x2, y) in XOR_DATA.iter() {
            head.sgd_step(&xor_features(x1, x2), y, XOR_LR);
        }
    }
    let mut trained_ok = true;
    for (k, &(x1, x2, y)) in XOR_DATA.iter().enumerate() {
        preds[k] = head.predict(&xor_features(x1, x2));
        trained_ok &= preds[k] == y as usize;
    }
    out(format_args!(
        "[XOR][TRAIN] {} luot: du doan={} {} {} {} -> {}",
        XOR_EPOCHS,
        preds[0],
        preds[1],
        preds[2],
        preds[3],
        verdict(trained_ok)
    ));

    fixed_ok && step_ok && trained_ok
}

/// Số cửa sổ, độ dài epoch và độ trễ nhãn của phép thử ghép nhãn qua ranh giới epoch
pub const EPOCH_TEST_IDS: u32 = 10;
pub const EPOCH_LEN: u32 = 5;
pub const EPOCH_LABEL_DELAY: u32 = 2;

/// Nhãn và đặc trưng giả cho cửa sổ `id` (chỉ để kiểm tra ghép nhãn, không đánh giá độ chính xác)
fn epoch_sample(id: u32) -> (u8, [f32; NUM_FEATURES]) {
    let label = (id % NUM_CLASSES as u32) as u8;
    let mut z = [0.0f32; NUM_FEATURES];
    z[label as usize] = 1.0;
    (label, z)
}

/// Phép thử ghép nhãn qua ranh giới epoch (Bảng 3): 10 cửa sổ ID 0..9, epoch = 5 cửa sổ,
/// nhãn trễ 2 bước, đủ ô và thời gian học. Trả về true nếu đạt mọi điểm đối chiếu.
pub fn run_epoch_matching<O: FnMut(fmt::Arguments<'_>)>(out: &mut O) -> bool {
    const N: usize = 16;
    let cfg = Config {
        r_protected: N as u32,
        q_max: N as u32,
        schedule: Schedule::QPriority,
        replay_enabled: false,
        lr: 0.1,
        seed: 1,
        retention: Retention::Fifo,
    };
    let mut learner: Learner<N, NUM_CLASSES> = Learner::new(&cfg);

    out(format_args!(
        "[EPOCH] {} cua so, epoch = {} cua so, nhan tre {} buoc",
        EPOCH_TEST_IDS, EPOCH_LEN, EPOCH_LABEL_DELAY
    ));

    let mut first_learns = [0u8; EPOCH_TEST_IDS as usize];
    let mut matched_ok = true; // Nhãn của ID i phải khớp đúng ở bước i + 2
    let mut p_after_step4 = [u32::MAX; 2];
    let mut p_after_step4_count = 0;

    let mut log = |e: Event| {
        match e.kind {
            EventKind::FirstLearn if e.id < EPOCH_TEST_IDS => first_learns[e.id as usize] += 1,
            EventKind::LabelMatched => matched_ok &= e.step == e.id + EPOCH_LABEL_DELAY,
            EventKind::LabelMissed | EventKind::LabelDuplicate | EventKind::DroppedQFull => matched_ok = false,
            _ => {}
        }
        out(format_args!("{}", e));
    };

    let last_step = EPOCH_TEST_IDS - 1 + EPOCH_LABEL_DELAY;
    for step in 0..=last_step {
        if step < EPOCH_TEST_IDS {
            let (_, features) = epoch_sample(step);
            learner.on_window(step, step, features, &mut log);
        }
        if step >= EPOCH_LABEL_DELAY {
            let id = step - EPOCH_LABEL_DELAY;
            let (label, _) = epoch_sample(id);
            learner.on_label(step, id, label, &mut log);
        }
        learner.train(step, N as u32, &mut log);

        // Bảng 3, bước 4: hai mẫu ID 3 và 4 còn trong P
        if step == 4 {
            for idx in learner.pool.indices(SlotState::P) {
                if p_after_step4_count < 2 {
                    p_after_step4[p_after_step4_count] = learner.pool.slot(idx).id;
                }
                p_after_step4_count += 1;
            }
        }
    }

    p_after_step4.sort_unstable();
    let p_ok = p_after_step4_count == 2 && p_after_step4 == [3, 4];
    let learned_once = first_learns.iter().all(|&n| n == 1);
    let empty_ok = learner.pool.count(SlotState::P) == 0 && learner.pool.count(SlotState::Q) == 0;

    out(format_args!("[EPOCH][P_SAU_BUOC_4] {} -> {}", p_after_step4_count, verdict(p_ok)));
    out(format_args!("[EPOCH][GHEP_NHAN] -> {}", verdict(matched_ok)));
    out(format_args!(
        "[EPOCH][HOC_LAN_DAU] moi ID dung 1 lan: {} | P,Q rong cuoi luong: {} -> {}",
        learned_once,
        empty_ok,
        verdict(learned_once && empty_ok)
    ));

    p_ok && matched_ok && learned_once && empty_ok
}
