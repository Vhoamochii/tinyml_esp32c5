//! Q1: kiem chung ba lich hoc lai. Day la test dung dan, KHONG phai so bao cao.
//! Interleaved phai tai lap ban cu; Off phai 0 luot hoc lai; QFirst bi chan
//! boi so lan hoc dau (moi lan hoc dau toi da 1 credit).

use pc_sim::head_linear::Quant;
use pc_sim::schedule::ScheduleMode;
use pc_sim::stream_runner::{
    load_csv, load_quant, run_case_replay, run_case_seeded, Row, ReplayMode, K,
};

fn setup() -> (Vec<Row>, Quant) {
    let csv = std::env::var("FEATURE_CSV")
        .unwrap_or_else(|_| "fixtures/features_paderborn.csv".to_string());
    let sc = std::env::var("FEATURE_SCALES")
        .unwrap_or_else(|_| "fixtures/feature_scales.csv".to_string());
    (load_csv(&csv), load_quant(&sc))
}

#[test]
fn interleaved_equals_legacy() {
    let (rows, q) = setup();
    let a = run_case_seeded::<K>(&rows, K / 4, ScheduleMode::Spread, "legacy", &q, 1);
    let b = run_case_replay::<K>(&rows, K / 4, ScheduleMode::Spread, "inter", &q, 1, ReplayMode::Interleaved);
    assert_eq!(a.preds, b.preds);
    assert_eq!(a.replay_learns, b.replay_learns);
    assert!((a.forget - b.forget).abs() < 1e-12);
    println!("[TEST][REPLAY][INTERLEAVED] preds={} replay={} khop_ban_cu PASS",
        b.preds.len(), b.replay_learns);
}

#[test]
fn off_is_true_control() {
    let (rows, q) = setup();
    let a = run_case_seeded::<K>(&rows, K / 4, ScheduleMode::Spread, "inter", &q, 1);
    let off = run_case_replay::<K>(&rows, K / 4, ScheduleMode::Spread, "off", &q, 1, ReplayMode::Off);
    assert_eq!(off.replay_learns, 0);
    assert_eq!(off.learned_first, a.learned_first);
    assert_ne!(off.preds, a.preds);
    println!("[TEST][REPLAY][OFF] replay=0 learned={} bang_interleaved preds_khac PASS",
        off.learned_first);
}

#[test]
fn r0_is_not_off() {
    let (rows, q) = setup();
    let z = run_case_seeded::<K>(&rows, 0, ScheduleMode::Spread, "r0", &q, 1);
    let off = run_case_replay::<K>(&rows, 0, ScheduleMode::Spread, "off", &q, 1, ReplayMode::Off);
    assert!(z.replay_learns > 0);
    assert_ne!(z.preds, off.preds);
    println!("[TEST][REPLAY][R0_VS_OFF] r0_replay={} off_replay=0 preds_khac PASS",
        z.replay_learns);
}

#[test]
fn qfirst_credit_bounds_and_determinism() {
    let (rows, q) = setup();
    let q1 = run_case_replay::<K>(&rows, K / 4, ScheduleMode::Spread, "qfirst", &q, 1, ReplayMode::QFirst);
    let q2 = run_case_replay::<K>(&rows, K / 4, ScheduleMode::Spread, "qfirst", &q, 1, ReplayMode::QFirst);
    assert!(q1.replay_learns > 0);
    assert!(q1.replay_learns <= q1.learned_first);
    assert_eq!(q1.preds, q2.preds);
    println!("[TEST][REPLAY][QFIRST] replay={} <= learned={} determinism PASS",
        q1.replay_learns, q1.learned_first);
}
