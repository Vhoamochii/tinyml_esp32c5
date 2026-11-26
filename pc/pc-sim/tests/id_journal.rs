//! E1: nhat ky theo ID phai khop cac tong da nghiem thu va ghi du ly do bo mau.
//! Chi cac chay co tag "JRN:" moi sinh file, tranh phinh dia o sweep/pressure.

use pc_sim::head_linear::Quant;
use pc_sim::schedule::ScheduleMode;
use pc_sim::stream_runner::{load_csv, load_quant, run_case_replay, Row, ReplayMode};

fn setup() -> (Vec<Row>, Quant) {
    (load_csv("fixtures/features_paderborn.csv"), load_quant("fixtures/feature_scales.csv"))
}

fn fname(tag: &str, mn: &str, seed: u32) -> String {
    let san: String = tag.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    format!("results/id_journal_{}_{}_{}.csv", san, mn, seed)
}

struct J { rows: usize, learned: usize, replays: u64, reclaim_p: usize, push_full: usize, pred_present: usize }

fn read_j(path: &str) -> J {
    let txt = std::fs::read_to_string(path).expect("thieu file nhat ky ID");
    let mut j = J { rows: 0, learned: 0, replays: 0, reclaim_p: 0, push_full: 0, pred_present: 0 };
    for line in txt.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 9, "dong nhat ky sai so cot: {line}");
        j.rows += 1;
        if f[6] != "4294967295" { j.learned += 1; }
        j.replays += f[7].parse::<u64>().unwrap();
        match f[8] { "reclaim_p" => j.reclaim_p += 1, "push_full" => j.push_full += 1, _ => {} }
        if !f[4].is_empty() { j.pred_present += 1; }
    }
    j
}

/// Happy: K=256 khong bo mau; tong journal khop learned/replay cua engine.
#[test]
fn journal_happy_k256_no_drops() {
    let (rows, q) = setup();
    let r = run_case_replay::<256>(&rows, 64, ScheduleMode::Spread, "JRN:happy", &q, 1, ReplayMode::Interleaved);
    let j = read_j(&fname("JRN:happy", "inter", 1));
    assert_eq!(j.rows, r.n_stream);
    assert_eq!(j.learned as u32, r.learned_first);
    assert_eq!(j.replays as u32, r.replay_learns);
    assert_eq!(j.reclaim_p + j.push_full, 0);
    assert_eq!(j.pred_present, j.rows); // ke ca mau bi bo van co du doan (de cuong muc 3)
    println!("[TEST][ID_JOURNAL][HAPPY] rows={} learned={} replays={} drops=0 PASS", j.rows, j.learned, j.replays);
}

/// Edge: K=16 lich dot ep thu hoi P; moi nhan mo coi khop dung mot su kien reclaim_p.
#[test]
fn journal_edge_k16_burst_eviction_reasons() {
    let (rows, q) = setup();
    let r = run_case_replay::<16>(&rows, 8, ScheduleMode::Burst, "JRN:edge", &q, 1, ReplayMode::Interleaved);
    let j = read_j(&fname("JRN:edge", "inter", 1));
    assert!(r.miss_labels > 0);
    assert_eq!(j.reclaim_p as u32, r.miss_labels);
    assert_eq!(j.push_full, 0);
    assert_eq!(j.rows, r.n_stream);
    println!("[TEST][ID_JOURNAL][EDGE] reclaim_p={} == miss={} PASS", j.reclaim_p, r.miss_labels);
}

/// QFirst: so luot hoc lai trong journal khop engine va bi chan boi so lan hoc dau.
#[test]
fn journal_qfirst_replay_consistency() {
    let (rows, q) = setup();
    let r = run_case_replay::<256>(&rows, 64, ScheduleMode::Spread, "JRN:qf", &q, 1, ReplayMode::QFirst);
    let j = read_j(&fname("JRN:qf", "qfirst", 1));
    assert_eq!(j.replays as u32, r.replay_learns);
    assert!(j.replays as u32 <= r.learned_first);
    println!("[TEST][ID_JOURNAL][QFIRST] replays={} <= learned={} PASS", j.replays, r.learned_first);
}
