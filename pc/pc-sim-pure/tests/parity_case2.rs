//! Parity tung ID giua Case 1 (stream_runner) va Case 2 (vong lap doc lap).
//! Cung seed, cung thu tu CSV, cung r, cung lich -> du doan phai khop.

use pc_sim::head_linear::{LinearHead, Quant};
use pc_sim::mem_pool::MemPool;
use pc_sim::schedule::{build, ScheduleMode};
use pc_sim::stream_runner::{balanced_accuracy, load_csv, load_quant, run_case_seeded, DEFAULT_SEED, K, LR};

const PURE_SEED: u32 = DEFAULT_SEED;

fn is_reserve(r: &pc_sim::stream_runner::Row) -> bool {
    r.cond == "N15_M07_F10" && r.record >= 17
}

#[test]
fn parity_case1_case2() {
    let csv = std::env::var("FEATURE_CSV")
        .unwrap_or_else(|_| "../pc-sim/fixtures/features_paderborn.csv".to_string());
    let scales = std::env::var("FEATURE_SCALES")
        .unwrap_or_else(|_| "../pc-sim/fixtures/feature_scales.csv".to_string());
    
    let rows = load_csv(&csv);
    if rows.is_empty() { std::process::exit(2); }
    let q = load_quant(&scales);

    let a = run_case_seeded::<K>(&rows, K / 4, ScheduleMode::Spread, "Case1", &q, PURE_SEED);

    let reserve: Vec<&pc_sim::stream_runner::Row> = rows.iter().filter(|r| is_reserve(r)).collect();
    let stream: Vec<&pc_sim::stream_runner::Row> = rows.iter().filter(|r| !is_reserve(r)).collect();
    let mut pool: MemPool<K> = MemPool::new(K / 4);
    let mut head: LinearHead<3> = LinearHead::zeros();
    let labels_true: Vec<u8> = stream.iter().map(|r| r.cls).collect();
    let events = build(stream.len(), ScheduleMode::Spread, &labels_true);
    let mut preds: Vec<(String, u8, u8)> = Vec::new();
    let mut rnd: u32 = PURE_SEED;
    let mut ei = 0usize;
    let mut da_doi = false;
    let mut ba_before = 0.0f64;

    fn eval(reserve: &[&pc_sim::stream_runner::Row], head: &LinearHead<3>, q: &Quant) -> f64 {
        let mut fp: Vec<(String, u8, u8)> = Vec::new();
        for r in reserve { fp.push((r.bearing.clone(), r.cls, head.predict(r.z, q))); }
        balanced_accuracy(&fp) * 100.0
    }

    for (step, row) in stream.iter().enumerate() {
        if row.cond == "N09_M07_F10" && !da_doi {
            da_doi = true;
            let snap = head.snapshot();
            ba_before = eval(&reserve, &snap, &q);
        }
        if row.cond == "N09_M07_F10" {
            let p = head.predict(row.z, &q);
            preds.push((row.bearing.clone(), row.cls, p));
        }
        let _ = pool.push(row.z);
        while ei < events.len() && events[ei].0 == step {
            let (_, id, lb) = events[ei];
            let _ = pool.on_label(id, lb);
            ei += 1;
        }
        while let Some(c) = pool.pop_ready() {
            head.sgd_step(c.z, &q, c.label.max(0) as u8, LR);
            pool.promote_to_replay(c);
            if let Some(rc) = pool.replay_draw(&mut rnd) {
                head.sgd_step(rc.z, &q, rc.label.max(0) as u8, LR);
            }
        }
    }
    while ei < events.len() {
        let (_, id, lb) = events[ei];
        let _ = pool.on_label(id, lb);
        ei += 1;
        while let Some(c) = pool.pop_ready() {
            head.sgd_step(c.z, &q, c.label.max(0) as u8, LR);
            pool.promote_to_replay(c);
        }
    }

    let ba_after = eval(&reserve, &head, &q);
    let preds_b: Vec<(String, u8, u8)> = preds;
    
    println!(
        "[TEST][PARITY][CASE2] expected_preds={} actual_preds={} so_lech={}",
        a.preds.len(), preds_b.len(),
        a.preds.iter().zip(preds_b.iter()).filter(|(x, y)| x != y).count()
    );
    assert_eq!(a.preds.len(), preds_b.len());
    assert!(a.preds == preds_b, "Case 2 va Case 1 lech tung ID");
    let fg_b = if da_doi { ba_before - ba_after } else { 0.0 };
    assert!((a.forget - fg_b).abs() < 1e-9, "forget lech: {} vs {}", a.forget, fg_b);
    println!("[TEST][PARITY][CASE2] expected=khop_tuyet_doi actual=khop PASS");
}
