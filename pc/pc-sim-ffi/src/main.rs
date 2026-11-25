//! Binary Case 3: vong lap Rust quan ly pool/lich, SGD/softmax goi C.
//! Ghi results/case3_preds.csv de doi chieu Case 1/2.

use pc_sim::head_linear::LinearHead;
use pc_sim::mem_pool::MemPool;
use pc_sim::schedule::{build, ScheduleMode};
use pc_sim::stream_runner::{balanced_accuracy, load_csv, load_quant, K, LR};
use pc_sim_ffi::{ffi_predict, ffi_sgd_step};

const FFI_SEED: u32 = 0x9E37_79B9;

fn is_reserve(r: &pc_sim::stream_runner::Row) -> bool {
    r.cond == "N15_M07_F10" && r.record >= 17
}

fn main() {
    let csv = std::env::var("FEATURE_CSV")
        .unwrap_or_else(|_| "../pc-sim/fixtures/features_paderborn.csv".to_string());
    let scales = std::env::var("FEATURE_SCALES")
        .unwrap_or_else(|_| "../pc-sim/fixtures/feature_scales.csv".to_string());
    let rows = load_csv(&csv);
    if rows.is_empty() {
        eprintln!("[CASE3][ERROR] CSV rong tai {csv}");
        std::process::exit(2);
    }
    let q = load_quant(&scales);

    let reserve: Vec<&pc_sim::stream_runner::Row> = rows.iter().filter(|r| is_reserve(r)).collect();
    let stream: Vec<&pc_sim::stream_runner::Row> = rows.iter().filter(|r| !is_reserve(r)).collect();

    let mut pool: MemPool<K> = MemPool::new(K / 4);
    let mut head: LinearHead<3> = LinearHead::zeros();
    let labels_true: Vec<u8> = stream.iter().map(|r| r.cls).collect();
    let events = build(stream.len(), ScheduleMode::Spread, &labels_true);
    let mut preds: Vec<(String, u8, u8)> = Vec::new();
    let mut rnd: u32 = FFI_SEED;
    let mut ei = 0usize;
    let mut da_doi = false;
    let mut ba_before = 0.0f64;

    fn eval(reserve: &[&pc_sim::stream_runner::Row], head: &LinearHead<3>,
            q: &pc_sim::head_linear::Quant) -> f64 {
        let mut fp: Vec<(String, u8, u8)> = Vec::new();
        for r in reserve {
            fp.push((r.bearing.clone(), r.cls, ffi_predict(head, r.z, q)));
        }
        balanced_accuracy(&fp) * 100.0
    }

    for (step, row) in stream.iter().enumerate() {
        if row.cond == "N09_M07_F10" && !da_doi {
            da_doi = true;
            let snap = head.snapshot();
            ba_before = eval(&reserve, &snap, &q);
        }
        if row.cond == "N09_M07_F10" {
            let p = ffi_predict(&head, row.z, &q);
            preds.push((row.bearing.clone(), row.cls, p));
        }
        let _ = pool.push(row.z);
        while ei < events.len() && events[ei].0 == step {
            let (_, id, lb) = events[ei];
            let _ = pool.on_label(id, lb);
            ei += 1;
        }
        while let Some(c) = pool.pop_ready() {
            ffi_sgd_step(&mut head, c.z, &q, c.label.max(0) as u8, LR);
            pool.promote_to_replay(c);
            if let Some(rc) = pool.replay_draw(&mut rnd) {
                ffi_sgd_step(&mut head, rc.z, &q, rc.label.max(0) as u8, LR);
            }
        }
    }
    while ei < events.len() {
        let (_, id, lb) = events[ei];
        let _ = pool.on_label(id, lb);
        ei += 1;
        while let Some(c) = pool.pop_ready() {
            ffi_sgd_step(&mut head, c.z, &q, c.label.max(0) as u8, LR);
            pool.promote_to_replay(c);
        }
    }
    let ba_after = eval(&reserve, &head, &q);
    let ba_new = balanced_accuracy(&preds) * 100.0;
    let forget = if da_doi { ba_before - ba_after } else { 0.0 };
    println!(
        "[CASE3][END] ba_new={:.2} ba_truoc={:.2} ba_sau={:.2} forget={:.2} match={:.2} learned={} replay={} miss={}",
        ba_new, ba_before, ba_after, forget,
        100.0 * pool.matched as f64 / stream.len() as f64,
        pool.learned_first, pool.replay_learns, pool.miss_labels
    );

    let mut out = String::from("bearing,class,pred\n");
    for (b, y, p) in &preds {
        out.push_str(&format!("{b},{y},{p}\n"));
    }
    std::fs::create_dir_all("results").unwrap();
    std::fs::write("results/case3_preds.csv", &out).unwrap();
    println!("[CASE3][DONE] ghi results/case3_preds.csv ({} dong du doan)", preds.len());
}
