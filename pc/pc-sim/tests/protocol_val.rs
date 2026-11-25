//! Q3: protocol de cuong muc 5-6 tren 3 o chon cau hinh K004/KI14/KA16.
//! C1 sweep r x 2 lich nhan x 5 seed + doi chung Off; C2 khoa r; C4 so lich hoc.
//! Chi chay sau Q2: cargo test --test protocol_val -- --ignored --nocapture

use pc_sim::schedule::ScheduleMode;
use pc_sim::stream_runner::{load_csv, load_quant, run_case_replay, ReplayMode, K};
use std::collections::BTreeMap;
use std::fs;

const VAL_BEARINGS: [&str; 3] = ["K004", "KI14", "KA16"];
const SEEDS: [u32; 5] = [1, 2, 3, 4, 5];

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

#[test]
#[ignore = "can fixtures_val cua Q2; chay -- --ignored"]
fn val_protocol_c1_c2_c4() {
    let csv = std::env::var("FEATURE_CSV")
        .unwrap_or_else(|_| "fixtures_val/features_val.csv".to_string());
    let sc = std::env::var("FEATURE_SCALES")
        .unwrap_or_else(|_| "fixtures/feature_scales.csv".to_string());
    let rows = load_csv(&csv);
    assert!(!rows.is_empty(), "fixtures_val chua co; chay Q2 truoc");
    for r in &rows {
        assert!(VAL_BEARINGS.contains(&r.bearing.as_str()),
            "o {} khong thuoc tap chon cau hinh", r.bearing);
    }
    let q = load_quant(&sc);
    let rs = [0usize, K / 4, K / 2];
    let scheds = [ScheduleMode::Spread, ScheduleMode::Burst];
    let mut c1 = String::from("sched,r_pct,mode,seed,ba_new,forget,match_pct,replay,learned\n");
    let mut agg: BTreeMap<(&'static str, usize, &'static str), (Vec<f64>, Vec<f64>)> = BTreeMap::new();

    for &s in &scheds {
        let sn = match s { ScheduleMode::Spread => "roi", _ => "dot" };
        for &r in &rs {
            for &seed in &SEEDS {
                let res = run_case_replay::<K>(&rows, r, s, "val", &q, seed, ReplayMode::Interleaved);
                let e = agg.entry((sn, r * 100 / K, "inter")).or_default();
                e.0.push(res.ba_new); e.1.push(res.forget);
                c1.push_str(&format!("{},{},inter,{},{:.4},{:.4},{:.4},{},{}\n",
                    sn, r * 100 / K, seed, res.ba_new, res.forget, res.match_pct,
                    res.replay_learns, res.learned_first));
            }
        }
        for &seed in &SEEDS {
            let res = run_case_replay::<K>(&rows, 0, s, "val", &q, seed, ReplayMode::Off);
            let e = agg.entry((sn, 0, "off")).or_default();
            e.0.push(res.ba_new); e.1.push(res.forget);
            c1.push_str(&format!("{},0,off,{},{:.4},{:.4},{:.4},{},{}\n",
                sn, seed, res.ba_new, res.forget, res.match_pct,
                res.replay_learns, res.learned_first));
        }
    }
    fs::create_dir_all("results").unwrap();
    fs::write("results/val_c1.csv", &c1).unwrap();

    let mut lock = String::from("sched,r_pct,ba_mean,forget_mean\n");
    let mut locked: BTreeMap<&'static str, usize> = BTreeMap::new();
    for &s in &scheds {
        let sn = match s { ScheduleMode::Spread => "roi", _ => "dot" };
        let base = mean(&agg[&(sn, 0, "inter")].1);
        let mut cands: Vec<usize> = rs.iter().copied()
            .filter(|&r| mean(&agg[&(sn, r * 100 / K, "inter")].1) <= base + 1e-9)
            .collect();
        assert!(!cands.is_empty(), "khong co r thoa man forget <= r0");
        cands.sort_by(|&a, &b| {
            let ka = (sn, a * 100 / K, "inter");
            let kb = (sn, b * 100 / K, "inter");
            mean(&agg[&kb].0).partial_cmp(&mean(&agg[&ka].0)).unwrap()
                .then(mean(&agg[&ka].1).partial_cmp(&mean(&agg[&kb].1)).unwrap())
                .then(a.cmp(&b))
        });
        let r = cands[0];
        locked.insert(sn, r);
        let bam = mean(&agg[&(sn, r * 100 / K, "inter")].0);
        let fgm = mean(&agg[&(sn, r * 100 / K, "inter")].1);
        lock.push_str(&format!("{},{},{:.4},{:.4}\n", sn, r * 100 / K, bam, fgm));
        println!("[LOCK][{}] r={} ba={:.2} forget={:.2} moc_r0={:.2}", sn, r * 100 / K, bam, fgm, base);
    }
    fs::write("results/val_lock.csv", &lock).unwrap();

    let mut c4 = String::from("sched,r_pct,mode,seed,ba_new,forget,replay,learned\n");
    for &s in &scheds {
        let sn = match s { ScheduleMode::Spread => "roi", _ => "dot" };
        let r = locked[sn];
        for &mode in &[ReplayMode::Interleaved, ReplayMode::QFirst] {
            let mn = match mode { ReplayMode::Interleaved => "inter", _ => "qfirst" };
            let mut bas = Vec::new();
            let mut fgs = Vec::new();
            for &seed in &SEEDS {
                let res = run_case_replay::<K>(&rows, r, s, "val", &q, seed, mode);
                assert!(res.replay_learns <= res.learned_first);
                bas.push(res.ba_new); fgs.push(res.forget);
                c4.push_str(&format!("{},{},{},{},{:.4},{:.4},{},{}\n",
                    sn, r * 100 / K, mn, seed, res.ba_new, res.forget,
                    res.replay_learns, res.learned_first));
            }
            println!("[C4][{}][{}] ba={:.2} forget={:.2}", sn, mn, mean(&bas), mean(&fgs));
        }
    }
    fs::write("results/val_c4.csv", &c4).unwrap();
    println!("[VAL][DONE] ghi results/val_c1.csv, val_lock.csv, val_c4.csv");
}
