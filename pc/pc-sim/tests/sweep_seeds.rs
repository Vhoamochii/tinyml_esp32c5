//! E2+E2b: quet chinh r=0/25/50 x 2 lich x 5 seed (K=256),
//! va nhanh AP LUC K=16/32 x r x 2 lich x 2 seed de r phan biet (F-A).
//! Phuc hoi: first-touch + ben vung SUSTAIN_W cua so lien tiep (E1e).
//! Test standalone RollingBa/percentile: happy / edge / error.

use pc_sim::schedule::ScheduleMode;
use pc_sim::stream_runner::{
    load_csv, load_quant, percentile, run_case_seeded, RollingBa, K, ROLL_W,
};
use std::collections::BTreeMap;

fn mean_sd(v: &[f64]) -> (f64, f64) {
    if v.is_empty() { return (0.0, 0.0); }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    if v.len() < 2 { return (m, 0.0); }
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64;
    (m, var.sqrt())
}

fn cycle6(i: usize) -> u8 {
    [0u8, 0, 2, 2, 1, 1][i % 6]
}

#[test]
fn rolling_ba_happy() {
    let mut r = RollingBa::new();
    for i in 0..ROLL_W { r.push(cycle6(i), true); }
    let ba = r.ba().expect("du 3 lop phai tra Some");
    println!("[TEST][ROLLING][happy] expected=100.00 actual={:.2} {}", ba,
        if (ba - 100.0).abs() < 1e-9 { "PASS" } else { "FAIL" });
    assert!(r.is_full() && (ba - 100.0).abs() < 1e-9);
}

#[test]
fn rolling_ba_edge_missing_class() {
    let mut r = RollingBa::new();
    assert!(r.ba().is_none());
    for _ in 0..10 { r.push(0, true); }
    let got = r.ba();
    println!("[TEST][ROLLING][edge] expected=None actual={:?} {}", got,
        if got.is_none() { "PASS" } else { "FAIL" });
    assert!(got.is_none());
}

#[test]
fn rolling_ba_error_and_wrap() {
    let mut r = RollingBa::new();
    r.push(9, true);
    assert!(r.ba().is_none() && !r.is_full());
    for i in 0..ROLL_W { r.push(cycle6(i), true); }
    r.push(0, false); r.push(0, false);
    r.push(2, true); r.push(2, true); r.push(1, true); r.push(1, true);
    let ba = r.ba().unwrap();
    let want = (90.0 + 100.0 + 100.0) / 3.0;
    println!("[TEST][ROLLING][wrap] expected={:.4} actual={:.4} {}", want, ba,
        if (ba - want).abs() < 1e-6 { "PASS" } else { "FAIL" });
    assert!((ba - want).abs() < 1e-6);
    assert_eq!(percentile(&[], 95), 0);
    let v: Vec<u64> = (1..=10).collect();
    assert_eq!(percentile(&v, 50), 6);
    assert_eq!(percentile(&v, 95), 10);
    println!("[TEST][PERCENTILE][edge] expected=0/6/10 actual=0/6/10 PASS");
}

struct Agg {
    ba: Vec<f64>, fg: Vec<f64>, mt: Vec<f64>, p95: Vec<f64>, ov: Vec<f64>,
    l50: Vec<f64>, l95: Vec<f64>, rec_ok: usize, rec_win: Vec<f64>,
    sus_ok: usize, sus_win: Vec<f64>,
    bear: BTreeMap<String, Vec<f64>>,
}

impl Agg {
    fn new() -> Self {
        Self { ba: vec![], fg: vec![], mt: vec![], p95: vec![], ov: vec![],
               l50: vec![], l95: vec![], rec_ok: 0, rec_win: vec![],
               sus_ok: 0, sus_win: vec![], bear: BTreeMap::new() }
    }
}

#[test]
fn sweep_r_seeds() {
    let path = std::env::var("FEATURE_CSV")
        .unwrap_or_else(|_| "fixtures/features_paderborn.csv".to_string());
    let scales = std::env::var("FEATURE_SCALES")
        .unwrap_or_else(|_| "fixtures/feature_scales.csv".to_string());
    let rows = load_csv(&path);
    if rows.is_empty() {
        eprintln!("[SWEEP][ERROR] CSV rong. Chay make_feature_csv.py truoc.");
        std::process::exit(2);
    }
    let q = load_quant(&scales);

    let rs = [0usize, K / 4, K / 2];
    let scheds = [ScheduleMode::Spread, ScheduleMode::Burst];
    let seeds = [1u32, 2, 3, 4, 5];

    let mut raw = String::from(
        "K,r_pct,sched,seed,ba_new,forget,match_pct,p95_us,over_pct,lat_p50,lat_p95,recovery_first,recovery_sustained,push_drops,final_slide_ba,ba_before,ba_after\n");
    let mut aggs: BTreeMap<(usize, &'static str), Agg> = BTreeMap::new();
    let mut n_runs = 0usize;

    for &r in &rs {
        for &sched in &scheds {
            let sname = match sched { ScheduleMode::Spread => "roi", ScheduleMode::Burst => "dot" };
            let key = (r * 100 / K, sname);
            let agg = aggs.entry(key).or_insert_with(Agg::new);
            for &seed in &seeds {
                let res = run_case_seeded::<K>(&rows, r, sched, "Case1", &q, seed);
                n_runs += 1;
                let rec_s = res.recovery_windows.map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string());
                let sus_s = res.recovery_sustained_windows.map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string());
                println!(
                    "[TEST][SWEEP][r={}][{}][seed{}] ba={:.2} forget={:.2} match={:.2} p95={}us lat50={} lat95={} rec_first={} rec_sus={} drops={} slide_cuoi={:.2}",
                    key.0, sname, seed, res.ba_new, res.forget, res.match_pct,
                    res.p95_us, res.lat_p50, res.lat_p95, rec_s, sus_s, res.push_drops,
                    res.final_slide_ba
                );
                raw.push_str(&format!(
                    "{},{},{},{},{:.4},{:.4},{:.4},{},{:.4},{},{},{},{},{},{:.4},{:.4},{:.4}\n",
                    K, key.0, sname, seed, res.ba_new, res.forget, res.match_pct,
                    res.p95_us, res.over_pct, res.lat_p50, res.lat_p95,
                    rec_s, sus_s, res.push_drops, res.final_slide_ba,
                    res.ba_before, res.ba_after
                ));
                agg.ba.push(res.ba_new); agg.fg.push(res.forget); agg.mt.push(res.match_pct);
                agg.p95.push(res.p95_us as f64); agg.ov.push(res.over_pct);
                agg.l50.push(res.lat_p50 as f64); agg.l95.push(res.lat_p95 as f64);
                if let Some(w) = res.recovery_windows { agg.rec_ok += 1; agg.rec_win.push(w as f64); }
                if let Some(w) = res.recovery_sustained_windows { agg.sus_ok += 1; agg.sus_win.push(w as f64); }
                for (b, _y, ok, tot) in &res.per_bearing {
                    if *tot > 0 {
                        agg.bear.entry(b.clone()).or_default()
                            .push(100.0 * *ok as f64 / *tot as f64);
                    }
                }
            }
        }
    }
    assert_eq!(n_runs, 30, "phai du 30 cau hinh K=256");

    let mut sum = String::from(
        "K,r_pct,sched,ba_mean,ba_sd,forget_mean,forget_sd,match_mean,p95_mean,over_mean,lat50_mean,lat95_mean,rec_first_ok,rec_first_win_mean,rec_sus_ok,rec_sus_win_mean,per_bearing_sd\n");
    let mut pb_out = String::from("K,r_pct,sched,bearing,recall_mean,recall_sd\n");
    let mut forget_table: BTreeMap<(usize, &'static str), f64> = BTreeMap::new();

    for ((rp, sn), a) in &aggs {
        let (ba_m, ba_s) = mean_sd(&a.ba);
        let (fg_m, fg_s) = mean_sd(&a.fg);
        let (mt_m, _) = mean_sd(&a.mt);
        let (p95_m, _) = mean_sd(&a.p95);
        let (ov_m, _) = mean_sd(&a.ov);
        let (l50_m, _) = mean_sd(&a.l50);
        let (l95_m, _) = mean_sd(&a.l95);
        let (rec_m, _) = mean_sd(&a.rec_win);
        let (sus_m, _) = mean_sd(&a.sus_win);
        let mut bear_means = Vec::new();
        for (b, v) in &a.bear {
            let (m, s) = mean_sd(v);
            bear_means.push(m);
            pb_out.push_str(&format!("{},{},{},{},{:.4},{:.4}\n", K, rp, sn, b, m, s));
        }
        let (_, bear_sd) = mean_sd(&bear_means);
        println!(
            "[TEST][SWEEP][SUM][r={}][{}] ba={:.2}+/-{:.2} forget={:.2}+/-{:.2} match={:.2} p95={:.0}us rec_first={}/5 rec_sus={}/5 phan_tan_o={:.2}",
            rp, sn, ba_m, ba_s, fg_m, fg_s, mt_m, p95_m, a.rec_ok, a.sus_ok, bear_sd
        );
        sum.push_str(&format!(
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.4},{:.2},{:.2},{},{:.2},{},{:.2},{:.4}\n",
            K, rp, sn, ba_m, ba_s, fg_m, fg_s, mt_m, p95_m, ov_m, l50_m, l95_m,
            a.rec_ok, rec_m, a.sus_ok, sus_m, bear_sd
        ));
        forget_table.insert((*rp, *sn), fg_m);
    }

    for &sched in &scheds {
        let sn = match sched { ScheduleMode::Spread => "roi", ScheduleMode::Burst => "dot" };
        let base = forget_table[&(0, sn)];
        for &r in &rs[1..] {
            let rp = r * 100 / K;
            let f = forget_table[&(rp, sn)];
            let pass = f <= base + 1e-9;
            println!("[SELECT][{}][r={}] expected_forget<={:.2} actual={:.2} {}",
                sn, rp, base, f, if pass { "PASS" } else { "FAIL" });
        }
    }

    std::fs::create_dir_all("results").unwrap();
    std::fs::write("results/sweep.csv", &sum).unwrap();
    std::fs::write("results/sweep_runs.csv", &raw).unwrap();
    std::fs::write("results/per_bearing.csv", &pb_out).unwrap();
    println!("[SWEEP][DONE] 30 chay K=256; ghi results/sweep*.csv, per_bearing.csv");
}

/// E2b: ap luc bo nho. K nho khien P canh tranh R, san r moi rang buoc.
/// Co che da xac nhan bang du lieu: lich dot don 9 nhan vao mot buoc khien
/// P dinh 9 o; voi r=8 (50% cua K=16) thi P+R = 17 > 16 nen moi nhom mat ~1 o P.
/// Lich roi giu P ~4 o nen khong bao gio tran -> miss 0 la dung, khong phai loi.
#[test]
fn pressure_sweep() {
    let path = std::env::var("FEATURE_CSV")
        .unwrap_or_else(|_| "fixtures/features_paderborn.csv".to_string());
    let scales = std::env::var("FEATURE_SCALES")
        .unwrap_or_else(|_| "fixtures/feature_scales.csv".to_string());
    let rows = load_csv(&path);
    if rows.is_empty() { std::process::exit(2); }
    let q = load_quant(&scales);

    let pcts = [0usize, 25, 50];
    let seeds = [1u32, 2];
    let mut miss_sum: BTreeMap<(usize, usize, &'static str), u64> = BTreeMap::new();
    let mut out = String::from("K,r_pct,sched,seed,match_pct,miss,push_drops,ba_new,forget\n");

    for &kp in &[16usize, 32] {
        for &pct in &pcts {
            for &sched in &[ScheduleMode::Spread, ScheduleMode::Burst] {
                let sn = match sched { ScheduleMode::Spread => "roi", ScheduleMode::Burst => "dot" };
                let r = kp * pct / 100;
                for &seed in &seeds {
                    let res = match kp {
                        16 => run_case_seeded::<16>(&rows, r, sched, "Case2p", &q, seed),
                        32 => run_case_seeded::<32>(&rows, r, sched, "Case2p", &q, seed),
                        _ => unreachable!(),
                    };
                    let miss = res.miss_labels as u64 + res.push_drops as u64;
                    *miss_sum.entry((kp, pct, sn)).or_insert(0) += miss;
                    println!(
                        "[TEST][PRESSURE][K={}][r={}][{}][seed{}] match={:.2} miss={} drops={} ba={:.2} forget={:.2}",
                        kp, pct, sn, seed, res.match_pct, res.miss_labels, res.push_drops,
                        res.ba_new, res.forget
                    );
                    out.push_str(&format!("{},{},{},{},{:.4},{},{},{:.4},{:.4}\n",
                        kp, pct, sn, seed, res.match_pct, res.miss_labels,
                        res.push_drops, res.ba_new, res.forget));
                }
            }
        }
    }

    let m50 = miss_sum[&(16, 50, "dot")];
    let m0 = miss_sum[&(16, 0, "dot")];
    let m25 = miss_sum[&(16, 25, "dot")];
    assert!(m50 > 0, "K=16 r=50% lich dot phai sinh miss/drop, actual={m50}");
    assert!(m0 == 0 && m25 == 0, "r=0/25% tai K=16 khong duoc mat mau, m0={m0} m25={m25}");
    println!("[TEST][PRESSURE][STRUCT][K=16][dot] expected_miss50>0 actual={m50} PASS");
    println!("[TEST][PRESSURE][OBSERVE][K=16][dot] miss r0={m0} r25={m25} r50={m50} (gradient r phan biet)");

    std::fs::create_dir_all("results").unwrap();
    std::fs::write("results/pressure.csv", &out).unwrap();
    println!("[PRESSURE][DONE] ghi results/pressure.csv");
}
