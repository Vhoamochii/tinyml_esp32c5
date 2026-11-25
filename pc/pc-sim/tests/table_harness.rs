use pc_sim::schedule::{build, mean_delay, ScheduleMode};
use pc_sim::stream_runner::{load_csv, load_quant, run_case, K};

#[test]
fn schedule_math() {
    // Moi lich trung binh tre 4 buoc (de cuong muc 5)
    let labels = vec![0u8; 90];
    let s = mean_delay(90, ScheduleMode::Spread, &labels);
    let b = mean_delay(90, ScheduleMode::Burst, &labels);
    println!("[TEST][SCHEDULE][Spread] expected=4.00 actual={:.2} {}", s, if (s - 4.0).abs() < 1e-9 { "PASS" } else { "FAIL" });
    println!("[TEST][SCHEDULE][Burst]   expected=4.00 actual={:.2} {}", b, if (b - 4.0).abs() < 1e-9 { "PASS" } else { "FAIL" });
    assert!((s - 4.0).abs() < 1e-9 && (b - 4.0).abs() < 1e-9);
    let ev = build(90, ScheduleMode::Burst, &labels);
    assert_eq!(ev.len(), 90);
}

#[test]
fn table_case1() {
    let path = std::env::var("FEATURE_CSV")
        .unwrap_or_else(|_| "fixtures/features_paderborn.csv".to_string());
    let scales = std::env::var("FEATURE_SCALES")
        .unwrap_or_else(|_| "fixtures/feature_scales.csv".to_string());
    let rows = load_csv(&path);
    if rows.is_empty() {
        eprintln!("[TABLE][ERROR] CSV rong tai {path}. Chay make_feature_csv.py truoc.");
        std::process::exit(2);
    }
    let q = load_quant(&scales);
    let r = K / 4; // 25% K, lam tron xuong
    let mut out = String::new();
    out.push_str("Phuong_an,BA_moi_p100,Muc_quen_p100,Nhan_khop_p100,RAM_proxy_byte,P95_us,Qua_han_p100,BA_truoc,BA_sau,Hoc_lan_dau,Hoc_lai,Ghep_truot,N_stream,N_reserve\n");
    for sched in [ScheduleMode::Spread, ScheduleMode::Burst] {
        let res = run_case(&rows, r, sched, "Case1_Rust_Python", &q);
        // Gate: phai co reserve de do quen va luong N09 de do BA moi
        assert!(res.n_reserve > 0 && res.n_stream > 0,
            "thieu reserve hoac luong: reserve={} stream={}", res.n_reserve, res.n_stream);
        out.push_str(&format!(
            "{},{:.2},{:.2},{:.2},{},{},{:.2},{:.2},{:.2},{},{},{},{},{}\n",
            res.tag, res.ba_new, res.forget, res.match_pct,
            res.ram_proxy, res.p95_us, res.over_pct,
            res.ba_before, res.ba_after,
            res.learned_first, res.replay_learns, res.miss_labels,
            res.n_stream, res.n_reserve
        ));
        println!(
            "[TEST][TABLE][{}] ba_new={:.2} ba_truoc={:.2} ba_sau={:.2} forget={:.2} match={:.2} ram={} p95={}us over={:.2} learned={} replay={} miss={} stream={} reserve={}",
            res.tag, res.ba_new, res.ba_before, res.ba_after, res.forget, res.match_pct,
            res.ram_proxy, res.p95_us, res.over_pct,
            res.learned_first, res.replay_learns, res.miss_labels,
            res.n_stream, res.n_reserve
        );
    }
    std::fs::create_dir_all("results").unwrap();
    std::fs::write("results/case1.csv", &out).unwrap();
    println!("[TABLE][DONE] da ghi results/case1.csv");
}
