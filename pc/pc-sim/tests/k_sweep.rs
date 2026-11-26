//! Quet ap luc bo nho K x r tren tap chon cau hinh (huong B, de cuong muc 5).
//! Dung nguyen engine MemPool (r_floor da noi dung), KHONG sua core.
//! Muc dich: tim nguong K ma r bat dau rang buoc (miss/drops > 0, ket qua tach khoi nhau).
//! Chi chay tren fixtures_val; tap 6 o test van dong.

use pc_sim::schedule::ScheduleMode;
use pc_sim::stream_runner::{load_csv, load_quant, run_case_replay, ReplayMode};

const SEEDS: [u32; 5] = [1, 2, 3, 4, 5];
const HEADER: &str = "K,r_pct,sched,mode,seed,ba_new,forget,match_pct,miss_labels,push_drops,learned_first,replay_learns\n";

macro_rules! sweep {
    ($KP:literal, $rows:expr, $q:expr, $out:expr) => {{
        for &sched in &[ScheduleMode::Spread, ScheduleMode::Burst] {
            let sn = match sched { ScheduleMode::Spread => "roi", _ => "dot" };
            for &r in &[0usize, $KP / 4, $KP / 2] {
                let (mut sb, mut sf, mut sm, mut sd) = (0f64, 0f64, 0u64, 0u64);
                for &seed in &SEEDS {
                    let res = run_case_replay::<$KP>($rows, r, sched, "val", $q, seed, ReplayMode::Interleaved);
                    sb += res.ba_new; sf += res.forget;
                    sm += res.miss_labels as u64; sd += res.push_drops as u64;
                    $out.push_str(&format!("{},{},{},inter,{},{:.4},{:.4},{:.4},{},{},{},{}\n",
                        $KP, r * 100 / $KP, sn, seed, res.ba_new, res.forget, res.match_pct,
                        res.miss_labels, res.push_drops, res.learned_first, res.replay_learns));
                }
                println!("[KSWEEP][K={}][{}][r={}] ba={:.2} forget={:.2} miss={} drops={}",
                    $KP, sn, r * 100 / $KP, sb / 5.0, sf / 5.0, sm, sd);
            }
            let (mut sb, mut sf) = (0f64, 0f64);
            for &seed in &SEEDS {
                let res = run_case_replay::<$KP>($rows, 0, sched, "val", $q, seed, ReplayMode::Off);
                sb += res.ba_new; sf += res.forget;
                $out.push_str(&format!("{},0,{},off,{},{:.4},{:.4},{:.4},{},{},{},{}\n",
                    $KP, sn, seed, res.ba_new, res.forget, res.match_pct,
                    res.miss_labels, res.push_drops, res.learned_first, res.replay_learns));
            }
            println!("[KSWEEP][K={}][{}][OFF] ba={:.2} forget={:.2}", $KP, sn, sb / 5.0, sf / 5.0);
        }
    }};
}

#[test]
fn k_sweep_pressure() {
    let csv = std::env::var("FEATURE_CSV")
        .unwrap_or_else(|_| "fixtures_val/features_val.csv".to_string());
    let sc = std::env::var("FEATURE_SCALES")
        .unwrap_or_else(|_| "fixtures/feature_scales.csv".to_string());
    let rows = load_csv(&csv);
    assert!(!rows.is_empty(), "fixtures_val chua co");
    let q = load_quant(&sc);
    let mut out = String::from(HEADER);
    sweep!(256, &rows, &q, &mut out);
    sweep!(128, &rows, &q, &mut out);
    sweep!(64, &rows, &q, &mut out);
    sweep!(32, &rows, &q, &mut out);
    sweep!(16, &rows, &q, &mut out);
    std::fs::create_dir_all("results").unwrap();
    std::fs::write("results/k_sweep.csv", &out).unwrap();
    println!("[KSWEEP][DONE] ghi results/k_sweep.csv");
}
