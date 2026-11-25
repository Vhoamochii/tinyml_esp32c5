//! F6: moc parity phia PC: 10 du doan N09 dau tien cua Case 1 (lich roi, seed mac dinh).
//! Firmware in cung 10 gia tri; so khop tung dong moi dat parity bo-PC.

use pc_sim::schedule::ScheduleMode;
use pc_sim::stream_runner::{load_csv, load_quant, run_case_seeded, DEFAULT_SEED, K};

#[test]
fn dump_parity_pc() {
    let rows = load_csv("fixtures/features_paderborn.csv");
    let q = load_quant("fixtures/feature_scales.csv");
    let res = run_case_seeded::<K>(&rows, K / 4, ScheduleMode::Spread, "Case1", &q, DEFAULT_SEED);
    let mut out = String::new();
    for (i, (b, y, p)) in res.preds.iter().take(10).enumerate() {
        out.push_str(&format!("{i}:{b}:{y}:{p}\n"));
        println!("[TEST][PARITY_PC][{i}] bearing={b} class={y} pred={p}");
    }
    assert_eq!(res.preds.len(), 30069, "so du doan phai dung 30069, khong duoc la stub");
    std::fs::create_dir_all("results").unwrap();
    std::fs::write("results/parity_pc.txt", &out).unwrap();
    println!("[TEST][PARITY_PC][DONE] ghi results/parity_pc.txt (10 dong)");
}
