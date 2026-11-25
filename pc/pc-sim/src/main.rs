//! Smoke check nhanh: gia tri vang XOR muc 4. Bang do that nam o tests.
fn main() {
    let (ok, err) = pc_sim::head_linear::run_head_golden();
    println!("[MAIN][HEAD_GOLDEN] ok={ok} max_err={err:.3e}");
    if !ok { std::process::exit(1); }
}
