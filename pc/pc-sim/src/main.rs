use pc_sim::xor_tanh::run_xor_tanh_test_silent;

fn main() {
    println!("[PC_SIM][START] Chay thu nghiem XOR tren PC");
    let r = run_xor_tanh_test_silent();
    println!(
        "[PC_SIM][END] correct={}/{} final_loss={:.6} passed={}",
        r.correct, r.total, r.final_loss, r.passed()
    );
}
