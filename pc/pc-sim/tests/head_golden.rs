use pc_sim::head_linear::run_head_golden;

#[test]
fn head_golden_matches_decuong() {
    let (ok, err) = run_head_golden();
    println!(
        "[TEST][HEAD_GOLDEN][Step200] expected_err<=1e-6 actual_err={:.3e} {}",
        err, if ok { "PASS" } else { "FAIL" }
    );
    assert!(ok, "head tuyen tinh lech gia tri vang de cuong muc 4");
}
