use tinyol_core::xor_tanh::{run_xor_tanh_test_silent, XorTanhNet, XOR_DATA, HIDDEN_DIM};

#[test]
fn happy_converges_4_of_4() {
    let r = run_xor_tanh_test_silent();
    println!("[TEST][XOR_TANH][happy] expected_correct=4 actual_correct={} final_loss={} PASS={}", r.correct, r.final_loss, r.passed());
    assert_eq!(r.correct, 4, "XOR tanh phai hoi tu 4/4 sau 2000 epoch");
    assert!(r.final_loss < 0.1, "Loss cuoi phai < 0.1, thuc te {}", r.final_loss);
}

#[test]
fn edge_predict_threshold_zero() {
    let net = XorTanhNet::zero();
    assert_eq!(net.predict([0.0, 0.0]), 0);
    println!("[TEST][XOR_TANH][edge_threshold] y=0 -> class 0 PASS");
}

#[test]
fn edge_deterministic_with_same_seed() {
    let a = XorTanhNet::new(12345);
    let b = XorTanhNet::new(12345);
    for i in 0..HIDDEN_DIM {
        for j in 0..2 { assert_eq!(a.w_hidden[i][j].to_bits(), b.w_hidden[i][j].to_bits()); }
    }
    println!("[TEST][XOR_TANH][edge_deterministic] same_seed_equal=true PASS");
}

#[test]
fn error_zero_init_cannot_learn() {
    let mut net = XorTanhNet::zero();
    for _ in 0..2000 {
        for (x, t) in XOR_DATA.iter() { net.train_step(*x, *t, 0.2); }
    }
    let mut correct = 0u8;
    for (x, t) in XOR_DATA.iter() { if net.predict(*x) == *t { correct += 1; } }
    println!("[TEST][XOR_TANH][error_zero_init] expected_correct=2 actual_correct={} PASS={}", correct, correct == 2);
    assert_eq!(correct, 2, "W=0 phai khong hoc, chi dung 2/4");
}
