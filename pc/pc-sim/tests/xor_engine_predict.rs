//! B4: XOR chay tren LinearHead THAT (de cuong muc 4), khong qua struct rieng.
//! z = (x1, x2, x1*x2, 0...0) 16 chieu; diem lop 1 = x1+x2-2*x1*x2-0.5, lop 0 = 0.

use pc_sim::head_linear::{LinearHead, Quant, N_FEATURES};

fn xor_z(x1: i8, x2: i8) -> [i8; N_FEATURES] {
    let mut z = [0i8; N_FEATURES]; // 13 chieu con lai = 0 dung de cuong
    z[0] = x1; z[1] = x2; z[2] = x1 * x2;
    z
}

fn xor_head() -> LinearHead<2> {
    let mut h: LinearHead<2> = LinearHead::zeros();
    h.w[1][0] = 1.0; h.w[1][1] = 1.0; h.w[1][2] = -2.0;
    h.b[1] = -0.5;
    h
}

/// Happy: 4/4 du doan XOR khop bang, diem khop tung gia tri.
#[test]
fn xor_predict_4_of_4_on_real_engine() {
    let h = xor_head();
    let q = Quant::identity();
    let cases = [(0i8, 0i8, 0u8, -0.5f32), (0, 1, 1, 0.5), (1, 0, 1, 0.5), (1, 1, 0, -0.5)];
    let mut ok = 0;
    for (x1, x2, want, want_s) in cases {
        let z = xor_z(x1, x2);
        let s = h.scores(&LinearHead::<2>::dequant(z, &q));
        let got = h.predict(z, &q);
        assert!((s[1] - s[0] - want_s).abs() < 1e-6, "score lech tai ({x1},{x2})");
        assert_eq!(got, want);
        ok += 1;
        println!("[TEST][XOR_ENGINE][{x1}{x2}] expected={want} score={want_s} actual={got}/{:.1} PASS", s[1] - s[0]);
    }
    assert_eq!(ok, 4);
}

/// Edge: (1,1) score am sat 0; hoa diem (score=0) -> chon lop 0 (argmax dau tien).
#[test]
fn xor_edge_tie_and_negative() {
    let mut h = xor_head();
    let q = Quant::identity();
    // (1,1): score = -0.5 < 0 -> lop 0
    assert_eq!(h.predict(xor_z(1, 1), &q), 0);
    // Hoa diem: dat b1 = 0, x=(1,0) khong co thanh phan tuong tac -> score1 = 1
    // thay vi tao hoa tu nhien, dat thang w1=[1,0..], b1=-1, x=(1,0) -> score=0
    h.w[1] = [0.0; N_FEATURES]; h.w[1][0] = 1.0; h.b[1] = -1.0;
    let s = h.scores(&LinearHead::<2>::dequant(xor_z(1, 0), &q));
    assert!((s[1] - s[0]).abs() < 1e-6);
    assert_eq!(h.predict(xor_z(1, 0), &q), 0); // tie -> lop 0, khai bao on dinh
    println!("[TEST][XOR_ENGINE][TIE] expected=0 actual=0 PASS");
}

/// Error: label >= C -> NaN, W/b bat bien (khong cap nhat).
#[test]
fn invalid_label_nan_no_update() {
    let mut h = xor_head();
    let q = Quant::identity();
    let before_w = h.w; let before_b = h.b;
    let loss = h.sgd_step(xor_z(1, 0), &q, 2, 0.1); // C=2, label 2 ngoai pham vi
    assert!(loss.is_nan());
    assert_eq!(h.w, before_w);
    assert_eq!(h.b, before_b);
    println!("[TEST][XOR_ENGINE][BAD_LABEL] expected=NaN+bat_bien actual=NaN+bat_bien PASS");
}
