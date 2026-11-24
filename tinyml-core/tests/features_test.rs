// Kiểm thử bộ trích xuất 16 đặc trưng. Giá trị mong đợi đã đối chiếu bằng NumPy.
use tinyml_core::features::{extract, SAMPLE_RATE_HZ, WINDOW_LEN};

fn sine(freq_hz: f32, amplitude: f32) -> [f32; WINDOW_LEN] {
    let mut x = [0.0f32; WINDOW_LEN];
    for (n, v) in x.iter_mut().enumerate() {
        *v = amplitude * (2.0 * std::f32::consts::PI * freq_hz * n as f32 / SAMPLE_RATE_HZ).sin();
    }
    x
}

#[test]
fn test_features_sine_5khz() {
    println!("[TEST][FEATURES][Sine5k] Bat dau...");
    // 5 kHz rơi đúng bin 80 (62,5 Hz/bin), không rò phổ
    let z = extract(&sine(5000.0, 1.0));
    let near = |v: f32, want: f32, tol: f32| (v - want).abs() <= tol;

    assert!(near(z[0], std::f32::consts::FRAC_1_SQRT_2, 1e-3), "rms = {}", z[0]);
    assert!(near(z[1], std::f32::consts::FRAC_1_SQRT_2, 1e-3), "std = {}", z[1]);
    assert!(near(z[2], 1.0, 1e-3), "peak = {}", z[2]);
    assert!(near(z[4], std::f32::consts::SQRT_2, 1e-2), "crest = {}", z[4]);
    assert!(near(z[5], 1.5, 1e-2), "kurtosis = {}", z[5]);
    assert!(near(z[6], 0.0, 1e-2), "skewness = {}", z[6]);

    // Năng lượng dồn vào dải 4–8 kHz: |X_80|^2 / N^2 = 0,25 -> log10 = -0,602
    let bands = &z[10..16];
    assert!(near(bands[3], -0.602, 1e-2), "dai 4-8k = {}", bands[3]);
    for (b, &e) in bands.iter().enumerate() {
        if b != 3 {
            assert!(e < bands[3] - 3.0, "dai {} = {} khong nho hon han dai 4-8k", b, e);
        }
    }
    println!("[TEST][FEATURES][Sine5k] PASS");
}

#[test]
fn test_features_zero_signal_is_finite() {
    println!("[TEST][FEATURES][Zero] Bat dau...");
    let z = extract(&[0.0; WINDOW_LEN]);
    assert!(z.iter().all(|v| v.is_finite()), "{:?}", z);
    println!("[TEST][FEATURES][Zero] PASS");
}

#[test]
fn test_features_scale_invariant_shape() {
    println!("[TEST][FEATURES][Scale] Bat dau...");
    // Nhân biên độ 10 lần: rms tăng 10 lần, còn kurtosis và hệ số đỉnh không đổi
    let a = extract(&sine(3000.0, 1.0));
    let b = extract(&sine(3000.0, 10.0));
    assert!((b[0] / a[0] - 10.0).abs() < 1e-2);
    assert!((b[5] - a[5]).abs() < 1e-3);
    assert!((b[4] - a[4]).abs() < 1e-3);
    println!("[TEST][FEATURES][Scale] PASS");
}
