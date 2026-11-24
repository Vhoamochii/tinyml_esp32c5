// Kiểm thử bộ trích xuất 16 đặc trưng (cửa sổ 16384 mẫu, có phổ bao).
// Giá trị mong đợi đã đối chiếu bằng bản tham chiếu NumPy (float64).
use tinyml_core::features::{extract, WindowExtractor, BLOCK_LEN, SAMPLE_RATE_HZ, WINDOW_LEN};

fn sine(freq_hz: f32, amplitude: f32) -> Vec<f32> {
    (0..WINDOW_LEN)
        .map(|n| amplitude * (2.0 * std::f32::consts::PI * freq_hz * n as f32 / SAMPLE_RATE_HZ).sin())
        .collect()
}

/// Tiếng gõ đều `rate_hz` lần/giây, mỗi cú kích cộng hưởng 5 kHz tắt dần (hằng số thời gian 0,5 ms)
fn impacts(rate_hz: f32, amplitude: f32) -> Vec<f32> {
    let mut x = vec![0.0f32; WINDOW_LEN];
    let mut k = 0;
    loop {
        let start = (k as f32 * SAMPLE_RATE_HZ / rate_hz).round() as usize;
        if start >= WINDOW_LEN {
            break;
        }
        for (i, v) in x[start..(start + 640).min(WINDOW_LEN)].iter_mut().enumerate() {
            let t = i as f32 / SAMPLE_RATE_HZ;
            *v += amplitude * (-t / 0.0005).exp() * (2.0 * std::f32::consts::PI * 5000.0 * t).sin();
        }
        k += 1;
    }
    x
}

fn features(x: &[f32]) -> [f32; 16] {
    extract(x.try_into().unwrap())
}

/// Chỉ số đặc trưng phổ bao (10..16) lớn nhất
fn strongest_envelope_band(z: &[f32; 16]) -> usize {
    (10..16).max_by(|&a, &b| z[a].total_cmp(&z[b])).unwrap()
}

#[test]
fn test_features_sine_5khz() {
    println!("[TEST][FEATURES][Sine5k] Bat dau...");
    let z = features(&sine(5000.0, 1.0));
    let near = |v: f32, want: f32, tol: f32| (v - want).abs() <= tol;

    assert!(near(z[0], std::f32::consts::SQRT_2, 1e-2), "crest = {}", z[0]);
    assert!(near(z[1], 1.5, 1e-2), "kurtosis = {}", z[1]);
    assert!(near(z[2], 0.0, 1e-2), "skewness = {}", z[2]);
    assert!(near(z[3], std::f32::consts::FRAC_PI_2, 1e-2), "impulse = {}", z[3]);
    assert!(near(z[4], 1.1116, 1e-2), "shape = {}", z[4]);

    // Toàn bộ năng lượng nằm ở dải 2-8 kHz: tỷ lệ 1 -> log10 = 0
    assert!(near(z[7], 0.0, 1e-3), "dai 2-8k = {}", z[7]);
    for b in [6, 8, 9] {
        assert!(z[b] < -6.0, "dai {} = {} khong nho hon han", b, z[b]);
    }
    println!("[TEST][FEATURES][Sine5k] PASS");
}

#[test]
fn test_features_zero_signal_is_finite() {
    println!("[TEST][FEATURES][Zero] Bat dau...");
    let z = features(&vec![0.0; WINDOW_LEN]);
    assert!(z.iter().all(|v| v.is_finite()), "{:?}", z);
    println!("[TEST][FEATURES][Zero] PASS");
}

#[test]
fn test_features_scale_invariant() {
    println!("[TEST][FEATURES][Scale] Bat dau...");
    // Cả 16 đặc trưng không phụ thuộc độ to: nhân biên độ 3 lần không đổi kết quả
    let a = features(&impacts(76.4, 1.0));
    let b = features(&impacts(76.4, 3.0));
    for i in 0..16 {
        assert!((a[i] - b[i]).abs() < 1e-3, "dac trung {}: {} vs {}", i, a[i], b[i]);
    }
    println!("[TEST][FEATURES][Scale] PASS");
}

#[test]
fn test_features_envelope_fault_rate() {
    println!("[TEST][FEATURES][Envelope] Bat dau...");
    // Ổ 6203 ở 1500 vòng/phút: vòng ngoài gõ ~76,4 Hz -> dải 60-100 Hz (chỉ số 13),
    // vòng trong gõ ~123,6 Hz -> dải 100-170 Hz (chỉ số 14)
    let outer = features(&impacts(76.4, 1.0));
    let inner = features(&impacts(123.6, 1.0));
    assert_eq!(strongest_envelope_band(&outer), 13, "vong ngoai N15: {:?}", &outer[10..]);
    assert_eq!(strongest_envelope_band(&inner), 14, "vong trong N15: {:?}", &inner[10..]);
    assert!(outer[13] > inner[13] + 3.0, "dai 60-100 Hz phai tach hai loi");

    // Ở 900 vòng/phút: vòng trong ~74,2 Hz rơi vào cùng dải với vòng ngoài ở 1500 vòng/phút.
    // Đây là độ lệch khi đổi tốc độ mà bo phải học lại.
    // Vòng ngoài ~45,8 Hz nằm sát 50 Hz nên bin chính bị loại cùng nhiễu điện lưới: dải 35-60 Hz
    // không còn cao nhất, nhưng vẫn tách rõ vòng ngoài với vòng trong ở cùng tốc độ.
    let outer_slow = features(&impacts(45.8, 1.0));
    let inner_slow = features(&impacts(74.2, 1.0));
    assert_eq!(strongest_envelope_band(&inner_slow), 13, "vong trong N09: {:?}", &inner_slow[10..]);
    assert!(outer_slow[12] > inner_slow[12] + 3.0, "dai 35-60 Hz phai tach hai loi o N09");
    println!("[TEST][FEATURES][Envelope] PASS");
}

#[test]
fn test_features_mains_rejected() {
    println!("[TEST][FEATURES][Mains] Bat dau...");
    // Tiếng gõ vòng trong 123,6 Hz cộng một thành phần 7 kHz bị điều biến 50 Hz (giả lập nhiễu điện lưới).
    // Nếu không loại các bin quanh bội số 50 Hz, dải 35-60 Hz sẽ lấn át; có loại thì đỉnh vẫn ở 100-170 Hz.
    let mut x = impacts(123.6, 1.0);
    for (n, v) in x.iter_mut().enumerate() {
        let t = n as f32 / SAMPLE_RATE_HZ;
        let tau = 2.0 * std::f32::consts::PI;
        *v += 0.3 * (1.0 + (tau * 50.0 * t).sin()) * (tau * 7000.0 * t).sin();
    }
    let z = features(&x);
    assert_eq!(strongest_envelope_band(&z), 14, "{:?}", &z[10..]);
    println!("[TEST][FEATURES][Mains] PASS");
}

#[test]
fn test_features_streaming_matches_window() {
    println!("[TEST][FEATURES][Stream] Bat dau...");
    // Nạp từng khối như trên bo: chỉ khối cuối trả kết quả, và kết quả giống extract() cả cửa sổ.
    // Cửa sổ thứ hai dùng lại cùng bộ trích xuất phải ra y hệt (đã tự đặt lại).
    let x = impacts(123.6, 1.0);
    let want = features(&x);
    let mut ex = WindowExtractor::new();
    for round in 0..2 {
        for (i, block) in x.chunks_exact(BLOCK_LEN).enumerate() {
            let out = ex.push_block(block.try_into().unwrap());
            if i + 1 < x.len() / BLOCK_LEN {
                assert!(out.is_none(), "vong {round}: khoi {i} khong duoc tra ket qua");
            } else {
                assert_eq!(out, Some(want), "vong {round}: ket qua khac extract()");
            }
        }
    }
    println!("[TEST][FEATURES][Stream] PASS");
}
