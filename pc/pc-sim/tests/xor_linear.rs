//! Test case T7: Chứng minh XOR có thể giải bằng tuyến tính nếu mở rộng đặc trưng.
//! Ánh xạ: z = (x1, x2, x1*x2)
//! Công thức score: score = x1 + x2 - 2*x1*x2 - 0.5

#[derive(Debug)]
struct LinearXorScorer;

impl LinearXorScorer {
    /// Trả về (features_mở_rộng, score, nhãn_dự_đoán)
    pub fn evaluate(x1: f32, x2: f32) -> ((f32, f32, f32), f32, u8) {
        // Mở rộng không gian đặc trưng từ 2D lên 3D (thêm thành phần tương tác)
        let z = (x1, x2, x1 * x2);
        
        // Tính điểm dựa trên công thức đề cương
        let score = z.0 + z.1 - 2.0 * z.2 - 0.5;
        
        // Quy tắc phân loại
        let predicted_class = if score > 0.0 { 1 } else { 0 };
        
        (z, score, predicted_class)
    }
}

#[test]
fn test_linear_xor_scorer() {
    let test_cases = [
        (0.0, 0.0, 0), // 0 XOR 0 = 0
        (0.0, 1.0, 1), // 0 XOR 1 = 1
        (1.0, 0.0, 1), // 1 XOR 0 = 1
        (1.0, 1.0, 0), // 1 XOR 1 = 0
    ];

    let mut passed = 0;
    for (x1, x2, expected) in test_cases.iter() {
        let (z, score, predicted) = LinearXorScorer::evaluate(*x1, *x2);
        
        let is_correct = predicted == *expected;
        if is_correct { passed += 1; }

        println!(
            "[TEST][XOR_LINEAR][score] x1={}, x2={} -> z=({},{},{}), score={}, predict={}, expected={} -> {}",
            x1, x2, z.0, z.1, z.2, score, predicted, expected,
            if is_correct { "PASS" } else { "FAIL" }
        );
    }

    assert_eq!(passed, 4, "Phải giải đúng 4/4 trường hợp XOR");
    println!("[TEST][XOR_LINEAR][happy] Hoàn tất: {}/4 PASS", passed);
}
