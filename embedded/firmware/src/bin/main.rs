// Sử dụng prelude để import tất cả các thành phần cần thiết
use ruv_fann::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // In log debug theo đúng quy định Master Prompt
    println!("[DEBUG][MAIN] Bat dau chuong trinh kiem thu ruv-FANN");

    // 1. Khoi tao mang Neural Network voi kien truc: 2 Input, 3 Hidden, 1 Output
    // Su dung NeuralNetwork::builder() theo API moi
    // Them bias (gia tri 1.0) vao input de hoi tu nhanh hon
    let mut net = NeuralNetwork::builder()
        .input_size(3) // 2 input thuc + 1 bias
        .hidden_layers(&[3])
        .output_size(1)
        .activation(Activation::Sigmoid)
        .learning_rate(0.7)
        .build()?;

    println!("[DEBUG][MAIN] Da khoi tao mang 2-3-1 (co bias) thanh cong");

    // 2. Chuan bi du lieu huan luyen cho bai toan XOR (co bias)
    let training_data = vec![
        (vec![0.0, 0.0, 1.0], vec![0.0]), // 0 XOR 0 = 0
        (vec![0.0, 1.0, 1.0], vec![1.0]), // 0 XOR 1 = 1
        (vec![1.0, 0.0, 1.0], vec![1.0]), // 1 XOR 0 = 1
        (vec![1.0, 1.0, 1.0], vec![0.0]), // 1 XOR 1 = 0
    ];

    // 3. Tien hanh huan luyen (Train)
    println!("[DEBUG][MAIN] Bat dau train bai toan XOR...");
    // train(&training_data, max_epochs, desired_error)
    net.train(&training_data, 2000, 0.001)?;
    println!("[DEBUG][MAIN] Huan luyen hoan tat");

    // 4. Kiem tra (Verify) ket qua du doan voi sai so cho phep la 0.1
    let mut all_passed = true;

    // Test Case 1: [0, 0] -> 0
    let out1 = net.run(&[0.0, 0.0, 1.0])?;
    let pass1 = (out1[0] - 0.0).abs() < 0.1;
    println!(
        "[TEST][RUV-FANN][XOR_00] expected=0.0 actual={:.4} -> {}",
        out1[0],
        if pass1 { "PASS" } else { "FAIL" }
    );
    if !pass1 { all_passed = false; }

    // Test Case 2: [0, 1] -> 1
    let out2 = net.run(&[0.0, 1.0, 1.0])?;
    let pass2 = (out2[0] - 1.0).abs() < 0.1;
    println!(
        "[TEST][RUV-FANN][XOR_01] expected=1.0 actual={:.4} -> {}",
        out2[0],
        if pass2 { "PASS" } else { "FAIL" }
    );
    if !pass2 { all_passed = false; }

    // Test Case 3: [1, 0] -> 1
    let out3 = net.run(&[1.0, 0.0, 1.0])?;
    let pass3 = (out3[0] - 1.0).abs() < 0.1;
    println!(
        "[TEST][RUV-FANN][XOR_10] expected=1.0 actual={:.4} -> {}",
        out3[0],
        if pass3 { "PASS" } else { "FAIL" }
    );
    if !pass3 { all_passed = false; }

    // Test Case 4: [1, 1] -> 0
    let out4 = net.run(&[1.0, 1.0, 1.0])?;
    let pass4 = (out4[0] - 0.0).abs() < 0.1;
    println!(
        "[TEST][RUV-FANN][XOR_11] expected=0.0 actual={:.4} -> {}",
        out4[0],
        if pass4 { "PASS" } else { "FAIL" }
    );
    if !pass4 { all_passed = false; }

    // 5. Tong ket
    if all_passed {
        println!("[TEST][RUV-FANN][XOR_TOTAL] PASS");
    } else {
        println!("[TEST][RUV-FANN][XOR_TOTAL] FAIL");
    }

    Ok(())
}
