use ruv_fann::{ActivationFunction, NetworkBuilder};
use pc_sim::memsim::{MemoryManager, Sample};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("[DEBUG][MAIN] Bat dau mo phong luong du lieu TinyML (10 mau)");

    // 1. Khoi tao bo quan ly bo nho (Gioi han 2000 bytes)
    let mut memory = MemoryManager::new(2000);
    
    // 2. Khoi tao mang Neural Network (16 Input, 8 Hidden, 3 Output)
    let mut network = NetworkBuilder::<f32>::new()
        .input_layer(16)
        .hidden_layer_with_activation(8, ActivationFunction::Tanh, 1.0)
        .output_layer_with_activation(3, ActivationFunction::Sigmoid, 1.0)
        .build();

    // Bien dem de tinh toan chi so
    let mut total_samples = 0;
    let mut successful_matches = 0;
    
    // Hang doi gia lap do tre nhan (chua ID va nhan tuong ung)
    let mut pending_labels: Vec<(u32, u8)> = Vec::new();

    // 3. Vong lap mo phong 10 buoc thoi gian
    for step in 1..=10 {
        println!("\n--- BUOC THOI GIAN {} ---", step);
        
        // 3.1. Nhan tin hieu moi (Tao mau)
        let sample_id = 100 + step;
        // Gia lap 16 dac trung bang gia tri ngau nhien tu step
        let mut features = [0.0f32; 16];
        for i in 0..16 { features[i] = (step as f32 * 0.1) + (i as f32 * 0.05); }
        
        let new_sample = Sample::new(sample_id, features, step as u64);
        if memory.add_to_p(new_sample).is_ok() {
            total_samples += 1;
            println!("[DEBUG][MEMORY] Them mau ID={} vao P. Kich thuoc P={}", sample_id, memory.p_queue.len());
        }

        // 3.2. Gia lap du doan
        let _prediction = network.run(&features);

        // 3.3. Gia lap nhan den tre (Vi du: nhan den tre 2 chu ky)
        // Moi khi step tang, chung ta se tao mot nhan den tre cho step - 2
        if step > 2 {
            let delayed_id = 100 + step - 2;
            let delayed_label = (delayed_id % 3) as u8; // Gia lap 3 lop loi
            println!("[DEBUG][LABEL] Nhan {} den tre cho mau ID={}", delayed_label, delayed_id);
            
            if memory.assign_label(delayed_id, delayed_label).is_ok() {
                successful_matches += 1;
            }
        }

        // 3.4. Xu ly hang doi Q (Huan luyen)
        while let Some(train_sample) = memory.pop_for_training() {
            println!("[DEBUG][TRAIN] Lay mau ID={} tu Q de hoc.", train_sample.id);
            
            let mut target_output = vec![0.0; 3];
            target_output[train_sample.label.unwrap() as usize] = 1.0;
            
            let inputs = vec![train_sample.features.to_vec()];
            let outputs = vec![target_output];
            
            // Train 1 epoch
            network.train(&inputs, &outputs, 0.001, 1)?;
        }

        // In trang thai bo nho
        println!("[DEBUG][STATUS] P={}, Q={}, R={} | Dung luong: {}/{} bytes", 
            memory.p_queue.len(), memory.q_queue.len(), memory.r_queue.len(), 
            memory.current_usage_bytes(), memory.max_memory_bytes);
    }

    // 4. Tong ket
    println!("\n=== TONG KET MO PHONG ===");
    println!("Tong so mau tao ra: {}", total_samples);
    println!("So nhan ghep thanh cong: {}", successful_matches);
    let match_rate = if total_samples > 0 { (successful_matches as f32 / total_samples as f32) * 100.0 } else { 0.0 };
    println!("Ty le ghep nhan (Match Rate): {:.2}%", match_rate);

    Ok(())
}
