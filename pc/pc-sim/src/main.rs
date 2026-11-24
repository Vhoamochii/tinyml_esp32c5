use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruv_fann::{ActivationFunction, NetworkBuilder};
use pc_sim::memsim::{MemoryManager, Sample};

// Tham số thí nghiệm
const STEPS: u64 = 3000;
const MEMORY_BUDGET_BYTES: usize = 2000; // 25 mẫu; R dùng tối đa một nửa
const NUM_FEATURES: usize = 16;
const NUM_CLASSES: usize = 3;
const NOISE: f32 = 0.6; // Biên độ nhiễu đều quanh tâm cụm của mỗi lớp
const MIN_DELAY: u64 = 1; // Nhãn đến trễ ngẫu nhiên trong [MIN_DELAY, MAX_DELAY] bước
const MAX_DELAY: u64 = 12;
const PENDING_TTL: u64 = 10; // Mẫu chờ nhãn quá số bước này bị loại khỏi P
const LEARNING_RATE: f32 = 0.1;
const REPLAY_K: usize = 2; // Số mẫu từ R học kèm mỗi mẫu mới
const REPORT_EVERY: u64 = 500;
const SEED: u64 = 42;

// Sinh một mẫu: chọn lớp ngẫu nhiên, đặc trưng = tâm cụm của lớp + nhiễu
fn make_sample(rng: &mut StdRng, centers: &[[f32; NUM_FEATURES]]) -> (u8, [f32; NUM_FEATURES]) {
    let class = rng.gen_range(0..NUM_CLASSES);
    let mut features = [0.0f32; NUM_FEATURES];
    for (f, c) in features.iter_mut().zip(centers[class].iter()) {
        *f = c + rng.gen_range(-NOISE..NOISE);
    }
    (class as u8, features)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("[DEBUG][MAIN] Bat dau mo phong TinyML nhan tre ({} buoc, seed={})", STEPS, SEED);

    let mut rng = StdRng::seed_from_u64(SEED);
    let mut memory = MemoryManager::new(MEMORY_BUDGET_BYTES);

    // Mạng Neural Network: 16 Input, 8 Hidden (Tanh), 3 Output (Sigmoid)
    let mut network = NetworkBuilder::<f32>::new()
        .input_layer(NUM_FEATURES)
        .hidden_layer_with_activation(8, ActivationFunction::Tanh, 1.0)
        .output_layer_with_activation(NUM_CLASSES, ActivationFunction::Sigmoid, 1.0)
        .build();

    // Tâm cụm cố định cho mỗi lớp
    let mut centers = [[0.0f32; NUM_FEATURES]; NUM_CLASSES];
    for center in centers.iter_mut() {
        for v in center.iter_mut() {
            *v = rng.gen_range(-1.0..1.0);
        }
    }

    // Kênh nhãn trễ: (bước nhãn đến, ID mẫu, nhãn thật)
    let mut label_channel: Vec<(u64, u32, u8)> = Vec::new();

    // Chỉ số đánh giá
    let mut correct_total = 0u64;
    let mut correct_window = 0u64;
    let mut dropped_full = 0u64; // Mẫu không vào được P vì hết bộ nhớ
    let mut expired = 0u64; // Mẫu bị loại khỏi P vì chờ nhãn quá TTL
    let mut labels_matched = 0u64;
    let mut labels_lost = 0u64; // Nhãn đến nhưng mẫu không còn trong P
    let mut trained = 0u64;

    for now in 1..=STEPS {
        // 1. Loại các mẫu chờ nhãn quá lâu
        expired += memory.expire_pending(now, PENDING_TTL) as u64;

        // 2. Nhận tín hiệu mới
        let id = now as u32;
        let (true_label, features) = make_sample(&mut rng, &centers);

        // 3. Dự đoán trước khi biết nhãn (prequential: test rồi mới train)
        let prediction = argmax(&network.run(&features));
        if prediction == true_label as usize {
            correct_total += 1;
            correct_window += 1;
        }

        // 4. Lưu mẫu vào P để chờ nhãn
        if memory.add_to_p(Sample::new(id, features, now)).is_err() {
            dropped_full += 1;
        }
        // Nhãn vẫn đến sau một độ trễ ngẫu nhiên, kể cả khi mẫu đã bị bỏ
        let arrival = now + rng.gen_range(MIN_DELAY..=MAX_DELAY);
        label_channel.push((arrival, id, true_label));

        // 5. Ghép các nhãn đến ở bước này vào mẫu trong P (P -> Q)
        let (arrived, waiting): (Vec<_>, Vec<_>) =
            label_channel.into_iter().partition(|&(t, _, _)| t <= now);
        label_channel = waiting;
        for (_, label_id, label) in arrived {
            if memory.assign_label(label_id, label).is_ok() {
                labels_matched += 1;
            } else {
                labels_lost += 1;
            }
        }

        // 6. Huấn luyện: mỗi mẫu mới trong Q học kèm REPLAY_K mẫu ngẫu nhiên từ R
        while let Some(train_sample) = memory.pop_for_training() {
            let mut batch: Vec<&Sample> = vec![&train_sample];
            if !memory.r_queue.is_empty() {
                for _ in 0..REPLAY_K {
                    batch.push(&memory.r_queue[rng.gen_range(0..memory.r_queue.len())]);
                }
            }

            let mut inputs = Vec::with_capacity(batch.len());
            let mut outputs = Vec::with_capacity(batch.len());
            for s in batch {
                let mut target = vec![0.0f32; NUM_CLASSES];
                target[s.label.expect("mau trong Q/R luon co nhan") as usize] = 1.0;
                inputs.push(s.features.to_vec());
                outputs.push(target);
            }
            network.train(&inputs, &outputs, LEARNING_RATE, 1)?;
            trained += 1;
        }

        // 7. Báo cáo định kỳ
        if now % REPORT_EVERY == 0 {
            println!(
                "[DEBUG][STATUS] Buoc {:>5} | Acc {} buoc gan nhat: {:>6.2}% | P={}, Q={}, R={} | Dung luong: {}/{} bytes",
                now,
                REPORT_EVERY,
                correct_window as f32 / REPORT_EVERY as f32 * 100.0,
                memory.p_queue.len(),
                memory.q_queue.len(),
                memory.r_queue.len(),
                memory.current_usage_bytes(),
                memory.max_memory_bytes
            );
            correct_window = 0;
        }
    }

    // Tổng kết
    println!("\n=== TONG KET MO PHONG ===");
    println!("Prequential accuracy: {:.2}%", correct_total as f32 / STEPS as f32 * 100.0);
    println!("Tong so mau: {}", STEPS);
    println!("Mau bi bo vi het bo nho P: {}", dropped_full);
    println!("Mau bi loai khoi P vi qua TTL ({} buoc): {}", PENDING_TTL, expired);
    println!("Nhan ghep thanh cong: {} | Nhan den nhung mau da mat: {} | Nhan chua den: {}",
        labels_matched, labels_lost, label_channel.len());
    println!("So mau da hoc: {} | R dang giu {}/{} mau (da tung thay {} mau)",
        trained, memory.r_queue.len(), memory.r_capacity, memory.r_seen);
    println!("Bo nho trong so mo hinh (f32, chua tinh vao ngan sach): {} bytes",
        network.total_connections() * std::mem::size_of::<f32>());

    Ok(())
}
