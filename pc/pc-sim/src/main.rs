// Bộ phát luồng + chấm điểm trên PC.
// Phía "PC" (giữ nhãn đúng, hẹn giờ trả nhãn, chấm điểm) nằm ở file này.
// Phía "thiết bị" (vùng nhớ P/Q/R, chính sách, lớp cuối) dùng tinyml-core, cùng mã với firmware.

use tinyml_core::head::Head;
use tinyml_core::memsim::{slots_for_budget, MemoryPool, SlotState, SLOT_SIZE_BYTES};
use tinyml_core::policy::SimplePolicy;
use tinyml_core::rng::XorShift32;
use tinyml_core::{NUM_CLASSES, NUM_FEATURES};

// Tham số thí nghiệm
const STEPS: u32 = 3000;
const MEMORY_BUDGET_BYTES: usize = 2000;
const N: usize = slots_for_budget(MEMORY_BUDGET_BYTES); // Tổng số ô
const R_MAX: usize = N / 2; // R dùng tối đa một nửa số ô
const NOISE: f32 = 0.6; // Biên độ nhiễu đều quanh tâm cụm của mỗi lớp
const MIN_DELAY: u32 = 1; // Nhãn đến trễ ngẫu nhiên trong [MIN_DELAY, MAX_DELAY] bước
const MAX_DELAY: u32 = 12;
const PENDING_TTL: u32 = 10; // Ô P chờ nhãn quá số bước này bị thu hồi
const LEARNING_RATE: f32 = 0.1;
const REPLAY_K: usize = 2; // Số mẫu từ R học kèm mỗi mẫu mới
const REPORT_EVERY: u32 = 500;
const SEED: u32 = 42;

// Sinh một mẫu: chọn lớp ngẫu nhiên, đặc trưng = tâm cụm của lớp + nhiễu
fn make_sample(rng: &mut XorShift32, centers: &[[f32; NUM_FEATURES]; NUM_CLASSES]) -> (u8, [f32; NUM_FEATURES]) {
    let class = rng.below(NUM_CLASSES as u32) as usize;
    let mut features = [0.0f32; NUM_FEATURES];
    for (f, c) in features.iter_mut().zip(centers[class].iter()) {
        *f = c + rng.range_f32(-NOISE, NOISE);
    }
    (class as u8, features)
}

fn main() {
    println!("[DEBUG][MAIN] Bat dau mo phong TinyML nhan tre ({} buoc, seed={})", STEPS, SEED);
    println!(
        "[DEBUG][MAIN] Moi o {} byte, ngan sach {} byte -> {} o (R toi da {} o)",
        SLOT_SIZE_BYTES, MEMORY_BUDGET_BYTES, N, R_MAX
    );

    // Phía PC: dữ liệu và lịch trả nhãn
    let mut data_rng = XorShift32::new(SEED);
    let mut centers = [[0.0f32; NUM_FEATURES]; NUM_CLASSES];
    for center in centers.iter_mut() {
        for v in center.iter_mut() {
            *v = data_rng.range_f32(-1.0, 1.0);
        }
    }
    // Kênh nhãn trễ: (bước nhãn đến, ID mẫu, nhãn thật)
    let mut label_channel: Vec<(u32, u32, u8)> = Vec::new();

    // Phía thiết bị: vùng nhớ, chính sách, lớp cuối
    let mut pool: MemoryPool<N> = MemoryPool::new();
    let mut policy = SimplePolicy::new(R_MAX, PENDING_TTL, SEED.wrapping_add(1));
    let mut head = Head::zeros();

    // Chỉ số đánh giá
    let mut correct_total = 0u32;
    let mut correct_window = 0u32;
    let mut dropped_full = 0u32; // Mẫu không được cấp ô P vì hết ô
    let mut expired = 0u32; // Ô P bị thu hồi vì chờ nhãn quá TTL
    let mut labels_matched = 0u32;
    let mut labels_lost = 0u32; // Nhãn đến nhưng mẫu không còn trong P (ghép trượt)
    let mut trained = 0u32;

    for now in 1..=STEPS {
        // 1. Thu hồi các ô P chờ nhãn quá lâu
        expired += policy.expire_pending(&mut pool, now) as u32;

        // 2. Nhận tín hiệu mới (mã mẫu = số bước)
        let id = now;
        let (true_label, features) = make_sample(&mut data_rng, &centers);

        // 3. Dự đoán trước khi biết nhãn (prequential: test rồi mới train)
        if head.predict(&features) == true_label as usize {
            correct_total += 1;
            correct_window += 1;
        }

        // 4. Cấp ô P cho mẫu; nhãn vẫn đến sau một độ trễ ngẫu nhiên, kể cả khi mẫu bị bỏ
        if pool.alloc_p(id, features, now).is_err() {
            dropped_full += 1;
        }
        let arrival = now + MIN_DELAY + data_rng.below(MAX_DELAY - MIN_DELAY + 1);
        label_channel.push((arrival, id, true_label));

        // 5. Ghép các nhãn đến ở bước này (P -> Q)
        let (arrived, waiting): (Vec<_>, Vec<_>) =
            label_channel.into_iter().partition(|&(t, _, _)| t <= now);
        label_channel = waiting;
        for (_, label_id, label) in arrived {
            if pool.assign_label(label_id, label, now).is_ok() {
                labels_matched += 1;
            } else {
                labels_lost += 1;
            }
        }

        // 6. Huấn luyện: mỗi mẫu trong Q + tối đa REPLAY_K mẫu từ R, một bước SGD trên lớp cuối
        while let Some(idx) = pool.oldest_q() {
            let mut features_batch = [[0.0f32; NUM_FEATURES]; 1 + REPLAY_K];
            let mut labels_batch = [0u8; 1 + REPLAY_K];
            features_batch[0] = pool.slot(idx).features;
            labels_batch[0] = pool.slot(idx).label;
            let mut n = 1;
            for _ in 0..REPLAY_K {
                if let Some(r) = policy.pick_replay(&pool) {
                    features_batch[n] = pool.slot(r).features;
                    labels_batch[n] = pool.slot(r).label;
                    n += 1;
                }
            }
            let batch: Vec<(&[f32; NUM_FEATURES], u8)> =
                (0..n).map(|i| (&features_batch[i], labels_batch[i])).collect();
            head.sgd_step(&batch, LEARNING_RATE);
            trained += 1;

            // Chính sách quyết định giữ mẫu vừa học vào R hay trả ô
            policy.after_first_training(&mut pool, idx, now);
        }

        // 7. Báo cáo định kỳ
        if now % REPORT_EVERY == 0 {
            println!(
                "[DEBUG][STATUS] Buoc {:>5} | Acc {} buoc gan nhat: {:>6.2}% | P={}, Q={}, R={} | Dung luong: {}/{} bytes",
                now,
                REPORT_EVERY,
                correct_window as f32 / REPORT_EVERY as f32 * 100.0,
                pool.count(SlotState::P),
                pool.count(SlotState::Q),
                pool.count(SlotState::R),
                pool.usage_bytes(),
                pool.capacity_bytes()
            );
            correct_window = 0;
        }
    }

    // Tổng kết
    println!("\n=== TONG KET MO PHONG ===");
    println!("Prequential accuracy: {:.2}%", correct_total as f32 / STEPS as f32 * 100.0);
    println!("Tong so mau: {}", STEPS);
    println!("Mau khong duoc cap o P vi het o: {}", dropped_full);
    println!("O P bi thu hoi vi qua TTL ({} buoc): {}", PENDING_TTL, expired);
    println!(
        "Nhan ghep thanh cong: {} | Ghep truot: {} | Nhan chua den: {}",
        labels_matched,
        labels_lost,
        label_channel.len()
    );
    println!(
        "So mau da hoc lan dau: {} | R dang giu {}/{} o (da tung thay {} mau)",
        trained,
        pool.count(SlotState::R),
        policy.r_max,
        policy.r_seen
    );
    println!("Bo nho tham so lop cuoi (chua tinh vao ngan sach): {} bytes", Head::PARAM_BYTES);
}
