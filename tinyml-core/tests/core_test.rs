// Kiểm thử độc lập cho tinyml-core: vùng nhớ P/Q/R, chính sách tạm thời, lớp cuối và bộ sinh ngẫu nhiên
use tinyml_core::head::Head;
use tinyml_core::memsim::{slots_for_budget, MemError, MemoryPool, SlotState, SLOT_SIZE_BYTES};
use tinyml_core::policy::SimplePolicy;
use tinyml_core::rng::XorShift32;
use tinyml_core::NUM_FEATURES;

#[test]
fn test_memsim_slot_size() {
    println!("[TEST][MEMSIM][SlotSize] Bat dau...");
    // 16 f32 (64) + id, t_request, t_state (12) + label, state (2) = 78 byte, căn chỉnh 4 -> 80
    assert_eq!(SLOT_SIZE_BYTES, 80);
    assert_eq!(slots_for_budget(2000), 25);
    println!("[TEST][MEMSIM][SlotSize] PASS");
}

#[test]
fn test_memsim_happy_path() {
    println!("[TEST][MEMSIM][HappyPath] Bat dau...");
    let mut pool: MemoryPool<5> = MemoryPool::new();
    let mut policy = SimplePolicy::new(2, 100, 1);

    // Cấp 2 ô cho P
    assert!(pool.alloc_p(1, [0.0; NUM_FEATURES], 10).is_ok());
    assert!(pool.alloc_p(2, [0.0; NUM_FEATURES], 11).is_ok());
    assert_eq!(pool.count(SlotState::P), 2);
    assert_eq!(pool.usage_bytes(), 2 * SLOT_SIZE_BYTES);

    // Nhãn của mẫu 1 đến -> P chuyển sang Q
    let idx = pool.assign_label(1, 1, 12).unwrap();
    assert_eq!(pool.count(SlotState::P), 1);
    assert_eq!(pool.count(SlotState::Q), 1);
    assert_eq!(pool.slot(idx).label, 1);

    // Học lần đầu -> chính sách giữ vào R (R còn chỗ); số ô dùng không đổi
    assert_eq!(pool.oldest_q(), Some(idx));
    assert!(policy.after_first_training(&mut pool, idx, 13));
    assert_eq!(pool.count(SlotState::Q), 0);
    assert_eq!(pool.count(SlotState::R), 1);
    assert_eq!(pool.usage_bytes(), 2 * SLOT_SIZE_BYTES);

    println!("[TEST][MEMSIM][HappyPath] PASS");
}

#[test]
fn test_memsim_edge_case_overflow() {
    println!("[TEST][MEMSIM][Overflow] Bat dau...");
    // Vùng nhớ chỉ có 1 ô
    let mut pool: MemoryPool<1> = MemoryPool::new();
    assert!(pool.alloc_p(1, [0.0; NUM_FEATURES], 0).is_ok());

    let result = pool.alloc_p(2, [0.0; NUM_FEATURES], 0);
    assert_eq!(result, Err(MemError::Full));
    assert_eq!(MemError::Full.as_str(), "[MEMSIM][ERROR] Khong con o trong de cap cho P");

    println!("[TEST][MEMSIM][Overflow] PASS");
}

#[test]
fn test_memsim_error_invalid_id() {
    println!("[TEST][MEMSIM][InvalidID] Bat dau...");
    let mut pool: MemoryPool<4> = MemoryPool::new();

    let result = pool.assign_label(99, 1, 0);
    assert_eq!(result, Err(MemError::IdNotFound));
    assert_eq!(MemError::IdNotFound.as_str(), "[MEMSIM][ERROR] Khong tim thay ID trong P");

    println!("[TEST][MEMSIM][InvalidID] PASS");
}

#[test]
fn test_policy_replay_capacity() {
    println!("[TEST][POLICY][ReplayCapacity] Bat dau...");
    // 5 ô, R giữ tối đa 2 ô
    let mut pool: MemoryPool<5> = MemoryPool::new();
    let mut policy = SimplePolicy::new(2, 100, 7);

    // Học 20 mẫu: nếu R không bị giới hạn thì P sẽ hết ô từ mẫu thứ 6
    for id in 1..=20 {
        assert!(pool.alloc_p(id, [0.0; NUM_FEATURES], id).is_ok());
        let idx = pool.assign_label(id, 0, id).unwrap();
        policy.after_first_training(&mut pool, idx, id);
    }

    assert_eq!(pool.count(SlotState::R), 2);
    assert_eq!(policy.r_seen, 20);
    assert_eq!(pool.usage_bytes(), 2 * SLOT_SIZE_BYTES);
    assert!(policy.pick_replay(&pool).is_some());

    println!("[TEST][POLICY][ReplayCapacity] PASS");
}

#[test]
fn test_policy_expire_pending() {
    println!("[TEST][POLICY][ExpirePending] Bat dau...");
    let mut pool: MemoryPool<10> = MemoryPool::new();
    let policy = SimplePolicy::new(5, 2, 1);

    // Mẫu ID n được tạo ở bước 1000 + n
    for id in 1..=3 {
        assert!(pool.alloc_p(id, [0.0; NUM_FEATURES], 1000 + id).is_ok());
    }

    // Tại bước 1005 với TTL = 2: chỉ giữ mẫu tạo từ bước 1003 trở đi
    assert_eq!(policy.expire_pending(&mut pool, 1005), 2);
    assert_eq!(pool.count(SlotState::P), 1);

    // Nhãn đến muộn cho mẫu đã bị thu hồi -> ghép trượt
    assert_eq!(pool.assign_label(1, 0, 1006), Err(MemError::IdNotFound));
    assert!(pool.assign_label(3, 0, 1006).is_ok());

    println!("[TEST][POLICY][ExpirePending] PASS");
}

#[test]
fn test_head_learns_separable_classes() {
    println!("[TEST][HEAD][Separable] Bat dau...");
    // Mỗi lớp c bật đặc trưng thứ c
    let mut x = [[0.0f32; NUM_FEATURES]; 3];
    for (c, xc) in x.iter_mut().enumerate() {
        xc[c] = 1.0;
    }

    let mut head = Head::zeros();
    let p0 = head.probs(&x[0]);
    assert!((p0.iter().sum::<f32>() - 1.0).abs() < 1e-6);

    let batch = [(&x[0], 0u8), (&x[1], 1u8), (&x[2], 2u8)];
    let first_loss = head.sgd_step(&batch, 0.5);
    let mut last_loss = first_loss;
    for _ in 0..200 {
        last_loss = head.sgd_step(&batch, 0.5);
    }

    assert!(last_loss < first_loss);
    for (c, xc) in x.iter().enumerate() {
        assert_eq!(head.predict(xc), c);
    }

    println!("[TEST][HEAD][Separable] PASS");
}

#[test]
fn test_rng_deterministic() {
    println!("[TEST][RNG][Deterministic] Bat dau...");
    let mut a = XorShift32::new(42);
    let mut b = XorShift32::new(42);
    for _ in 0..100 {
        assert_eq!(a.next_u32(), b.next_u32());
        let v = a.range_f32(-1.0, 1.0);
        assert!((-1.0..1.0).contains(&v));
        b.range_f32(-1.0, 1.0);
        assert!(a.below(12) < 12);
        b.below(12);
    }
    println!("[TEST][RNG][Deterministic] PASS");
}
