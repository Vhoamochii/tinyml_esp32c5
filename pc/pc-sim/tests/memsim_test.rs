// File kiểm thử độc lập cho module memsim
use pc_sim::memsim::{MemoryManager, Sample, SampleStatus};

// Hàm hỗ trợ tạo mẫu nhanh
fn create_dummy_sample(id: u32) -> Sample {
    Sample::new(id, [0.0; 16], 1000 + id as u64)
}

#[test]
fn test_memsim_happy_path() {
    println!("[TEST][MEMSIM][HappyPath] Bat dau...");
    // Cấp phát bộ nhớ cho 5 mẫu
    let mut manager = MemoryManager::new(5 * 80);
    
    // Thêm 2 mẫu vào P
    assert!(manager.add_to_p(create_dummy_sample(1)).is_ok());
    assert!(manager.add_to_p(create_dummy_sample(2)).is_ok());
    
    assert_eq!(manager.p_queue.len(), 2);
    assert_eq!(manager.current_usage_bytes(), 160);

    // Gán nhãn cho mẫu ID 1 -> Chuyển sang Q
    assert!(manager.assign_label(1, 1).is_ok());
    assert_eq!(manager.p_queue.len(), 1);
    assert_eq!(manager.q_queue.len(), 1);
    assert_eq!(manager.q_queue[0].status, SampleStatus::Q);

    // Lấy mẫu từ Q để train -> Chuyển sang R
    let train_sample = manager.pop_for_training().unwrap();
    assert_eq!(train_sample.id, 1);
    assert_eq!(manager.q_queue.len(), 0);
    assert_eq!(manager.r_queue.len(), 1);
    assert_eq!(manager.r_queue[0].status, SampleStatus::R);

    println!("[TEST][MEMSIM][HappyPath] PASS");
}

#[test]
fn test_memsim_edge_case_overflow() {
    println!("[TEST][MEMSIM][Overflow] Bat dau...");
    // Cấp phát bộ nhớ chỉ đủ cho 1 mẫu
    let mut manager = MemoryManager::new(80);
    
    // Thêm mẫu 1 thành công
    assert!(manager.add_to_p(create_dummy_sample(1)).is_ok());
    
    // Thêm mẫu 2 phải thất bại vì tràn bộ nhớ
    let result = manager.add_to_p(create_dummy_sample(2));
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "[MEMSIM][ERROR] Khong du bo nho de them vao P");
    
    println!("[TEST][MEMSIM][Overflow] PASS");
}

#[test]
fn test_memsim_error_invalid_id() {
    println!("[TEST][MEMSIM][InvalidID] Bat dau...");
    let mut manager = MemoryManager::new(1000);
    
    // Cố gắng gán nhãn cho ID không tồn tại trong P
    let result = manager.assign_label(99, 1);
    
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "[MEMSIM][ERROR] Khong tim thay ID trong P");
    
    println!("[TEST][MEMSIM][InvalidID] PASS");
}

#[test]
fn test_memsim_replay_capacity() {
    println!("[TEST][MEMSIM][ReplayCapacity] Bat dau...");
    // Ngân sách 5 mẫu, R chỉ giữ tối đa 2 mẫu
    let mut manager = MemoryManager::with_r_capacity(5 * 80, 2);

    // Học 20 mẫu: nếu R không bị giới hạn thì P sẽ hết chỗ từ mẫu thứ 6
    for id in 1..=20 {
        assert!(manager.add_to_p(create_dummy_sample(id)).is_ok());
        assert!(manager.assign_label(id, 0).is_ok());
        assert!(manager.pop_for_training().is_some());
    }

    assert_eq!(manager.r_queue.len(), 2);
    assert_eq!(manager.r_seen, 20);
    assert_eq!(manager.current_usage_bytes(), 2 * 80);

    println!("[TEST][MEMSIM][ReplayCapacity] PASS");
}

#[test]
fn test_memsim_expire_pending() {
    println!("[TEST][MEMSIM][ExpirePending] Bat dau...");
    let mut manager = MemoryManager::new(10 * 80);

    // Timestamp của mẫu ID n là 1000 + n
    for id in 1..=3 {
        assert!(manager.add_to_p(create_dummy_sample(id)).is_ok());
    }

    // Tại thời điểm 1005 với TTL = 2: chỉ giữ mẫu có timestamp >= 1003
    assert_eq!(manager.expire_pending(1005, 2), 2);
    assert_eq!(manager.p_queue.len(), 1);
    assert_eq!(manager.p_queue[0].id, 3);

    // Nhãn đến muộn cho mẫu đã bị loại -> không tìm thấy trong P
    let result = manager.assign_label(1, 0);
    assert_eq!(result.unwrap_err(), "[MEMSIM][ERROR] Khong tim thay ID trong P");

    println!("[TEST][MEMSIM][ExpirePending] PASS");
}
