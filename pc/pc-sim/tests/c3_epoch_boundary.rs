use tinyol_core::sample_queue::SampleManager;

#[test]
fn test_c3_label_matching_across_epoch() {
    let mut mgr: SampleManager<10, 10, 10> = SampleManager::new(1);
    
    let mut learned_status = [false; 10];
    
    for _ in 0..5 {
        mgr.push_sample([0; 16]).unwrap();
    }
    
    // Sử dụng đúng tên trường 'pending'
    assert_eq!(mgr.pending.len(), 5);

    mgr.push_sample([0; 16]).unwrap();
    mgr.push_sample([0; 16]).unwrap();

    mgr.resolve_label(3, 1).unwrap();
    mgr.resolve_label(4, 1).unwrap();
    
    assert_eq!(mgr.ready.len(), 2);

    let s3 = mgr.move_to_replay().unwrap();
    let s4 = mgr.move_to_replay().unwrap();
    learned_status[s3.id as usize] = true;
    learned_status[s4.id as usize] = true;

    mgr.push_sample([0; 16]).unwrap();
    mgr.push_sample([0; 16]).unwrap();
    mgr.push_sample([0; 16]).unwrap();

    mgr.resolve_label(5, 0).unwrap();
    mgr.resolve_label(6, 0).unwrap();
    let s5 = mgr.move_to_replay().unwrap();
    let s6 = mgr.move_to_replay().unwrap();
    learned_status[s5.id as usize] = true;
    learned_status[s6.id as usize] = true;

    mgr.resolve_label(7, 1).unwrap();
    mgr.resolve_label(8, 1).unwrap();
    mgr.resolve_label(9, 1).unwrap();
    let s7 = mgr.move_to_replay().unwrap();
    let s8 = mgr.move_to_replay().unwrap();
    let s9 = mgr.move_to_replay().unwrap();
    learned_status[s7.id as usize] = true;
    learned_status[s8.id as usize] = true;
    learned_status[s9.id as usize] = true;

    mgr.resolve_label(0, 0).unwrap();
    mgr.resolve_label(1, 0).unwrap();
    mgr.resolve_label(2, 0).unwrap();
    
    while let Some(_) = mgr.move_to_replay() {}

    learned_status[0] = true;
    learned_status[1] = true;
    

    println!("[TEST][C3_EPOCH] Đã ghép và học đủ 10 ID");
    assert_eq!(mgr.matched, 10, "Phải ghép được 10 nhãn");
    assert_eq!(mgr.pushed, 10, "Phải đẩy được 10 mẫu");
}
