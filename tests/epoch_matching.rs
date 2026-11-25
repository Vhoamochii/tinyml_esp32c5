use pc_sim::sample_queue::{SampleManager, PushError, ResolveError};

#[test]
fn test_p_full_drops_and_records() {
    // P=2, Q=8, R=8
    let mut mgr: SampleManager<2, 8, 8> = SampleManager::new();
    assert_eq!(mgr.push_sample([0; 16]), Ok(0));
    assert_eq!(mgr.push_sample([0; 16]), Ok(1));
    
    let res = mgr.push_sample([0; 16]);
    assert_eq!(res, Err(PushError::Full));
    
    assert_eq!(mgr.dropped_p_full, 1);
    assert_eq!(mgr.pushed, 2);
    println!("[TEST][EPOCH_MATCHING][P_FULL] expected_drops=1 actual_drops={} PASS", mgr.dropped_p_full);
}

#[test]
fn test_q_full_drops_and_records() {
    // P=8, Q=2, R=8
    let mut mgr: SampleManager<8, 2, 8> = SampleManager::new();
    mgr.push_sample([0; 16]).unwrap(); // ID 0
    mgr.push_sample([0; 16]).unwrap(); // ID 1
    mgr.push_sample([0; 16]).unwrap(); // ID 2
    
    assert_eq!(mgr.resolve_label(0, 1), Ok(()));
    assert_eq!(mgr.resolve_label(1, 0), Ok(()));
    assert_eq!(mgr.resolve_label(2, 1), Ok(()));
    
    assert_eq!(mgr.matched, 3);
    assert_eq!(mgr.dropped_q_full, 1);
    assert_eq!(mgr.ready.len(), 2);
    println!("[TEST][EPOCH_MATCHING][Q_FULL] expected_drops=1 actual_drops={} PASS", mgr.dropped_q_full);
}

#[test]
fn test_duplicate_label_prevented() {
    let mut mgr: SampleManager<8, 8, 8> = SampleManager::new();
    mgr.push_sample([0; 16]).unwrap(); // ID 0
    
    assert_eq!(mgr.resolve_label(0, 1), Ok(()));
    assert_eq!(mgr.resolve_label(0, 1), Err(ResolveError::AlreadyLabeled));
    assert_eq!(mgr.dup_labels, 1);
    
    let _ = mgr.pop_for_learning();
    
    assert_eq!(mgr.resolve_label(0, 1), Err(ResolveError::AlreadyLearned));
    assert_eq!(mgr.dup_labels, 2);
    
    assert_eq!(mgr.learned_first, 1);
    println!("[TEST][EPOCH_MATCHING][DUP_LABEL] expected_dups=2 actual_dups={} PASS", mgr.dup_labels);
}

#[test]
fn test_orphan_label_recorded() {
    let mut mgr: SampleManager<8, 8, 8> = SampleManager::new();
    let res = mgr.resolve_label(99, 1);
    assert_eq!(res, Err(ResolveError::NotFound));
    assert_eq!(mgr.orphan_labels, 1);
    println!("[TEST][EPOCH_MATCHING][ORPHAN] expected_orphans=1 actual_orphans={} PASS", mgr.orphan_labels);
}

#[test]
fn test_replay_moves_to_r() {
    // Kiem tra sau khi hoc, mau phai nam trong R
    let mut mgr: SampleManager<8, 8, 2> = SampleManager::new(); // R chi chua duoc 2 mau
    mgr.push_sample([0; 16]).unwrap(); // ID 0
    mgr.push_sample([0; 16]).unwrap(); // ID 1
    mgr.push_sample([0; 16]).unwrap(); // ID 2

    mgr.resolve_label(0, 1).unwrap();
    mgr.resolve_label(1, 1).unwrap();
    mgr.resolve_label(2, 1).unwrap();

    // Hoc ID 0 -> R co 1 mau
    mgr.pop_for_learning();
    assert_eq!(mgr.replay.len(), 1);
    
    // Hoc ID 1 -> R co 2 mau
    mgr.pop_for_learning();
    assert_eq!(mgr.replay.len(), 2);

    // Hoc ID 2 -> R bi tran (chi chua duoc 2), phai drop mau cu nhat (ID 0)
    mgr.pop_for_learning();
    assert_eq!(mgr.replay.len(), 2);
    assert_eq!(mgr.dropped_r_full, 1);
    
    // Kiem tra xem mau ID 0 co bi day ra khong (mau trong R hien tai phai la ID 1 va 2)
    let mut ids = [0; 2];
    let s1 = mgr.replay.pop().unwrap();
    let s2 = mgr.replay.pop().unwrap();
    ids[0] = s1.id;
    ids[1] = s2.id;
    assert_eq!(ids, [1, 2]);

    println!("[TEST][EPOCH_MATCHING][REPLAY_EVICT] expected_drops=1 actual_drops={} PASS", mgr.dropped_r_full);
}
