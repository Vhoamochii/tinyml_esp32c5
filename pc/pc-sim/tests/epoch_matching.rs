use pc_sim::sample_queue::{SampleManager, PushError, ResolveError};

#[test]
fn test_p_full_drops_and_records() {
    let mut mgr: SampleManager<2, 8> = SampleManager::new();
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
    let mut mgr: SampleManager<8, 2> = SampleManager::new();
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
    let mut mgr: SampleManager<8, 8> = SampleManager::new();
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
    let mut mgr: SampleManager<8, 8> = SampleManager::new();
    
    let res = mgr.resolve_label(99, 1);
    assert_eq!(res, Err(ResolveError::NotFound));
    assert_eq!(mgr.orphan_labels, 1);
    
    println!("[TEST][EPOCH_MATCHING][ORPHAN] expected_orphans=1 actual_orphans={} PASS", mgr.orphan_labels);
}
