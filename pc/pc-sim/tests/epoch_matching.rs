use tinyol_core::sample_queue::{SampleManager, ResolveError};

#[test]
fn test_p_full_drops_and_records() {
    let mut mgr: SampleManager<2, 8, 8> = SampleManager::new(0);
    assert_eq!(mgr.push_sample([0; 16]), Ok(0));
    assert_eq!(mgr.push_sample([0; 16]), Ok(1));
    assert_eq!(mgr.push_sample([0; 16]), Ok(2)); 
    
    assert_eq!(mgr.dropped_p_full, 1);
    assert_eq!(mgr.pushed, 3);
    println!("[TEST][EPOCH_MATCHING][P_FULL] expected_drops=1 actual_drops={} PASS", mgr.dropped_p_full);
}

#[test]
fn test_q_full_drops_and_records() {
    let mut mgr: SampleManager<8, 2, 8> = SampleManager::new(0);
    mgr.push_sample([0; 16]).unwrap();
    mgr.push_sample([0; 16]).unwrap();
    mgr.push_sample([0; 16]).unwrap();
    
    mgr.resolve_label(0, 1).unwrap();
    mgr.resolve_label(1, 0).unwrap();
    mgr.resolve_label(2, 1).unwrap();
    
    assert_eq!(mgr.matched, 3);
    assert_eq!(mgr.dropped_q_full, 1);
    assert_eq!(mgr.ready.len(), 2);
    println!("[TEST][EPOCH_MATCHING][Q_FULL] expected_drops=1 actual_drops={} PASS", mgr.dropped_q_full);
}

#[test]
fn test_duplicate_label_prevented() {
    let mut mgr: SampleManager<8, 8, 8> = SampleManager::new(0);
    mgr.push_sample([0; 16]).unwrap();
    
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
    let mut mgr: SampleManager<8, 8, 8> = SampleManager::new(0);
    let res = mgr.resolve_label(99, 1);
    assert_eq!(res, Err(ResolveError::NotFound));
    assert_eq!(mgr.orphan_labels, 1);
    println!("[TEST][EPOCH_MATCHING][ORPHAN] expected_orphans=1 actual_orphans={} PASS", mgr.orphan_labels);
}

#[test]
fn test_replay_moves_to_r() {
    let mut mgr: SampleManager<8, 8, 2> = SampleManager::new(0); 
    mgr.push_sample([0; 16]).unwrap();
    mgr.push_sample([0; 16]).unwrap();
    mgr.push_sample([0; 16]).unwrap();

    mgr.resolve_label(0, 1).unwrap();
    mgr.resolve_label(1, 1).unwrap();
    mgr.resolve_label(2, 1).unwrap();

    mgr.pop_for_learning();
    mgr.pop_for_learning();
    mgr.pop_for_learning();
    assert_eq!(mgr.replay.len(), 2);
    assert_eq!(mgr.dropped_r_full, 1);
    
    let s1 = mgr.replay.pop().unwrap();
    let s2 = mgr.replay.pop().unwrap();
    assert_eq!([s1.id, s2.id], [1, 2]);
    println!("[TEST][EPOCH_MATCHING][REPLAY_EVICT] expected_drops=1 actual_drops={} PASS", mgr.dropped_r_full);
}

#[test]
fn test_eviction_policy_t6() {
    let mut mgr: SampleManager<2, 8, 8> = SampleManager::new(0);
    mgr.push_sample([0; 16]).unwrap();
    mgr.push_sample([0; 16]).unwrap();
    mgr.push_sample([0; 16]).unwrap(); 
    
    assert_eq!(mgr.dropped_p_full, 1);
    let p1 = mgr.pending.pop().unwrap();
    let p2 = mgr.pending.pop().unwrap();
    assert_eq!([p1.id, p2.id], [1, 2]);

    let mut mgr2: SampleManager<8, 8, 2> = SampleManager::new(0);
    for i in 0..4 {
        mgr2.push_sample([0; 16]).unwrap();
        mgr2.resolve_label(i, 1).unwrap();
    }
    mgr2.pop_for_learning();
    mgr2.pop_for_learning();
    mgr2.pop_for_learning();
    assert_eq!(mgr2.dropped_r_full, 1);

    println!("[TEST][EPOCH_MATCHING][EVICTION_POLICY] expected_dropsP=1 dropsR=1 actual_dropsP={} dropsR={} PASS", mgr.dropped_p_full, mgr2.dropped_r_full);
}
