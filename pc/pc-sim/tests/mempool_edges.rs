//! B2: edge case tren engine MemPool that (khong phai sample_queue minh hoa).
//! Ngu nghia Q khai bao: FIFO theo thu tu push (on_label khong doi age).
//! Day chinh la ngu nghia oracle da nghiem thu; neu de cuong muon FIFO theo
//! thu tu nhan-ve-Q thi phai doi engine va nghiem thu lai toan bo so.

use pc_sim::mem_pool::{DropReason, EvictReason, LabelError, MemPool};

fn z(id: u32) -> [i8; 16] {
    let mut v = [0i8; 16];
    v[0] = (id as i8) + 1;
    v
}

/// Happy: nhan ve khac thu tu van ghep dung ID; Q phuc vu theo thu tu push.
#[test]
fn out_of_order_labels_match_correct_id() {
    let mut p: MemPool<8> = MemPool::new(2);
    for i in 0..4 { assert_eq!(p.push(z(i)).unwrap(), i); }
    // Nhan ve xao tron: 2, 0, 3, 1
    assert!(p.on_label(2, 2).is_ok());
    assert!(p.on_label(0, 0).is_ok());
    assert!(p.on_label(3, 0).is_ok());
    assert!(p.on_label(1, 1).is_ok());
    assert_eq!(p.matched, 4);
    assert_eq!(p.miss_labels, 0);
    let mut order = Vec::new();
    while let Some(c) = p.pop_ready() {
        order.push(c.id);
        p.promote_to_replay(c);
    }
    // Khai bao: FIFO theo push -> 0,1,2,3 du nhan ve 2,0,3,1
    assert_eq!(order, vec![0, 1, 2, 3]);
    assert_eq!(p.learned_first, 4);
    println!("[TEST][MEMPOOL][OOO_LABELS] expected_order=[0,1,2,3] actual={:?} PASS", order);
}

/// Edge: thu hoi R (khi R > r) TRUOC P; dung ID cu nhat; journal ghi dung.
#[test]
fn reclaim_priority_r_before_p_with_journal() {
    let mut p: MemPool<4> = MemPool::new(1);
    for i in 0..4 { p.push(z(i)).unwrap(); }
    // Hoc 0,1 -> R=2, P={2,3}
    for id in [0u32, 1] {
        p.on_label(id, 0).unwrap();
        let c = p.pop_ready().unwrap();
        p.promote_to_replay(c);
    }
    // Push 4: het o trong, R=2 > r=1 -> thu hoi R cu nhat (ID 0)
    assert_eq!(p.push(z(4)).unwrap(), 4);
    assert_eq!(p.recovered_r, 1);
    assert_eq!(p.recovered_p, 0);
    // Push 5: R=1 khong > 1 -> thu hoi P lau nhat (ID 2)
    assert_eq!(p.push(z(5)).unwrap(), 5);
    assert_eq!(p.recovered_p, 1);
    // Nhan cua ID 2 ve sau khi bi thu hoi -> EvictedMiss
    assert_eq!(p.on_label(2, 2), Err(LabelError::EvictedMiss));
    assert_eq!(p.miss_labels, 1);
    let j = p.drop_journal();
    assert_eq!(j, &[(0, EvictReason::ReclaimR), (2, EvictReason::ReclaimP)]);
    assert_eq!(p.drop_total(), 2);
    println!("[TEST][MEMPOOL][RECLAIM_PRIO] expected=[(0,R),(2,P)] actual={:?} PASS", j);
}

/// Error: toan Q -> push Err(PoolFull), Q khong bi duoi, journal ghi ID du kien.
#[test]
fn q_protected_pool_full_logged() {
    let mut p: MemPool<4> = MemPool::new(0);
    for i in 0..4 { p.push(z(i)).unwrap(); }
    for i in 0..4 { p.on_label(i, 0).unwrap(); } // Q=4, P=0, R=0
    assert_eq!(p.push(z(9)), Err(DropReason::PoolFull));
    assert_eq!(p.drop_journal(), &[(4, EvictReason::PushFull)]);
    // Q con nguyen: hoc du 4 mau
    let mut n = 0;
    while let Some(c) = p.pop_ready() { p.promote_to_replay(c); n += 1; }
    assert_eq!(n, 4);
    assert_eq!(p.learned_first, 4);
    println!("[TEST][MEMPOOL][Q_PROTECTED] expected_push=Err(PoolFull) learned=4 actual=4 PASS");
}

/// Error: nhan trung tren Q va tren R khong tao lan hoc dau thu hai; nhan mo coi.
#[test]
fn duplicate_and_orphan_labels() {
    let mut p: MemPool<8> = MemPool::new(2);
    p.push(z(0)).unwrap();
    p.on_label(0, 1).unwrap();
    // Trung khi con o Q
    assert_eq!(p.on_label(0, 1), Err(LabelError::AlreadyLabeled));
    let c = p.pop_ready().unwrap();
    p.promote_to_replay(c);
    // Trung khi da o R
    assert_eq!(p.on_label(0, 1), Err(LabelError::AlreadyLearned));
    // Mo coi
    assert_eq!(p.on_label(99, 0), Err(LabelError::EvictedMiss));
    assert_eq!(p.dup_labels, 2);
    assert_eq!(p.miss_labels, 1);
    assert_eq!(p.learned_first, 1); // khong co lan hoc dau thu hai
    println!("[TEST][MEMPOOL][DUP_ORPHAN] expected_dup=2 orphan=1 learned=1 actual=2/1/1 PASS");
}
