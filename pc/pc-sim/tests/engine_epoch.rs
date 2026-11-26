//! Epoch test tren engine that MemPool (de cuong muc 4 Bang 3, huong dan muc 3).
//! Kiem tra: lich 10 ID nhan tre 2 buoc, thu tu su kien trong buoc, nhan trung,
//! va tinh huong day o kem nhat ky ID + ly do (ReclaimP/ReclaimR/PushFull).
use pc_sim::mem_pool::{CellState, EvictReason, LabelError, MemPool};

fn ids_of<const K: usize>(m: &MemPool<K>, st: CellState) -> Vec<u32> {
    let mut v = Vec::new();
    m.for_each_cell(|c| if c.state == st { v.push(c.id); });
    v
}

#[test]
fn bang3_10_id_tren_engine_that() {
    let mut m: MemPool<10> = MemPool::new(0);
    let mut learned: Vec<u32> = Vec::new();
    let mut events: Vec<String> = Vec::new();
    for step in 0u32..10 {
        let id = m.push([0i8; 16]).unwrap();
        events.push(format!("pred{}_s{}", id, step));
        if step >= 2 {
            m.on_label(step - 2, 0).unwrap();
            events.push(format!("label{}_s{}", step - 2, step));
        }
        while let Some(c) = m.pop_ready() {
            let lid = c.id;
            m.promote_to_replay(c);
            learned.push(lid);
            events.push(format!("learn{}_s{}", lid, step));
        }
        if step == 4 {
            let mut p = ids_of(&m, CellState::Pending);
            p.sort();
            assert_eq!(p, vec![3, 4], "P sau buoc 4 phai con dung ID 3 va 4");
        }
    }
    // Buoc 10-11: tra not nhan cho ID 8, 9
    m.on_label(8, 0).unwrap();
    while let Some(c) = m.pop_ready() { let id = c.id; m.promote_to_replay(c); learned.push(id); }
    m.on_label(9, 0).unwrap();
    while let Some(c) = m.pop_ready() { let id = c.id; m.promote_to_replay(c); learned.push(id); }
    assert_eq!(learned, (0..10u32).collect::<Vec<_>>(), "moi ID hoc dau dung mot lan");
    assert_eq!(m.learned_first, 10);
    assert_eq!(m.miss_labels, 0);
    assert_eq!(m.drop_total(), 0);
    // Buoc 5: du doan ID5 dung truoc nhan ID3, nhan ID3 dung truoc hoc ID3
    let i_pred5 = events.iter().position(|e| e == "pred5_s5").unwrap();
    let i_lab3 = events.iter().position(|e| e == "label3_s5").unwrap();
    let i_learn3 = events.iter().position(|e| e == "learn3_s5").unwrap();
    assert!(i_pred5 < i_lab3 && i_lab3 < i_learn3, "thu tu su kien buoc 5 sai");
}

#[test]
fn nhan_trung_khong_tao_hoc_dau_thu_hai() {
    let mut m: MemPool<10> = MemPool::new(0);
    m.push([0i8; 16]).unwrap();
    m.on_label(0, 1).unwrap();
    let c = m.pop_ready().unwrap();
    m.promote_to_replay(c);
    assert_eq!(m.learned_first, 1);
    assert!(matches!(m.on_label(0, 1), Err(LabelError::AlreadyLearned)));
    assert_eq!(m.learned_first, 1, "nhan trung khong duoc hoc lan hai");
}

#[test]
fn day_o_ghi_ro_id_va_ly_do() {
    // K=4, day P: push thu 5 thu hoi P cu nhat (id0), nhan cua id0 thanh miss.
    let mut m: MemPool<4> = MemPool::new(0);
    for _ in 0..4 { m.push([0i8; 16]).unwrap(); }
    assert!(m.push([0i8; 16]).is_ok());
    assert_eq!(m.drop_total(), 1);
    assert_eq!(m.drop_journal()[0], (0, EvictReason::ReclaimP));
    assert!(matches!(m.on_label(0, 0), Err(LabelError::EvictedMiss)));
    assert_eq!(m.miss_labels, 1);
    // K=3, ca ba o sang Q: pool toan Ready -> push moi bi PushFull, nhat ky ghi id 3.
    let mut q: MemPool<3> = MemPool::new(0);
    for i in 0..3u32 { q.push([0i8; 16]).unwrap(); q.on_label(i, 0).unwrap(); }
    assert_eq!(q.snapshot_pqr(), (0, 3, 0));
    assert!(q.push([0i8; 16]).is_err());
    assert_eq!(q.drop_journal()[0], (3, EvictReason::PushFull));
}

#[test]
fn r_bao_ve_mau_cu_dung_thu_tu_thu_hoi() {
    let build = |r: usize| -> MemPool<4> {
        let mut m = MemPool::<4>::new(r);
        for i in 0..3u32 {
            m.push([0i8; 16]).unwrap();
            m.on_label(i, 0).unwrap();
            let c = m.pop_ready().unwrap();
            m.promote_to_replay(c);
        }
        m.push([0i8; 16]).unwrap(); // id3 vao P, pool day
        let _ = m.push([0i8; 16]);  // id4 gay ap luc thu hoi
        m
    };
    let m0 = build(0);
    let m3 = build(3);
    assert_eq!(m0.drop_journal()[0], (0, EvictReason::ReclaimR));
    assert_eq!(m3.drop_journal()[0], (3, EvictReason::ReclaimP));
    assert!(ids_of(&m3, CellState::Replay).contains(&0), "r=3 phai giu mau cu id0");
    assert!(!ids_of(&m0, CellState::Replay).contains(&0), "r=0 da thu hoi id0");
}
