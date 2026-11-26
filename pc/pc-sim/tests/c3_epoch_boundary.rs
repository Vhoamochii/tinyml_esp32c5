//! B3: Bang 3 de cuong — 10 ID, nhan tre 2 buoc, epoch 5 cua so.
//! Moc buoc 4 phai kiem tra SAU vong pop cung buoc: vi du doan ghi truoc cap
//! nhat, lan hoc dau cua ID 2 xay ra trong chinh buoc 4.
//! So mau con trong P suy ra tu bo dem: pending = pushed - matched - recovered_p
//! (test nay K=16 du o nen khong co mau bi bo).

use pc_sim::head_linear::{LinearHead, Quant, N_FEATURES};
use pc_sim::mem_pool::MemPool;

const LR: f32 = 0.1;

fn z(id: u32) -> [i8; N_FEATURES] {
    let mut v = [0i8; N_FEATURES];
    v[0] = (id as i8) + 1;
    v
}
fn label(id: u32) -> u8 { (id % 3) as u8 }

/// Head tham chieu doc lap: zeros + cac buoc SGD theo dung thu tu hoc that.
fn ref_head(n_steps: u32) -> LinearHead<3> {
    let mut h: LinearHead<3> = LinearHead::zeros();
    let q = Quant::identity();
    for id in 0..n_steps {
        h.sgd_step(z(id), &q, label(id), LR);
    }
    h
}

#[test]
fn test_c3_label_matching_across_epoch() {
    let mut pool: MemPool<16> = MemPool::new(4); // du o de quan sat rieng loi ghep
    let mut head: LinearHead<3> = LinearHead::zeros();
    let q = Quant::identity();
    let mut journal: Vec<(u32, u8)> = Vec::new(); // (id, pred) tai buoc phat
    let mut learn_order: Vec<u32> = Vec::new();

    for step in 0u32..12 {
        // 1. Phat cua so + ghi du doan TRUOC moi xu ly nhan/hoc
        if step < 10 {
            let pred = head.predict(z(step), &q);
            journal.push((step, pred));
            assert_eq!(pool.push(z(step)).unwrap(), step);
        }
        // 2. Nhan tre 2 buoc: tai buoc s tra nhan ID s-2
        if step >= 2 && step - 2 < 10 {
            let lid = step - 2;
            assert!(pool.on_label(lid, label(lid)).is_ok(), "ghep truot ID {lid}");
        }
        // 3. Hoc Q theo FIFO; cap nhat chi dien ra SAU khi du doan da ghi
        while let Some(c) = pool.pop_ready() {
            learn_order.push(c.id);
            head.sgd_step(c.z, &q, c.label.max(0) as u8, LR);
            pool.promote_to_replay(c);
        }
        // Moc Bang 3 buoc 3-4: cuoi epoch 1, sau xu ly buoc 4:
        // ID 0,1,2 da hoc lan dau; ID 3,4 con trong P vi chua co nhan.
        if step == 4 {
            assert_eq!(pool.pushed, 5);
            assert_eq!(pool.matched, 3);
            assert_eq!(pool.learned_first, 3);
            assert_eq!(pool.miss_labels, 0);
            let pending = pool.pushed - pool.matched - pool.recovered_p;
            assert_eq!(pending, 2, "P phai con dung 2 mau (ID 3,4) sau buoc 4");
            println!("[TEST][C3_EPOCH][MOC_3_4] expected=P{{3,4}} learned=3 pending=2 actual=3/{pending} PASS");
        }
        // Moc 5-6: dau epoch 2, ghep lan luot ID 3 roi ID 4
        if step == 5 {
            assert_eq!(pool.matched, 4);
            assert_eq!(learn_order.last(), Some(&3));
        }
        if step == 6 {
            assert_eq!(pool.matched, 5);
            assert_eq!(learn_order.last(), Some(&4));
        }
    }

    // Moc 10-11: hai nhan cuoi tra trong vong lap; het Q thi 10 ID hoc dung 1 lan
    assert_eq!(learn_order, (0..10).collect::<Vec<u32>>());
    assert_eq!(pool.learned_first, 10);
    assert_eq!(pool.miss_labels, 0);
    assert_eq!(journal.len(), 10);

    // Du doan ghi TRUOC cap nhat: pred ID 3 chi thay head hoc ID 0;
    // pred ID 5 chi thay head hoc ID 0,1,2. Doi chieu head tham chieu doc lap.
    assert_eq!(journal[0].1, 0, "head zeros phai chon lop 0");
    assert_eq!(journal[3].1, ref_head(1).predict(z(3), &q));
    assert_eq!(journal[5].1, ref_head(3).predict(z(5), &q));
    println!("[TEST][C3_EPOCH][PRED_BEFORE] expected=ref_head(1,3) actual=khop PASS");

    // Nhan trung ID 3 (da o R) khong tao lan hoc dau thu hai
    let dup = pool.on_label(3, label(3));
    assert!(dup.is_err());
    assert_eq!(pool.learned_first, 10);
    println!("[TEST][C3_EPOCH][DUP] expected=err+learned=10 actual=err+10 PASS");
    println!("[TEST][C3_EPOCH] Da ghep va hoc du 10 ID");
}
