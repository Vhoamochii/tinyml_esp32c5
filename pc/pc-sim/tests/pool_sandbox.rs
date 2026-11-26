//! Sandbox pool chung K o theo de cuong muc 3. KHONG tich hop vao core that.
//! Nghiem thu: epoch 10 ID (Bang 3), nhan trung, Q day, va quy tac thu hoi r.
//! Log [TEST][Pool] expected/actual PASS/FAIL.
#![allow(dead_code)]

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum St { Pending, Ready, Replay }

#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub id: u32,
    pub label: Option<u8>,
    pub feats: [i8; 16],
    pub st: St,
    pub seq: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PErr { NoSlot, QFull, AlreadyLearned, AlreadyLabeled, NotFound }

pub struct PoolManager<const K: usize> {
    buf: [Option<Slot>; K],
    pub q_cap: usize,
    pub r_min: usize,
    seq: u32,
    pub pushed: u32,
    pub matched: u32,
    pub learned_first: u32,
    pub replay_learns: u32,
    pub evict_r: u32,
    pub evict_p: u32,
    pub drop_no_slot: u32,
    pub drop_q_full: u32,
    pub dup: u32,
    pub orphan: u32,
    pub r_min_seen: usize,
    pub r_below: u32,
    completed: u64,
}

impl<const K: usize> PoolManager<K> {
    pub fn new(q_cap: usize, r_min: usize) -> Self {
        Self {
            buf: [None; K], q_cap, r_min, seq: 0,
            pushed: 0, matched: 0, learned_first: 0, replay_learns: 0,
            evict_r: 0, evict_p: 0, drop_no_slot: 0, drop_q_full: 0,
            dup: 0, orphan: 0, r_min_seen: K, r_below: 0, completed: 0,
        }
    }
    fn cnt(&self, st: St) -> usize {
        self.buf.iter().filter(|s| matches!(s, Some(x) if x.st == st)).count()
    }
    fn idx_free(&self) -> Option<usize> { self.buf.iter().position(|s| s.is_none()) }
    fn idx_oldest(&self, st: St) -> Option<usize> {
        let mut best: Option<(u32, usize)> = None;
        for (i, s) in self.buf.iter().enumerate() {
            if let Some(x) = s {
                if x.st == st && (best.is_none() || x.seq < best.unwrap().0) {
                    best = Some((x.seq, i));
                }
            }
        }
        best.map(|(_, i)| i)
    }
    fn idx_by_id(&self, id: u32, st: St) -> Option<usize> {
        self.buf.iter().position(|s| matches!(s, Some(x) if x.id == id && x.st == st))
    }
    fn track_r(&mut self) {
        let r = self.cnt(St::Replay);
        if r < self.r_min_seen { self.r_min_seen = r; }
        if r <= self.r_min { self.r_below += 1; }
    }
    pub fn snapshot(&self) -> (usize, usize, usize) {
        (self.cnt(St::Pending), self.cnt(St::Ready), self.cnt(St::Replay))
    }
    pub fn r_by_class(&self) -> [usize; 3] {
        let mut c = [0usize; 3];
        for s in self.buf.iter().flatten() {
            if s.st == St::Replay {
                if let Some(l) = s.label { if (l as usize) < 3 { c[l as usize] += 1; } }
            }
        }
        c
    }
    pub fn push_sample(&mut self, feats: [i8; 16]) -> Result<u32, PErr> {
        let id = self.pushed;
        let mut free = self.idx_free();
        if free.is_none() {
            // De cuong: thu hoi R cu nhat khi R > r_min, neu khong moi den P. Q bao ve.
            if self.cnt(St::Replay) > self.r_min {
                if let Some(i) = self.idx_oldest(St::Replay) { self.buf[i] = None; self.evict_r += 1; }
            } else if self.cnt(St::Pending) > 0 {
                if let Some(i) = self.idx_oldest(St::Pending) { self.buf[i] = None; self.evict_p += 1; }
            }
            free = self.idx_free();
        }
        let Some(i) = free else {
            self.drop_no_slot += 1; self.pushed += 1; return Err(PErr::NoSlot);
        };
        self.buf[i] = Some(Slot { id, label: None, feats, st: St::Pending, seq: self.seq });
        self.seq += 1; self.pushed += 1;
        self.track_r();
        Ok(id)
    }
    pub fn resolve_label(&mut self, id: u32, label: u8) -> Result<(), PErr> {
        if id < 64 && (self.completed & (1u64 << id)) != 0 {
            self.dup += 1; return Err(PErr::AlreadyLearned);
        }
        if let Some(i) = self.idx_by_id(id, St::Pending) {
            if self.buf[i].unwrap().label.is_some() { self.dup += 1; return Err(PErr::AlreadyLabeled); }
            if self.cnt(St::Ready) >= self.q_cap {
                self.buf[i] = None; self.drop_q_full += 1; return Err(PErr::QFull);
            }
            let mut s = self.buf[i].unwrap();
            s.label = Some(label); s.st = St::Ready; self.buf[i] = Some(s);
            self.matched += 1;
            return Ok(());
        }
        if self.idx_by_id(id, St::Ready).is_some() || self.idx_by_id(id, St::Replay).is_some() {
            self.dup += 1; return Err(PErr::AlreadyLabeled);
        }
        self.orphan += 1; Err(PErr::NotFound)
    }
    pub fn learn_first(&mut self) -> Option<Slot> {
        let i = self.idx_oldest(St::Ready)?;
        let mut s = self.buf[i].unwrap();
        s.st = St::Replay; self.buf[i] = Some(s);
        self.learned_first += 1;
        if s.id < 64 { self.completed |= 1u64 << s.id; }
        self.track_r();
        Some(s)
    }
    pub fn replay_one(&mut self, rng: &mut u32, skip: Option<u32>) -> Option<Slot> {
        let cand: Vec<usize> = self.buf.iter().enumerate()
            .filter(|(_, s)| matches!(s, Some(x) if x.st == St::Replay && Some(x.id) != skip))
            .map(|(i, _)| i).collect();
        if cand.is_empty() { return None; }
        *rng ^= *rng << 13; *rng ^= *rng >> 17; *rng ^= *rng << 5;
        let pick = cand[(*rng as usize) % cand.len()];
        self.replay_learns += 1;
        self.buf[pick]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn chk(name: &str, exp: String, act: String) {
        let ok = exp == act;
        println!("[TEST][Pool][{}] expected={} actual={} {}", name, exp, act, if ok { "PASS" } else { "FAIL" });
        assert!(ok, "{}", name);
    }

    #[test]
    fn epoch_10_id_theo_bang3() {
        let mut m = PoolManager::<10>::new(10, 0);
        let mut learn_order: Vec<u32> = Vec::new();
        for step in 0u32..10 {
            let _ = m.push_sample([0i8; 16]);
            if step >= 2 { let _ = m.resolve_label(step - 2, 0); }
            if let Some(s) = m.learn_first() { learn_order.push(s.id); }
            if step == 4 {
                let mut p: Vec<u32> = m.buf.iter().flatten()
                    .filter(|s| s.st == St::Pending).map(|s| s.id).collect();
                p.sort();
                chk("pending_sau_buoc4", "[3, 4]".into(), format!("{:?}", p));
            }
        }
        let _ = m.resolve_label(8, 0); if let Some(s) = m.learn_first() { learn_order.push(s.id); }
        let _ = m.resolve_label(9, 0); if let Some(s) = m.learn_first() { learn_order.push(s.id); }
        chk("learned_first", "10".into(), m.learned_first.to_string());
        let mut uniq = learn_order.clone(); uniq.sort(); uniq.dedup();
        chk("moi_id_hoc_dung_1_lan", "10".into(), uniq.len().to_string());
    }

    #[test]
    fn nhan_trung_khong_hoc_lai_lan_dau() {
        let mut m = PoolManager::<4>::new(4, 0);
        let _ = m.push_sample([0i8; 16]);
        let _ = m.resolve_label(0, 1);
        let _ = m.learn_first();
        let r = m.resolve_label(0, 1);
        chk("dup_err", "AlreadyLearned".into(), format!("{:?}", r.unwrap_err()));
        chk("learned_first_giu_nguyen", "1".into(), m.learned_first.to_string());
    }

    #[test]
    fn q_day_bo_mau_ghi_ly_do() {
        let mut m = PoolManager::<4>::new(1, 0);
        let _ = m.push_sample([0i8; 16]);
        let _ = m.push_sample([0i8; 16]);
        let _ = m.resolve_label(0, 0);
        let r = m.resolve_label(1, 0);
        chk("q_full_err", "QFull".into(), format!("{:?}", r.unwrap_err()));
        chk("drop_q_full", "1".into(), m.drop_q_full.to_string());
        let _ = m.learn_first();
        chk("learned_first", "1".into(), m.learned_first.to_string());
    }

    #[test]
    fn r_policy_that_su_rang_buoc() {
        // K=4, q_cap=1. Tao R={0,1,2}, P={3} roi day them id4.
        // r_min=0: R=3>0 -> thu hoi R (mat id0). r_min=3: R=3 khong>3 -> thu hoi P (mat id3).
        let build = |r_min: usize| -> PoolManager<4> {
            let mut m = PoolManager::<4>::new(1, r_min);
            for id in 0u32..3 {
                let _ = m.push_sample([0i8; 16]);
                let _ = m.resolve_label(id, 0);
                let _ = m.learn_first();
            }
            let _ = m.push_sample([0i8; 16]); // id3 -> P
            let _ = m.push_sample([0i8; 16]); // id4 -> gay ap luc
            m
        };
        let m0 = build(0);
        let m3 = build(3);
        chk("r0_evict_r", "1".into(), m0.evict_r.to_string());
        chk("r0_evict_p", "0".into(), m0.evict_p.to_string());
        chk("r3_evict_r", "0".into(), m3.evict_r.to_string());
        chk("r3_evict_p", "1".into(), m3.evict_p.to_string());
        let giu_id0 = m3.buf.iter().flatten().any(|s| s.id == 0);
        chk("r3_giu_mau_cu_id0", "true".into(), giu_id0.to_string());
        let mat_id0 = !m0.buf.iter().flatten().any(|s| s.id == 0);
        chk("r0_mat_mau_cu_id0", "true".into(), mat_id0.to_string());
    }

    #[test]
    fn bien_k1_va_pool_rong() {
        let mut m = PoolManager::<1>::new(1, 0);
        chk("rong_learn_none", "true".into(), m.learn_first().is_none().to_string());
        let _ = m.push_sample([0i8; 16]);
        let _ = m.push_sample([0i8; 16]); // day -> thu hoi P cu
        chk("k1_pushed", "2".into(), m.pushed.to_string());
        chk("k1_evict_p", "1".into(), m.evict_p.to_string());
    }
}
