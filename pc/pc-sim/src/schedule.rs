use std::vec::Vec;

/// Bo sinh lich tra nhan theo de cuong muc 5: nhom 9 yeu cau lien tiep.
/// Lich roi: yeu cau vi tri j co tre j buoc. Lich dot: tre 8-j,
/// ca 9 nhan cung ve tai mot buoc. Trung binh moi lich = 4 buoc.

pub const GROUP: usize = 9;

#[derive(Clone, Copy, PartialEq)]
pub enum ScheduleMode { Spread, Burst }

/// Vec (step_den, id, label) sap theo step_den tang dan.
pub fn build(n_requests: usize, mode: ScheduleMode, labels: &[u8]) -> Vec<(usize, u32, u8)> {
    let mut out = Vec::new();
    let mut g = 0usize;
    while g < n_requests {
        let m = (n_requests - g).min(GROUP);
        for j in 0..m {
            let id = (g + j) as u32;
            let delay = match mode {
                ScheduleMode::Spread => j,
                ScheduleMode::Burst => (GROUP - 1) - j, // 8 - j
            };
            let arrive = id as usize + delay;
            out.push((arrive, id, labels[id as usize]));
        }
        g += m;
    }
    out.sort_by_key(|e| (e.0, e.1)); // cung buoc: xu ly theo ID tang dan
    out
}

/// Test cong thuc: moi lich dung n nhan, trung binh tre = 4 (voi nhom du 9).
pub fn mean_delay(n_requests: usize, mode: ScheduleMode, labels: &[u8]) -> f64 {
    let ev = build(n_requests, mode, labels);
    if ev.is_empty() { return 0.0; }
    let sum: usize = ev.iter().map(|(a, id, _)| a - *id as usize).sum();
    sum as f64 / ev.len() as f64
}
