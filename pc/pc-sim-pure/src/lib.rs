//! Case 2: Rust thuan standalone. Vong lap rieng, doc lap voi stream_runner,
//! de kiem chung parity tung ID giua hai hinh trien khai (de cuong muc 6:
//! cung o cung hat tao thanh cap so sanh).
//! Thuat toan giong Case 1 by construction; parity test canh troi giua hai ben.

use pc_sim::head_linear::{LinearHead, Quant};
use pc_sim::mem_pool::MemPool;
use pc_sim::schedule::{build, ScheduleMode};
use pc_sim::stream_runner::{balanced_accuracy, Row};

pub const PURE_SEED: u32 = 0x9E37_79B9;

pub struct PureResult {
    pub preds: Vec<(String, u8, u8)>,
    pub ba_new: f64,
    pub ba_before: f64,
    pub ba_after: f64,
    pub forget: f64,
    pub match_pct: f64,
    pub learned_first: u32,
    pub replay_learns: u32,
    pub miss_labels: u32,
}

fn is_reserve(r: &Row) -> bool {
    r.cond == "N15_M07_F10" && r.record >= 17
}

fn eval_reserve(reserve: &[&Row], head: &LinearHead<3>, q: &Quant) -> f64 {
    let mut fp: Vec<(String, u8, u8)> = Vec::new();
    for r in reserve {
        fp.push((r.bearing.clone(), r.cls, head.predict(r.z, q)));
    }
    balanced_accuracy(&fp) * 100.0
}

/// Vong lap Case 2: khong dung stream_runner, tu quan ly pool/lich/hoc.
/// Thu tu thao tac phai khop stream_runner tung buoc de parity dung.
pub fn pure_run<const KP: usize>(
    rows: &[Row], r_floor: usize, sched: ScheduleMode, q: &Quant, seed: u32,
) -> PureResult {
    let reserve: Vec<&Row> = rows.iter().filter(|r| is_reserve(r)).collect();
    let stream: Vec<&Row> = rows.iter().filter(|r| !is_reserve(r)).collect();

    let mut pool: MemPool<KP> = MemPool::new(r_floor);
    let mut head: LinearHead<3> = LinearHead::zeros();
    let labels_true: Vec<u8> = stream.iter().map(|r| r.cls).collect();
    let events = build(stream.len(), sched, &labels_true);
    let mut preds: Vec<(String, u8, u8)> = Vec::new();
    let mut rnd: u32 = if seed == 0 { PURE_SEED } else { seed };
    let mut ei = 0usize;
    let mut da_doi = false;
    let mut ba_before = 0.0f64;

    for (step, row) in stream.iter().enumerate() {
        if row.cond == "N09_M07_F10" && !da_doi {
            da_doi = true;
            let snap = head.snapshot();
            ba_before = eval_reserve(&reserve, &snap, q);
        }
        if row.cond == "N09_M07_F10" {
            let p = head.predict(row.z, q);
            preds.push((row.bearing.clone(), row.cls, p));
        }
        let _ = pool.push(row.z);
        while ei < events.len() && events[ei].0 == step {
            let (_, id, lb) = events[ei];
            let _ = pool.on_label(id, lb);
            ei += 1;
        }
        while let Some(c) = pool.pop_ready() {
            head.sgd_step(c.z, q, c.label.max(0) as u8, pc_sim::stream_runner::LR);
            pool.promote_to_replay(c);
            if let Some(rc) = pool.replay_draw(&mut rnd) {
                head.sgd_step(rc.z, q, rc.label.max(0) as u8, pc_sim::stream_runner::LR);
            }
        }
    }
    while ei < events.len() {
        let (_, id, lb) = events[ei];
        let _ = pool.on_label(id, lb);
        ei += 1;
        while let Some(c) = pool.pop_ready() {
            head.sgd_step(c.z, q, c.label.max(0) as u8, pc_sim::stream_runner::LR);
            pool.promote_to_replay(c);
        }
    }
    let ba_after = eval_reserve(&reserve, &head, q);
    let n = stream.len() as f64;
    PureResult {
        ba_new: balanced_accuracy(&preds) * 100.0,
        forget: if da_doi { ba_before - ba_after } else { 0.0 },
        match_pct: if n > 0.0 { 100.0 * pool.matched as f64 / n } else { 0.0 },
        learned_first: pool.learned_first,
        replay_learns: pool.replay_learns,
        miss_labels: pool.miss_labels,
        preds,
        ba_before,
        ba_after,
    }
}
