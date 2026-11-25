//! E1+E1e+E2b-a: luong bang tren du lieu that.
//! - Reserve (N15, record >= 17) tach khoi luong/pool/hoc lai, chi dung cham quen.
//! - BA moi chi cham tren cua so N09, ngay khi den, TRUOC cap nhat.
//! - Seed chi dieu khien quay so replay; thu tu CSV bat bien giua cac case.
//! - E1b: RollingBa cua so 60; phuc hoi cham lan dau khi cua so day dat >= ba_before.
//! - E1e: phuc hoi BEN VUNG = SUSTAIN_W cua so day lien tiep dat >= ba_before.
//! - E2b-a: ham generic theo KP (kich thuoc pool); dem push_drops khi pool day.
//! Logic hoc giu nguyen byte-for-byte so voi D6/D7.

use crate::head_linear::{LinearHead, Quant, N_FEATURES};
use crate::mem_pool::MemPool;
use crate::schedule::{build, ScheduleMode};
use std::collections::BTreeMap;
use std::fs;
use std::time::Instant;

pub const K: usize = 256;
pub const LR: f32 = 0.1;
pub const DEADLINE_US: u64 = 16_000;
pub const DEFAULT_SEED: u32 = 0x9E37_79B9;
pub const ROLL_W: usize = 60;
pub const SUSTAIN_W: usize = 10; // E1e: so cua so day lien tiep de goi la ben vung

/// Q1: lich hoc lai. Interleaved = ban cu, giu bit-for-bit oracle.
/// QFirst: chi hoc lai khi Q rong; moi lan hoc dau tao 1 credit, tran K.
/// Off: doi chung tat hoc lai; R van chiem o (r=0 khong phai Off).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ReplayMode { Interleaved, QFirst, Off }

pub struct Row {
    pub bearing: String,
    pub cls: u8,
    pub cond: String,
    pub record: u32,
    pub win: u32,
    pub z: [i8; N_FEATURES],
}

pub struct RunResult {
    pub tag: String,
    pub preds: Vec<(String, u8, u8)>,
    pub ba_new: f64,
    pub ba_before: f64,
    pub ba_after: f64,
    pub forget: f64,
    pub match_pct: f64,
    pub ram_proxy: u64,
    pub p95_us: u64,
    pub over_pct: f64,
    pub learned_first: u32,
    pub replay_learns: u32,
    pub miss_labels: u32,
    pub n_stream: usize,
    pub n_reserve: usize,
    pub seed: u32,
    pub recovery_windows: Option<usize>,
    pub recovery_sustained_windows: Option<usize>, // E1e
    pub push_drops: u32,                            // E2b-a
    pub final_slide_ba: f64,
    pub lat_p50: u64,
    pub lat_p95: u64,
    pub per_bearing: Vec<(String, u8, u32, u32)>,
}

/// BA truot O(1), dem vong tron, chi tra Some khi du ba lop.
pub struct RollingBa {
    buf: [(u8, bool); ROLL_W],
    pos: usize,
    len: usize,
    tot: [u32; 3],
    ok: [u32; 3],
}

impl RollingBa {
    pub const fn new() -> Self {
        Self { buf: [(0, false); ROLL_W], pos: 0, len: 0, tot: [0; 3], ok: [0; 3] }
    }
    pub fn push(&mut self, class: u8, correct: bool) {
        if class as usize >= 3 { return; }
        if self.len == ROLL_W {
            let (c, o) = self.buf[self.pos];
            self.tot[c as usize] -= 1;
            if o { self.ok[c as usize] -= 1; }
        } else {
            self.len += 1;
        }
        self.buf[self.pos] = (class, correct);
        self.tot[class as usize] += 1;
        if correct { self.ok[class as usize] += 1; }
        self.pos = (self.pos + 1) % ROLL_W;
    }
    pub fn is_full(&self) -> bool { self.len == ROLL_W }
    pub fn ba(&self) -> Option<f64> {
        if self.tot[0] == 0 || self.tot[1] == 0 || self.tot[2] == 0 { return None; }
        let mut s = 0.0;
        for c in 0..3 { s += self.ok[c] as f64 / self.tot[c] as f64; }
        Some(100.0 * s / 3.0)
    }
}

pub fn percentile(vals: &[u64], p: usize) -> u64 {
    if vals.is_empty() { return 0; }
    let mut v = vals.to_vec();
    v.sort_unstable();
    let idx = ((v.len() * p) / 100).min(v.len() - 1);
    v[idx]
}

pub fn load_csv(path: &str) -> Vec<Row> {
    let text = fs::read_to_string(path).expect("khong doc duoc CSV dac trung");
    let mut rows = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 21 { continue; }
        let mut z = [0i8; N_FEATURES];
        for i in 0..N_FEATURES {
            z[i] = f[5 + i].parse().unwrap_or(0);
        }
        rows.push(Row {
            bearing: f[0].to_string(),
            cls: f[1].parse().unwrap(),
            cond: f[2].to_string(),
            record: f[3].parse().unwrap(),
            win: f[4].parse().unwrap(),
            z,
        });
    }
    rows
}

pub fn load_quant(path: &str) -> Quant {
    let text = fs::read_to_string(path)
        .expect("khong doc duoc feature_scales.csv; chay make_feature_csv.py truoc");
    let mut q = Quant::identity();
    let mut seen = 0usize;
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 2 { continue; }
        let dim: usize = f[0].trim().parse().expect("dim khong hop le");
        let s: f32 = f[1].trim().parse().expect("scale khong hop le");
        if dim < N_FEATURES && s.is_finite() && s > 0.0 {
            q.scale[dim] = s;
            seen += 1;
        }
    }
    assert!(seen == N_FEATURES, "feature_scales.csv thieu chieu: seen={seen}");
    q
}

pub fn balanced_accuracy(preds: &[(String, u8, u8)]) -> f64 {
    use std::collections::HashMap;
    let mut per: HashMap<(String, u8), (u32, u32)> = HashMap::new();
    for (b, y, p) in preds {
        let e = per.entry((b.clone(), *y)).or_insert((0, 0));
        e.0 += 1;
        if y == p { e.1 += 1; }
    }
    let mut by_class: HashMap<u8, Vec<f64>> = HashMap::new();
    for ((_, y), (tot, ok)) in per {
        by_class.entry(y).or_default().push(ok as f64 / tot as f64);
    }
    let mut accs = Vec::new();
    for (_, v) in by_class {
        accs.push(v.iter().sum::<f64>() / v.len() as f64);
    }
    if accs.is_empty() { 0.0 } else { accs.iter().sum::<f64>() / accs.len() as f64 }
}

fn class_recalls(preds: &[(String, u8, u8)]) -> [f64; 3] {
    let mut tot = [0u32; 3];
    let mut ok = [0u32; 3];
    for (_, y, p) in preds {
        let c = *y as usize;
        if c < 3 {
            tot[c] += 1;
            if y == p { ok[c] += 1; }
        }
    }
    let mut r = [0.0f64; 3];
    for c in 0..3 {
        r[c] = if tot[c] > 0 { 100.0 * ok[c] as f64 / tot[c] as f64 } else { 0.0 };
    }
    r
}

pub fn p95_us(samples: &[u64]) -> u64 {
    if samples.is_empty() { return 0; }
    let mut v = samples.to_vec();
    v.sort_unstable();
    v[(v.len() * 95) / 100]
}

fn eval_reserve_detail(reserve: &[&Row], head: &LinearHead<3>, q: &Quant) -> (f64, [f64; 3]) {
    let mut fp: Vec<(String, u8, u8)> = Vec::new();
    for r in reserve {
        fp.push((r.bearing.clone(), r.cls, head.predict(r.z, q)));
    }
    (balanced_accuracy(&fp) * 100.0, class_recalls(&fp))
}

fn head_norm(head: &LinearHead<3>) -> f32 {
    let mut s = 0.0f32;
    for c in 0..3 {
        for i in 0..N_FEATURES { s += head.w[c][i] * head.w[c][i]; }
        s += head.b[c] * head.b[c];
    }
    s.sqrt()
}

fn is_reserve(r: &Row) -> bool {
    r.cond == "N15_M07_F10" && r.record >= 17
}

/// Wrapper giu chu ky cu: K=256, seed mac dinh, de bang Case 1 tai lap dung.
pub fn run_case(
    rows: &[Row], r_floor: usize, sched: ScheduleMode, label: &str, q: &Quant,
) -> RunResult {
    run_case_seeded::<K>(rows, r_floor, sched, label, q, DEFAULT_SEED)
}

/// E2b-a: generic theo KP de quat kich thuoc pool (ap luc bo nho).
/// Wrapper giu chu ky cu: mac dinh Interleaved de tai lap oracle.
pub fn run_case_seeded<const KP: usize>(
    rows: &[Row], r_floor: usize, sched: ScheduleMode, label: &str, q: &Quant, seed: u32,
) -> RunResult {
    run_case_replay::<KP>(rows, r_floor, sched, label, q, seed, ReplayMode::Interleaved)
}

/// Q1: ban day du co chon lich hoc lai.
pub fn run_case_replay<const KP: usize>(
    rows: &[Row], r_floor: usize, sched: ScheduleMode, label: &str, q: &Quant, seed: u32,
    mode: ReplayMode,
) -> RunResult {
    let tag = format!("{} r={} K={} {}", sched_desc(sched), r_floor, KP, label);
    let reserve: Vec<&Row> = rows.iter().filter(|r| is_reserve(r)).collect();
    let stream: Vec<&Row> = rows.iter().filter(|r| !is_reserve(r)).collect();

    let mut pool: MemPool<KP> = MemPool::new(r_floor);
    let mut head: LinearHead<3> = LinearHead::zeros();
    let labels_true: Vec<u8> = stream.iter().map(|r| r.cls).collect();
    let events = build(stream.len(), sched, &labels_true);
    let mut preds: Vec<(String, u8, u8)> = Vec::new();
    let mut times: Vec<u64> = Vec::new();
    let mut rnd: u32 = if seed == 0 { DEFAULT_SEED } else { seed };
    let mut ei = 0usize;
    let mut da_doi = false;
    let mut ba_before = 0.0f64;
    let mut max_abs_z = 0.0f32;
    let mut push_drops: u32 = 0;
    let mut credits: usize = 0; // Q1: QFirst

    let mut roll = RollingBa::new();
    let mut n09_seen = 0usize;
    let mut recovery_windows: Option<usize> = None;
    let mut rec_sus: Option<usize> = None;
    let mut consec: usize = 0;
    let mut final_slide_ba = 0.0f64;

    let mut label_step: Vec<u32> = vec![u32::MAX; stream.len()];
    let mut latencies: Vec<u64> = Vec::new();

    for (step, row) in stream.iter().enumerate() {
        let t0 = Instant::now();
        for i in 0..N_FEATURES {
            let v = (row.z[i] as f32).abs() * q.scale[i];
            if v > max_abs_z { max_abs_z = v; }
        }
        if row.cond == "N09_M07_F10" && !da_doi {
            da_doi = true;
            let snap = head.snapshot();
            let (ba, rec) = eval_reserve_detail(&reserve, &snap, q);
            ba_before = ba;
            println!(
                "[TEST][TABLE][{}][RECALL_BEFORE] c0={:.1} c1={:.1} c2={:.1} ba={:.2} w_norm={:.3e} max_abs_z={:.3e}",
                tag, rec[0], rec[1], rec[2], ba, head_norm(&snap), max_abs_z
            );
        }
        if row.cond == "N09_M07_F10" {
            let p = head.predict(row.z, q);
            preds.push((row.bearing.clone(), row.cls, p));
            n09_seen += 1;
            roll.push(row.cls, p == row.cls);
            if roll.is_full() {
                if let Some(b) = roll.ba() {
                    final_slide_ba = b;
                    if recovery_windows.is_none() && b >= ba_before {
                        recovery_windows = Some(n09_seen);
                    }
                    // E1e: chuoi cua so day lien tiep dat nguong
                    if b >= ba_before {
                        consec += 1;
                        if rec_sus.is_none() && consec >= SUSTAIN_W {
                            rec_sus = Some(n09_seen);
                        }
                    } else {
                        consec = 0;
                    }
                }
            }
        }
        // E2b-a: dem mau bi bo vi pool day thay vi bo im
        if pool.push(row.z).is_err() {
            push_drops += 1;
        }
        while ei < events.len() && events[ei].0 == step {
            let (_, id, lb) = events[ei];
            if pool.on_label(id, lb).is_ok() && (id as usize) < label_step.len() {
                label_step[id as usize] = step as u32;
            }
            ei += 1;
        }
        while let Some(c) = pool.pop_ready() {
            if (c.id as usize) < label_step.len() && label_step[c.id as usize] != u32::MAX {
                latencies.push(step.saturating_sub(label_step[c.id as usize] as usize) as u64);
            }
            head.sgd_step(c.z, q, c.label.max(0) as u8, LR);
            pool.promote_to_replay(c);
            match mode {
                ReplayMode::Interleaved => {
                    if let Some(rc) = pool.replay_draw(&mut rnd) {
                        head.sgd_step(rc.z, q, rc.label.max(0) as u8, LR);
                    }
                }
                ReplayMode::QFirst => { credits = (credits + 1).min(KP); }
                ReplayMode::Off => {}
            }
        }
        // Q1: QFirst chi hoc lai khi Q da rong; moi credit tieu 1 luot.
        if mode == ReplayMode::QFirst {
            while credits > 0 {
                credits -= 1;
                if let Some(rc) = pool.replay_draw(&mut rnd) {
                    head.sgd_step(rc.z, q, rc.label.max(0) as u8, LR);
                }
            }
        }
        times.push(t0.elapsed().as_micros() as u64);
    }

    let mut dstep = stream.len();
    while ei < events.len() {
        let (_, id, lb) = events[ei];
        if pool.on_label(id, lb).is_ok() && (id as usize) < label_step.len() {
            label_step[id as usize] = dstep as u32;
        }
        ei += 1;
        while let Some(c) = pool.pop_ready() {
            if (c.id as usize) < label_step.len() && label_step[c.id as usize] != u32::MAX {
                latencies.push(dstep.saturating_sub(label_step[c.id as usize] as usize) as u64);
            }
            head.sgd_step(c.z, q, c.label.max(0) as u8, LR);
            pool.promote_to_replay(c);
            match mode {
                ReplayMode::Interleaved => {} // giu ban cu: cuoi luong khong hoc lai
                ReplayMode::QFirst => { credits = (credits + 1).min(KP); }
                ReplayMode::Off => {}
            }
        }
        if mode == ReplayMode::QFirst {
            while credits > 0 {
                credits -= 1;
                if let Some(rc) = pool.replay_draw(&mut rnd) {
                    head.sgd_step(rc.z, q, rc.label.max(0) as u8, LR);
                }
            }
        }
        dstep += 1;
    }

    let (ba_after, rec_after) = eval_reserve_detail(&reserve, &head, q);
    println!(
        "[TEST][TABLE][{}][RECALL_AFTER] c0={:.1} c1={:.1} c2={:.1} ba={:.2} w_norm={:.3e} max_abs_z={:.3e}",
        tag, rec_after[0], rec_after[1], rec_after[2], ba_after, head_norm(&head), max_abs_z
    );
    let forget = if da_doi { ba_before - ba_after } else { 0.0 };

    let mut pb: BTreeMap<(String, u8), (u32, u32)> = BTreeMap::new();
    for (b, y, p) in &preds {
        let e = pb.entry((b.clone(), *y)).or_insert((0, 0));
        e.1 += 1;
        if y == p { e.0 += 1; }
    }
    let per_bearing: Vec<(String, u8, u32, u32)> =
        pb.into_iter().map(|((b, y), (ok, tot))| (b, y, ok, tot)).collect();

    let n_req = stream.len() as f64;
    let over = times.iter().filter(|t| **t > DEADLINE_US).count() as u64;
    RunResult {
        tag,
        ba_new: balanced_accuracy(&preds) * 100.0,
        preds,
        ba_before,
        ba_after,
        forget,
        match_pct: if n_req > 0.0 { 100.0 * pool.matched as f64 / n_req } else { 0.0 },
        ram_proxy: (KP * MemPool::<KP>::cell_size_bytes()) as u64,
        p95_us: p95_us(&times),
        over_pct: if n_req > 0.0 { 100.0 * over as f64 / n_req } else { 0.0 },
        learned_first: pool.learned_first,
        replay_learns: pool.replay_learns,
        miss_labels: pool.miss_labels,
        n_stream: stream.len(),
        n_reserve: reserve.len(),
        seed,
        recovery_windows,
        recovery_sustained_windows: rec_sus,
        push_drops,
        final_slide_ba,
        lat_p50: percentile(&latencies, 50),
        lat_p95: percentile(&latencies, 95),
        per_bearing,
    }
}

fn sched_desc(s: ScheduleMode) -> &'static str {
    match s { ScheduleMode::Spread => "roi", ScheduleMode::Burst => "dot" }
}
