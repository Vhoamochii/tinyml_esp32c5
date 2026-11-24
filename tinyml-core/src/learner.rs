// Phía thiết bị: mảng ô P/Q/R + chính sách bộ nhớ + lịch học + lớp cuối.
// Ubuntu (pc-sim) và ESP32-C5 (firmware) cùng gọi đúng ba hàm theo thứ tự trong mỗi bước:
//   1. on_window: dự đoán và ghi dự đoán TRƯỚC, rồi mới cấp ô P
//   2. on_label:  xử lý các nhãn đến hạn (bên gọi đưa vào theo ID tăng dần)
//   3. train:     học tối đa `max_updates` lượt theo lịch học

use crate::event::{Event, EventKind};
use crate::head::Head;
use crate::memsim::MemoryPool;
use crate::policy::{MemoryPolicy, Retention};
use crate::schedule::{Pick, Schedule, Scheduler};
use crate::NUM_FEATURES;

/// H: số byte quản lý nằm ngoài mảng ô (bộ đếm thứ tự, chính sách, lịch học)
pub const MANAGEMENT_BYTES: usize =
    core::mem::size_of::<MemoryPool<0>>() + core::mem::size_of::<MemoryPolicy>() + core::mem::size_of::<Scheduler>();

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub r_protected: u32,     // r: số mẫu R được bảo vệ
    pub q_max: u32,           // Giới hạn Q
    pub schedule: Schedule,   // Xen kẽ hoặc ưu tiên Q
    pub replay_enabled: bool, // false = đối chứng tắt học lại
    pub lr: f32,              // Tốc độ học
    pub seed: u32,            // Hạt giống cho việc chọn mẫu học lại và vé số reservoir
    pub retention: Retention, // Chọn mẫu R bị thu hồi (đề cương: Fifo)
}

pub struct Learner<const N: usize, const C: usize> {
    pub pool: MemoryPool<N>,
    pub head: Head<C>,
    pub policy: MemoryPolicy,
    pub scheduler: Scheduler,
    pub lr: f32,
}

impl<const N: usize, const C: usize> Learner<N, C> {
    pub fn new(cfg: &Config) -> Self {
        Self {
            pool: MemoryPool::new(),
            head: Head::zeros(),
            policy: MemoryPolicy {
                r_protected: cfg.r_protected,
                q_max: cfg.q_max,
                retention: cfg.retention,
                seed: cfg.seed,
            },
            scheduler: Scheduler::new(cfg.schedule, cfg.replay_enabled, N as u32, cfg.seed),
            lr: cfg.lr,
        }
    }

    /// Cửa sổ mới đến: ghi dự đoán trước, sau đó mới cấp ô P. Trả về lớp dự đoán.
    pub fn on_window<L: FnMut(Event)>(&mut self, now: u32, id: u32, features: [f32; NUM_FEATURES], log: &mut L) -> u8 {
        let pred = self.head.predict(&features) as u8;
        log(Event { step: now, id, kind: EventKind::Predicted(pred) });
        self.policy.admit_window(&mut self.pool, now, id, features, log);
        pred
    }

    /// Nhãn của mẫu `id` đến
    pub fn on_label<L: FnMut(Event)>(&mut self, now: u32, id: u32, label: u8, log: &mut L) {
        self.policy.admit_label(&mut self.pool, now, id, label, log);
    }

    /// Học tối đa `max_updates` lượt theo lịch học. Trả về số lượt đã học.
    pub fn train<L: FnMut(Event)>(&mut self, now: u32, max_updates: u32, log: &mut L) -> u32 {
        let mut done = 0;
        while done < max_updates {
            let Some(pick) = self.scheduler.next(&self.pool) else { break };
            match pick {
                Pick::New(idx) => {
                    let slot = *self.pool.slot(idx);
                    self.head.sgd_step(&slot.features, slot.label, self.lr);
                    self.pool.to_r(idx);
                    log(Event { step: now, id: slot.id, kind: EventKind::FirstLearn });
                }
                Pick::Replay(idx) => {
                    let slot = *self.pool.slot(idx);
                    self.head.sgd_step(&slot.features, slot.label, self.lr);
                    log(Event { step: now, id: slot.id, kind: EventKind::Replay });
                }
            }
            done += 1;
        }
        done
    }
}
