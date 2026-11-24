// Lịch học (đề cương, mục "Tạo cùng độ trễ nhưng khác thời điểm dồn nhãn"):
// - Xen kẽ: học lại một mẫu cũ sau mỗi lần học mẫu mới.
// - Ưu tiên Q: chỉ học lại khi Q rỗng.
// Mỗi lần học đầu tạo một lượt học lại, bộ đếm tối đa K. Mẫu học lại được chọn đều trong R,
// bỏ qua ID vừa học ở lượt liền trước. Q được học theo thứ tự vào hàng.

use crate::memsim::{MemoryPool, SlotState};
use crate::rng::XorShift32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Interleaved, // Xen kẽ
    QPriority,   // Ưu tiên Q
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    New(usize),    // Học lần đầu ô Q này
    Replay(usize), // Học lại ô R này
}

pub struct Scheduler {
    pub schedule: Schedule,
    pub replay_enabled: bool, // false = đối chứng tắt học lại
    pub credits: u32,         // Số lượt học lại đang có
    credit_cap: u32,          // Tối đa K lượt
    last_learned: Option<u32>,
    last_was_new: bool,
    rng: XorShift32,
}

impl Scheduler {
    pub fn new(schedule: Schedule, replay_enabled: bool, credit_cap: u32, seed: u32) -> Self {
        Self {
            schedule,
            replay_enabled,
            credits: 0,
            credit_cap,
            last_learned: None,
            last_was_new: false,
            rng: XorShift32::new(seed),
        }
    }

    /// Chọn lượt học tiếp theo; None nếu không còn gì để học
    pub fn next<const N: usize>(&mut self, pool: &MemoryPool<N>) -> Option<Pick> {
        let replay_first = self.schedule == Schedule::Interleaved && self.last_was_new;
        let pick = if replay_first {
            match self.pick_replay(pool) {
                Some(p) => Some(p),
                None => pool.oldest(SlotState::Q).map(Pick::New),
            }
        } else {
            match pool.oldest(SlotState::Q) {
                Some(idx) => Some(Pick::New(idx)),
                None => self.pick_replay(pool),
            }
        };

        match pick {
            Some(Pick::New(idx)) => {
                self.last_was_new = true;
                self.last_learned = Some(pool.slot(idx).id);
                self.credits = (self.credits + 1).min(self.credit_cap);
            }
            Some(Pick::Replay(idx)) => {
                self.last_was_new = false;
                self.last_learned = Some(pool.slot(idx).id);
                self.credits -= 1;
            }
            None => {}
        }
        pick
    }

    /// Chọn đều một ô R, bỏ qua ID vừa học ở lượt liền trước
    fn pick_replay<const N: usize>(&mut self, pool: &MemoryPool<N>) -> Option<Pick> {
        if !self.replay_enabled || self.credits == 0 {
            return None;
        }
        let last = self.last_learned;
        let eligible = |idx: &usize| Some(pool.slot(*idx).id) != last;
        let n = pool.indices(SlotState::R).filter(eligible).count();
        if n == 0 {
            return None;
        }
        let j = self.rng.below(n as u32) as usize;
        pool.indices(SlotState::R).filter(eligible).nth(j).map(Pick::Replay)
    }
}
