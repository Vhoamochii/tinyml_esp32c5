// Chính sách phân bổ TẠM THỜI, chưa phải đối chứng hay chính sách đề xuất trong báo cáo:
// - P: thu hồi ô chờ nhãn quá `p_ttl` bước.
// - R: giữ tối đa `r_max` ô; khi đầy thì thay thế theo reservoir sampling.
// - Replay: chọn ngẫu nhiên đều trong R.
// Đối chứng "chia tỷ lệ cố định, cho mượn ô" và chính sách "số ô P thích ứng (định luật Little)"
// sẽ được viết thành các module cùng vai trò, dùng chung memsim.rs.

use crate::memsim::{MemoryPool, SlotState};
use crate::rng::XorShift32;

pub struct SimplePolicy {
    pub r_max: usize,
    pub p_ttl: u32,
    pub r_seen: u32, // Tổng số mẫu đã học lần đầu (dùng cho reservoir sampling)
    rng: XorShift32,
}

impl SimplePolicy {
    pub fn new(r_max: usize, p_ttl: u32, seed: u32) -> Self {
        Self { r_max, p_ttl, r_seen: 0, rng: XorShift32::new(seed) }
    }

    /// Thu hồi các ô P đã chờ nhãn quá `p_ttl` bước. Trả về số ô bị thu hồi.
    pub fn expire_pending<const N: usize>(&self, pool: &mut MemoryPool<N>, now: u32) -> usize {
        let mut expired = 0;
        for idx in 0..N {
            let slot = pool.slot(idx);
            if slot.state == SlotState::P && now.saturating_sub(slot.t_request) > self.p_ttl {
                pool.free(idx);
                expired += 1;
            }
        }
        expired
    }

    /// Gọi sau lần học đầu của ô `idx` (đang ở Q): giữ vào R hay trả ô.
    /// Reservoir sampling (Algorithm R): mỗi mẫu đã học có cùng xác suất r_max / r_seen nằm trong R.
    /// Trả về true nếu mẫu được giữ trong R.
    pub fn after_first_training<const N: usize>(&mut self, pool: &mut MemoryPool<N>, idx: usize, now: u32) -> bool {
        self.r_seen += 1;
        if pool.count(SlotState::R) < self.r_max {
            pool.move_to_r(idx, now);
            return true;
        }
        if self.r_max > 0 {
            let j = self.rng.below(self.r_seen) as usize;
            if j < self.r_max {
                // Thay thế ô R thứ j
                let victim = pool.indices(SlotState::R).nth(j);
                if let Some(victim) = victim {
                    pool.free(victim);
                }
                pool.move_to_r(idx, now);
                return true;
            }
        }
        pool.free(idx);
        false
    }

    /// Chọn ngẫu nhiên một ô R để replay
    pub fn pick_replay<const N: usize>(&mut self, pool: &MemoryPool<N>) -> Option<usize> {
        let n = pool.count(SlotState::R);
        if n == 0 {
            return None;
        }
        let j = self.rng.below(n as u32) as usize;
        pool.indices(SlotState::R).nth(j)
    }
}
