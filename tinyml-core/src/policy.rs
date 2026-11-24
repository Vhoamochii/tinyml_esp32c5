// Chính sách bộ nhớ theo đề cương (mục "Giới hạn bộ nhớ" và Bảng 2):
// Khi nhận mẫu mới, dùng ô trống trước. Nếu mảng đầy, thu hồi mẫu cũ nhất ở R khi R có hơn r mẫu,
// sau đó mới xét mẫu chờ lâu nhất ở P. Q được bảo vệ. Khi hết ô hoặc Q đầy, ghi ID và lý do bỏ mẫu.
// "Mẫu cũ nhất ở R" = mẫu VÀO R sớm nhất (nhóm chốt ngày 24/11/2026).
// Các trường dùng u32 để kích thước H giống nhau trên PC (64 bit) và bo (32 bit).

use crate::event::{Event, EventKind};
use crate::memsim::{MemoryPool, SlotState};
use crate::NUM_FEATURES;

#[derive(Clone, Copy, Debug)]
pub struct MemoryPolicy {
    pub r_protected: u32, // r: số mẫu R được bảo vệ khỏi thu hồi
    pub q_max: u32,       // Giới hạn số mẫu trong Q
}

impl MemoryPolicy {
    /// Cấp ô P cho cửa sổ mới, thu hồi ô theo thứ tự của đề cương nếu cần
    pub fn admit_window<const N: usize, L: FnMut(Event)>(
        &self,
        pool: &mut MemoryPool<N>,
        now: u32,
        id: u32,
        features: [f32; NUM_FEATURES],
        log: &mut L,
    ) {
        let slot = match pool.find_free() {
            Some(idx) => Some(idx),
            None => self.reclaim(pool, now, log),
        };
        match slot {
            Some(idx) => {
                pool.put_p(idx, id, features);
                log(Event { step: now, id, kind: EventKind::Admitted });
            }
            None => log(Event { step: now, id, kind: EventKind::DroppedNoSlot }),
        }
    }

    /// Nhãn đến cho mẫu `id`
    pub fn admit_label<const N: usize, L: FnMut(Event)>(
        &self,
        pool: &mut MemoryPool<N>,
        now: u32,
        id: u32,
        label: u8,
        log: &mut L,
    ) {
        match pool.find_id(id) {
            Some(idx) if pool.slot(idx).state == SlotState::P => {
                log(Event { step: now, id, kind: EventKind::LabelMatched });
                if pool.count(SlotState::Q) as u32 >= self.q_max {
                    pool.free(idx);
                    log(Event { step: now, id, kind: EventKind::DroppedQFull });
                } else {
                    pool.to_q(idx, label);
                }
            }
            Some(_) => log(Event { step: now, id, kind: EventKind::LabelDuplicate }),
            None => log(Event { step: now, id, kind: EventKind::LabelMissed }),
        }
    }

    /// Thu hồi một ô khi mảng đầy: R vào sớm nhất (nếu R > r), sau đó P chờ lâu nhất. Không đụng Q.
    fn reclaim<const N: usize, L: FnMut(Event)>(&self, pool: &mut MemoryPool<N>, now: u32, log: &mut L) -> Option<usize> {
        if pool.count(SlotState::R) as u32 > self.r_protected {
            if let Some(idx) = pool.oldest(SlotState::R) {
                log(Event { step: now, id: pool.slot(idx).id, kind: EventKind::EvictedR });
                pool.free(idx);
                return Some(idx);
            }
        }
        if let Some(idx) = pool.oldest(SlotState::P) {
            log(Event { step: now, id: pool.slot(idx).id, kind: EventKind::EvictedP });
            pool.free(idx);
            return Some(idx);
        }
        None
    }
}
