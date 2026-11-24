// Chính sách bộ nhớ theo đề cương (mục "Giới hạn bộ nhớ" và Bảng 2):
// Khi nhận mẫu mới, dùng ô trống trước. Nếu mảng đầy, thu hồi mẫu cũ nhất ở R khi R có hơn r mẫu,
// sau đó mới xét mẫu chờ lâu nhất ở P. Q được bảo vệ. Khi hết ô hoặc Q đầy, ghi ID và lý do bỏ mẫu.
// "Mẫu cũ nhất ở R" = mẫu VÀO R sớm nhất (nhóm chốt ngày 24/11/2026).
// Các trường dùng u32 để kích thước H giống nhau trên PC (64 bit) và bo (32 bit).
// Retention::Reservoir là THÍ NGHIỆM PHỤ ngoài đề cương, để nhóm cân nhắc khi định nghĩa 3 case.

use crate::event::{Event, EventKind};
use crate::memsim::{MemoryPool, SlotState};
use crate::NUM_FEATURES;

/// Cách chọn mẫu R bị thu hồi khi cần ô
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retention {
    /// Theo đề cương: thu hồi mẫu vào R sớm nhất (vào trước ra trước). R luôn chứa các mẫu mới nhất,
    /// nên sau khi đổi tốc độ một thời gian, R không còn mẫu tốc độ cũ.
    Fifo,
    /// Thí nghiệm phụ: mỗi ID có một "vé số" cố định reservoir_key(ID); thu hồi mẫu R có vé nhỏ nhất.
    /// R giữ các mẫu có vé lớn nhất trong mọi mẫu từng vào R, tức một mẫu ngẫu nhiên đều của toàn bộ
    /// lịch sử, nên vẫn còn mẫu tốc độ cũ sau khi đổi tốc độ.
    Reservoir,
}

/// Vé số của một ID: hàm trộn bit fmix32 của MurmurHash3. Là song ánh nên hai ID không bao giờ trùng vé.
/// Chỉ dùng phép tính số nguyên nên PC và bo cho cùng kết quả.
pub fn reservoir_key(id: u32) -> u32 {
    let mut h = id;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryPolicy {
    pub r_protected: u32,     // r: số mẫu R được bảo vệ khỏi thu hồi
    pub q_max: u32,           // Giới hạn số mẫu trong Q
    pub retention: Retention, // Chọn mẫu R bị thu hồi (đề cương: Fifo)
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

    /// Thu hồi một ô khi mảng đầy: một mẫu R (nếu R > r; mẫu nào do `retention` quyết định),
    /// sau đó P chờ lâu nhất. Không đụng Q.
    fn reclaim<const N: usize, L: FnMut(Event)>(&self, pool: &mut MemoryPool<N>, now: u32, log: &mut L) -> Option<usize> {
        if pool.count(SlotState::R) as u32 > self.r_protected {
            let victim = match self.retention {
                Retention::Fifo => pool.oldest(SlotState::R),
                Retention::Reservoir => pool.indices(SlotState::R).min_by_key(|&i| reservoir_key(pool.slot(i).id)),
            };
            if let Some(idx) = victim {
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
