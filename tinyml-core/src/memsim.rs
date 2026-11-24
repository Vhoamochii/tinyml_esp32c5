// Vùng nhớ dùng chung P/Q/R (Chương 2, mục "Mô hình vùng nhớ dùng chung").
// Mỗi mẫu chiếm đúng MỘT ô và chỉ lưu một bản đặc trưng; trạng thái ô đổi tại chỗ:
//   Trống -> P (chờ nhãn) -> Q (chờ học) -> R (học lại) -> Trống
// Module này chỉ cung cấp thao tác trên ô. Quyết định giữ hay thu hồi ô nào nằm ở policy.rs,
// để đối chứng và chính sách đề xuất chỉ khác nhau ở module quyết định phân bổ (YC3).

use crate::NUM_FEATURES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    Free, // Ô trống, thuộc vùng chung
    P,    // Chờ nhãn
    Q,    // Đã có nhãn, chờ học lần đầu
    R,    // Đã học, giữ lại để replay
}

#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub features: [f32; NUM_FEATURES],
    pub id: u32,        // Mã mẫu (mã cửa sổ), dùng để ghép nhãn
    pub t_request: u32, // Bước mẫu được tạo, tức thời điểm yêu cầu nhãn
    pub t_state: u32,   // Bước ô vào trạng thái hiện tại (dùng đo thời gian chờ trong Q)
    pub label: u8,      // Chỉ có nghĩa khi ô ở Q hoặc R
    pub state: SlotState,
}

impl Slot {
    const EMPTY: Slot = Slot {
        features: [0.0; NUM_FEATURES],
        id: 0,
        t_request: 0,
        t_state: 0,
        label: 0,
        state: SlotState::Free,
    };
}

/// Kích thước thực của một ô sau biên dịch (gồm cả padding), không dùng số ước lượng
pub const SLOT_SIZE_BYTES: usize = core::mem::size_of::<Slot>();

/// Số ô tối đa với ngân sách `budget_bytes`: N = floor(B / s)
pub const fn slots_for_budget(budget_bytes: usize) -> usize {
    budget_bytes / SLOT_SIZE_BYTES
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemError {
    Full,       // Không còn ô trống để cấp cho P
    IdNotFound, // Nhãn đến nhưng không có ô P nào mang mã này (ghép trượt)
}

impl MemError {
    pub fn as_str(self) -> &'static str {
        match self {
            MemError::Full => "[MEMSIM][ERROR] Khong con o trong de cap cho P",
            MemError::IdNotFound => "[MEMSIM][ERROR] Khong tim thay ID trong P",
        }
    }
}

/// Vùng nhớ N ô, cấp phát tĩnh
pub struct MemoryPool<const N: usize> {
    slots: [Slot; N],
}

impl<const N: usize> Default for MemoryPool<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> MemoryPool<N> {
    pub const fn new() -> Self {
        Self { slots: [Slot::EMPTY; N] }
    }

    pub fn slot(&self, idx: usize) -> &Slot {
        &self.slots[idx]
    }

    pub fn count(&self, state: SlotState) -> usize {
        self.slots.iter().filter(|s| s.state == state).count()
    }

    /// Byte đang dùng = số ô không trống * kích thước ô
    pub fn usage_bytes(&self) -> usize {
        (N - self.count(SlotState::Free)) * SLOT_SIZE_BYTES
    }

    pub const fn capacity_bytes(&self) -> usize {
        N * SLOT_SIZE_BYTES
    }

    /// Chỉ số các ô đang ở trạng thái `state`, theo thứ tự ô
    pub fn indices(&self, state: SlotState) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(move |(_, s)| s.state == state)
            .map(|(i, _)| i)
    }

    /// Cấp một ô trống cho mẫu mới (Trống -> P)
    pub fn alloc_p(&mut self, id: u32, features: [f32; NUM_FEATURES], now: u32) -> Result<usize, MemError> {
        let idx = self.indices(SlotState::Free).next().ok_or(MemError::Full)?;
        self.slots[idx] = Slot {
            features,
            id,
            t_request: now,
            t_state: now,
            label: 0,
            state: SlotState::P,
        };
        Ok(idx)
    }

    /// Nhãn đến: tìm ô P theo mã mẫu và chuyển sang Q (P -> Q)
    pub fn assign_label(&mut self, id: u32, label: u8, now: u32) -> Result<usize, MemError> {
        let idx = self
            .slots
            .iter()
            .position(|s| s.state == SlotState::P && s.id == id)
            .ok_or(MemError::IdNotFound)?;
        let slot = &mut self.slots[idx];
        slot.label = label;
        slot.state = SlotState::Q;
        slot.t_state = now;
        Ok(idx)
    }

    /// Ô Q chờ lâu nhất (FIFO theo thời điểm vào Q)
    pub fn oldest_q(&self) -> Option<usize> {
        self.indices(SlotState::Q).min_by_key(|&i| self.slots[i].t_state)
    }

    /// Sau lần học đầu: giữ ô lại làm mẫu replay (Q -> R)
    pub fn move_to_r(&mut self, idx: usize, now: u32) {
        debug_assert_eq!(self.slots[idx].state, SlotState::Q);
        self.slots[idx].state = SlotState::R;
        self.slots[idx].t_state = now;
    }

    /// Trả ô về vùng chung: thu hồi P, không giữ sau khi học, hoặc bị thay thế trong R
    pub fn free(&mut self, idx: usize) {
        self.slots[idx].state = SlotState::Free;
    }
}
