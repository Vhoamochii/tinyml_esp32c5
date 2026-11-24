// Mảng ô nhớ dùng chung P/Q/R (đề cương, Bảng 2 "Vòng đời của một mẫu" và mục "Giới hạn bộ nhớ").
// Mỗi mẫu chiếm đúng MỘT ô; trạng thái ô đổi tại chỗ: Trống -> P -> Q -> R -> Trống.
// Module này chỉ cung cấp thao tác trên ô. Quyết định thu hồi ô nào nằm ở policy.rs,
// quyết định học mẫu nào nằm ở schedule.rs.

use crate::NUM_FEATURES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    Free, // Ô trống
    P,    // Chờ nhãn
    Q,    // Đã khớp nhãn, chờ học lần đầu
    R,    // Đã học lần đầu, có thể được học lại
}

#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub features: [f32; NUM_FEATURES],
    pub id: u32,    // ID cửa sổ, dùng để ghép nhãn
    pub order: u32, // Thứ tự ô vào trạng thái hiện tại (số tăng dần trong cả mảng)
    pub label: u8,  // Chỉ có nghĩa khi ô ở Q hoặc R
    pub state: SlotState,
}

impl Slot {
    const EMPTY: Slot = Slot { features: [0.0; NUM_FEATURES], id: 0, order: 0, label: 0, state: SlotState::Free };
}

/// Kích thước thực s của một ô sau biên dịch, kể cả ID, nhãn, thứ tự, trạng thái và phần đệm
pub const SLOT_SIZE_BYTES: usize = core::mem::size_of::<Slot>();

/// Số ô K = floor((M - H) / s), với M byte cho bộ đệm và H byte quản lý ngoài mảng
pub const fn slots_for_budget(budget_bytes: usize, overhead_bytes: usize) -> usize {
    budget_bytes.saturating_sub(overhead_bytes) / SLOT_SIZE_BYTES
}

/// Mảng N ô, cấp phát tĩnh
pub struct MemoryPool<const N: usize> {
    slots: [Slot; N],
    next_order: u32, // Bộ đếm thứ tự; đủ cho 4 tỷ lần đổi trạng thái
}

impl<const N: usize> Default for MemoryPool<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> MemoryPool<N> {
    pub const fn new() -> Self {
        Self { slots: [Slot::EMPTY; N], next_order: 0 }
    }

    pub fn slot(&self, idx: usize) -> &Slot {
        &self.slots[idx]
    }

    pub fn count(&self, state: SlotState) -> usize {
        self.slots.iter().filter(|s| s.state == state).count()
    }

    /// Byte đang dùng = số ô không trống * s
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

    pub fn find_free(&self) -> Option<usize> {
        self.indices(SlotState::Free).next()
    }

    /// Ô (không trống) đang giữ mẫu có ID này
    pub fn find_id(&self, id: u32) -> Option<usize> {
        self.slots.iter().position(|s| s.state != SlotState::Free && s.id == id)
    }

    /// Ô vào trạng thái `state` sớm nhất:
    /// P -> mẫu chờ lâu nhất, Q -> mẫu đầu hàng, R -> mẫu vào R sớm nhất
    pub fn oldest(&self, state: SlotState) -> Option<usize> {
        self.indices(state).min_by_key(|&i| self.slots[i].order)
    }

    /// Ô trống nhận mẫu mới (Trống -> P)
    pub fn put_p(&mut self, idx: usize, id: u32, features: [f32; NUM_FEATURES]) {
        debug_assert_eq!(self.slots[idx].state, SlotState::Free);
        let order = self.next_order();
        self.slots[idx] = Slot { features, id, order, label: 0, state: SlotState::P };
    }

    /// Nhãn khớp ID (P -> Q)
    pub fn to_q(&mut self, idx: usize, label: u8) {
        debug_assert_eq!(self.slots[idx].state, SlotState::P);
        let order = self.next_order();
        let slot = &mut self.slots[idx];
        slot.label = label;
        slot.state = SlotState::Q;
        slot.order = order;
    }

    /// Sau lần học đầu tiên (Q -> R)
    pub fn to_r(&mut self, idx: usize) {
        debug_assert_eq!(self.slots[idx].state, SlotState::Q);
        let order = self.next_order();
        self.slots[idx].state = SlotState::R;
        self.slots[idx].order = order;
    }

    /// Trả ô về trạng thái trống
    pub fn free(&mut self, idx: usize) {
        self.slots[idx].state = SlotState::Free;
    }

    fn next_order(&mut self) -> u32 {
        let order = self.next_order;
        self.next_order = order.wrapping_add(1);
        order
    }
}
