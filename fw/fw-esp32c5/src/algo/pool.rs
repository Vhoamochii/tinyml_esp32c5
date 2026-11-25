// Module quản lý bộ nhớ Pool (P-Q-R)
use core::mem::size_of;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PoolItem {
    pub id: u32,
    pub label: u8,
    pub timestamp: u32,
    pub state: [i8; 16], // INT8 16B theo đề cương
    pub padding: [u8; 3], // Đệm để căn lề bộ nhớ
}

impl PoolItem {
    pub const SIZE: usize = size_of::<PoolItem>();
}

pub struct Pool {
    pub k: usize, // Số ô tối đa
}

impl Pool {
    // Tính K = (M - H) / s
    pub fn new(total_ram_mb: usize, reserved_heap: usize) -> Self {
        let total_bytes = total_ram_mb * 1024 * 1024;
        let k = (total_bytes - reserved_heap) / PoolItem::SIZE;
        Self { k }
    }
}
