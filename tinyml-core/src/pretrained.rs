// FILE SINH TỰ ĐỘNG bởi `cargo run --release --bin pretrain` trong pc/pc-sim. Không sửa tay.
// Hệ số chuẩn hóa 16 đặc trưng và trọng số lớp cuối huấn luyện sẵn trên 8 ổ huấn luyện ở N15_M07_F10.
// Bản này là chỗ giữ chỗ: chưa chạy pretrain nên AVAILABLE = false.

use crate::head::Head;
use crate::{NUM_CLASSES, NUM_FEATURES};

pub const AVAILABLE: bool = false;
pub const FEATURE_MEAN: [f32; NUM_FEATURES] = [0.0; NUM_FEATURES];
pub const FEATURE_STD: [f32; NUM_FEATURES] = [1.0; NUM_FEATURES];
pub const WEIGHTS: [[f32; NUM_FEATURES]; NUM_CLASSES] = [[0.0; NUM_FEATURES]; NUM_CLASSES];
pub const BIAS: [f32; NUM_CLASSES] = [0.0; NUM_CLASSES];

/// Lớp cuối với trọng số huấn luyện sẵn (trọng số ban đầu chung cho mọi lần chạy)
pub fn head() -> Head<NUM_CLASSES> {
    Head { weights: WEIGHTS, bias: BIAS }
}
