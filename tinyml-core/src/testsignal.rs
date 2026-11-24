// Tín hiệu giả dùng chung cho Ubuntu và ESP32-C5: bo dùng để đo thời gian bộ trích xuất,
// Ubuntu dùng để in cùng vectơ đặc trưng. Hai bên phải in ra 16 số giống hệt nhau
// (đề cương: đối chiếu đầu ra bộ trích xuất giữa PC và bo trước các thí nghiệm chính).

use crate::features::{WindowExtractor, BLOCKS_PER_WINDOW, BLOCK_LEN};
use crate::rng::XorShift32;
use crate::NUM_FEATURES;

/// Số cửa sổ tín hiệu giả dùng khi đo và đối chiếu
pub const CHECK_WINDOWS: usize = 4;

/// Nhiễu trắng + tiếng gõ ~76 Hz tắt dần. Thời gian tính gần như không phụ thuộc giá trị mẫu,
/// chỉ cần tín hiệu có biên độ thực tế.
pub struct FakeSignal {
    rng: XorShift32,
    n: u32,
    amp: f32,
}

impl Default for FakeSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeSignal {
    pub fn new() -> Self {
        Self { rng: XorShift32::new(1), n: 0, amp: 0.0 }
    }

    /// Sinh khối 1024 mẫu tiếp theo
    pub fn fill(&mut self, block: &mut [f32; BLOCK_LEN]) {
        for v in block.iter_mut() {
            if self.n % 838 == 0 {
                self.amp = 2.0; // 64000 / 838 ≈ 76,4 cú gõ mỗi giây
            }
            self.amp *= 0.97;
            let ring = if self.n % 2 == 0 { self.amp } else { -self.amp };
            *v = 0.1 * self.rng.range_f32(-1.0, 1.0) + ring;
            self.n += 1;
        }
    }
}

/// 16 đặc trưng của cửa sổ cuối khi nạp CHECK_WINDOWS cửa sổ tín hiệu giả qua một bộ trích xuất
pub fn check_features() -> [f32; NUM_FEATURES] {
    let mut signal = FakeSignal::new();
    let mut extractor = WindowExtractor::new();
    let mut block = [0.0f32; BLOCK_LEN];
    let mut z = [0.0f32; NUM_FEATURES];
    for _ in 0..CHECK_WINDOWS * BLOCKS_PER_WINDOW {
        signal.fill(&mut block);
        if let Some(features) = extractor.push_block(&block) {
            z = features;
        }
    }
    z
}
