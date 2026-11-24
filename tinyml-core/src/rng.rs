// Bộ sinh số ngẫu nhiên xorshift32: nhỏ, không cần thư viện, cho cùng một dãy số trên PC và trên bo.

#[derive(Clone, Debug)]
pub struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    pub const fn new(seed: u32) -> Self {
        // Trạng thái 0 làm xorshift kẹt mãi ở 0
        Self { state: if seed == 0 { 0x9E37_79B9 } else { seed } }
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Số nguyên trong [0, n), với n > 0
    pub fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }

    /// Số thực trong [lo, hi)
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        // Lấy 24 bit cao để được số thực đều trong [0, 1)
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        lo + (hi - lo) * unit
    }
}
