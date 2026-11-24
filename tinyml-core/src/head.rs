// Lớp cuối FP32 (đề cương, mục "Trích đặc trưng và cập nhật lớp cuối"):
// tính Wz + b cho C lớp, dùng softmax và cập nhật W, b bằng một bước SGD với hàm mất mát entropy chéo.
// Đây là phần DUY NHẤT được học trên thiết bị; bộ trích xuất INT8 giữ nguyên trọng số.
// C là hằng số lúc biên dịch: 3 lớp cho ổ lăn, 2 lớp cho phép thử XOR.

use crate::NUM_FEATURES;

#[derive(Clone, Debug)]
pub struct Head<const C: usize> {
    pub weights: [[f32; NUM_FEATURES]; C],
    pub bias: [f32; C],
}

impl<const C: usize> Default for Head<C> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<const C: usize> Head<C> {
    /// Số byte tham số (trọng số + hệ số chệch)
    pub const PARAM_BYTES: usize = (C * NUM_FEATURES + C) * core::mem::size_of::<f32>();

    /// W = 0, b = 0 (trọng số ban đầu dùng trong phép thử XOR).
    /// Khi đã có trọng số huấn luyện ngoại tuyến thì gán thẳng vào `weights` và `bias`.
    pub const fn zeros() -> Self {
        Self { weights: [[0.0; NUM_FEATURES]; C], bias: [0.0; C] }
    }

    /// Điểm của từng lớp: Wz + b
    pub fn logits(&self, z: &[f32; NUM_FEATURES]) -> [f32; C] {
        let mut s = self.bias;
        for (sc, wc) in s.iter_mut().zip(self.weights.iter()) {
            for (w, zi) in wc.iter().zip(z.iter()) {
                *sc += w * zi;
            }
        }
        s
    }

    pub fn probs(&self, z: &[f32; NUM_FEATURES]) -> [f32; C] {
        softmax(self.logits(z))
    }

    /// Lớp có điểm cao nhất (nếu bằng nhau thì lấy lớp có chỉ số nhỏ hơn)
    pub fn predict(&self, z: &[f32; NUM_FEATURES]) -> usize {
        let s = self.logits(z);
        let mut best = 0;
        for c in 1..C {
            if s[c] > s[best] {
                best = c;
            }
        }
        best
    }

    /// Một bước SGD trên MỘT mẫu. Trả về loss entropy chéo, tính trước khi cập nhật.
    pub fn sgd_step(&mut self, z: &[f32; NUM_FEATURES], label: u8, lr: f32) -> f32 {
        let label = label as usize;
        debug_assert!(label < C);
        let p = self.probs(z);
        // Đạo hàm của entropy chéo theo điểm lớp c: p_c - [c == nhãn]
        for c in 0..C {
            let g = p[c] - if c == label { 1.0 } else { 0.0 };
            for (w, zi) in self.weights[c].iter_mut().zip(z.iter()) {
                *w -= lr * g * zi;
            }
            self.bias[c] -= lr * g;
        }
        -libm::logf(p[label].max(1e-12))
    }
}

fn softmax<const C: usize>(s: [f32; C]) -> [f32; C] {
    // Trừ giá trị lớn nhất trước khi lấy mũ để tránh tràn số
    let max = s.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut p = [0.0f32; C];
    let mut sum = 0.0f32;
    for (pc, sc) in p.iter_mut().zip(s.iter()) {
        *pc = libm::expf(sc - max);
        sum += *pc;
    }
    for pc in p.iter_mut() {
        *pc /= sum;
    }
    p
}
