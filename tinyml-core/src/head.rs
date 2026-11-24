// Lớp cuối FP32 (16 -> 3). Đây là lớp DUY NHẤT được cập nhật trên thiết bị;
// bộ trích xuất đặc trưng INT8 giữ nguyên trọng số (Chương 2).
// Softmax + cross-entropy, một bước SGD trên lô gồm mẫu mới và các mẫu replay (nhân K3 ở Chương 3).

use crate::{NUM_CLASSES, NUM_FEATURES};

#[derive(Clone, Debug)]
pub struct Head {
    pub weights: [[f32; NUM_FEATURES]; NUM_CLASSES],
    pub bias: [f32; NUM_CLASSES],
}

impl Default for Head {
    fn default() -> Self {
        Self::zeros()
    }
}

impl Head {
    /// Số byte tham số (trọng số + bias)
    pub const PARAM_BYTES: usize = (NUM_CLASSES * NUM_FEATURES + NUM_CLASSES) * core::mem::size_of::<f32>();

    /// Khởi tạo bằng 0. Lớp tuyến tính + softmax là bài toán lồi nên không cần phá đối xứng.
    /// Khi đã có trọng số huấn luyện ngoại tuyến thì gán thẳng vào `weights` và `bias`.
    pub const fn zeros() -> Self {
        Self { weights: [[0.0; NUM_FEATURES]; NUM_CLASSES], bias: [0.0; NUM_CLASSES] }
    }

    pub fn logits(&self, x: &[f32; NUM_FEATURES]) -> [f32; NUM_CLASSES] {
        let mut z = self.bias;
        for (zc, wc) in z.iter_mut().zip(self.weights.iter()) {
            for (w, xi) in wc.iter().zip(x.iter()) {
                *zc += w * xi;
            }
        }
        z
    }

    pub fn probs(&self, x: &[f32; NUM_FEATURES]) -> [f32; NUM_CLASSES] {
        softmax(self.logits(x))
    }

    /// Lớp dự đoán = chỉ số logit lớn nhất
    pub fn predict(&self, x: &[f32; NUM_FEATURES]) -> usize {
        let z = self.logits(x);
        let mut best = 0;
        for c in 1..NUM_CLASSES {
            if z[c] > z[best] {
                best = c;
            }
        }
        best
    }

    /// Một bước SGD trên cả lô (gradient lấy trung bình).
    /// Trả về loss cross-entropy trung bình, tính trước khi cập nhật.
    pub fn sgd_step(&mut self, batch: &[(&[f32; NUM_FEATURES], u8)], lr: f32) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }
        let mut grad_w = [[0.0f32; NUM_FEATURES]; NUM_CLASSES];
        let mut grad_b = [0.0f32; NUM_CLASSES];
        let mut loss = 0.0f32;

        for &(x, label) in batch {
            let label = label as usize;
            debug_assert!(label < NUM_CLASSES);
            let p = self.probs(x);
            loss -= libm::logf(p[label].max(1e-12));
            // Đạo hàm của cross-entropy theo logit: p - one_hot(label)
            for c in 0..NUM_CLASSES {
                let g = p[c] - if c == label { 1.0 } else { 0.0 };
                for (gw, xi) in grad_w[c].iter_mut().zip(x.iter()) {
                    *gw += g * xi;
                }
                grad_b[c] += g;
            }
        }

        let scale = lr / batch.len() as f32;
        for c in 0..NUM_CLASSES {
            for (w, gw) in self.weights[c].iter_mut().zip(grad_w[c].iter()) {
                *w -= scale * gw;
            }
            self.bias[c] -= scale * grad_b[c];
        }
        loss / batch.len() as f32
    }
}

fn softmax(z: [f32; NUM_CLASSES]) -> [f32; NUM_CLASSES] {
    // Trừ giá trị lớn nhất trước khi lấy mũ để tránh tràn số
    let max = z.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut p = [0.0f32; NUM_CLASSES];
    let mut sum = 0.0f32;
    for (pc, zc) in p.iter_mut().zip(z.iter()) {
        *pc = libm::expf(zc - max);
        sum += *pc;
    }
    for pc in p.iter_mut() {
        *pc /= sum;
    }
    p
}
