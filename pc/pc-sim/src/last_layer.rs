//! Lop cuoi (Last Layer): Tuyen tinh Wz + b, Softmax, Cross-Entropy, SGD.
//! Ho tro ca 3 lop (o lan) va 2 lop (phep thu XOR).
//! Ho tro giai luong tu INT8 -> FP32 truoc buoc SGD (de cuong muc 3.1).

use libm::{expf, logf};

/// Lop cuoi tuyen tinh tong quat.
/// C: so lop (3 cho o lan, 2 cho XOR).
/// F: so dac trung (16 theo de cuong).
pub struct LastLayer<const C: usize, const F: usize = 16> {
    pub w: [[f32; F]; C],
    pub b: [f32; C],
}

impl<const C: usize, const F: usize> LastLayer<C, F> {
    /// Khoi tao trong so va he so chenh bang 0.
    pub fn new() -> Self {
        Self {
            w: [[0.0; F]; C],
            b: [0.0; C],
        }
    }

    /// Tinh diem (logits) cho tat ca cac lop: s = Wz + b.
    pub fn scores(&self, z: &[f32; F]) -> [f32; C] {
        let mut s = self.b;
        let mut c = 0;
        while c < C {
            let mut i = 0;
            while i < F {
                s[c] += self.w[c][i] * z[i];
                i += 1;
            }
            c += 1;
        }
        s
    }

    /// Softmax on dinh: tru max de tranh tran exp.
    pub fn softmax(scores: [f32; C]) -> [f32; C] {
        let mut max_val = scores[0];
        let mut c = 1;
        while c < C {
            if scores[c] > max_val {
                max_val = scores[c];
            }
            c += 1;
        }
        let mut exps = [0.0; C];
        let mut sum = 0.0;
        c = 0;
        while c < C {
            exps[c] = expf(scores[c] - max_val);
            sum += exps[c];
            c += 1;
        }
        c = 0;
        while c < C {
            exps[c] /= sum;
            c += 1;
        }
        exps
    }

    /// Du doan lop co diem cao nhat.
    pub fn predict(&self, z: &[f32; F]) -> u8 {
        let s = self.scores(z);
        let mut max_idx = 0;
        let mut c = 1;
        while c < C {
            if s[c] > s[max_idx] {
                max_idx = c;
            }
            c += 1;
        }
        max_idx as u8
    }

    /// Mot buoc SGD voi dac trung FP32. Tra ve loss Cross-Entropy.
    pub fn train_step(&mut self, z: &[f32; F], label: u8, lr: f32) -> f32 {
        if (label as usize) >= C {
            return f32::NAN; // Edge case: nhan ngoai pham vi
        }
        let s = self.scores(z);
        let p = Self::softmax(s);
        let loss = -logf(p[label as usize]);

        let mut c = 0;
        while c < C {
            let grad = p[c] - if c == label as usize { 1.0 } else { 0.0 };
            self.b[c] -= lr * grad;
            let mut i = 0;
            while i < F {
                self.w[c][i] -= lr * grad * z[i];
                i += 1;
            }
            c += 1;
        }
        loss
    }

    /// Giai luong tu INT8 sang FP32 va cap nhat.
    /// z_fp = (z_i8 - zero) * scale. (De cuong muc 3.1)
    pub fn train_step_int8(
        &mut self,
        z_i8: &[i8; F],
        scale: f32,
        zero: i8,
        label: u8,
        lr: f32,
    ) -> f32 {
        let mut z_fp = [0.0; F];
        let mut i = 0;
        while i < F {
            z_fp[i] = (z_i8[i] as f32 - zero as f32) * scale;
            i += 1;
        }
        self.train_step(&z_fp, label, lr)
    }
}
