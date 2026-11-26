//! Lop cuoi tuyen tinh C lop, softmax + cross-entropy, SGD mot buoc FP32.
//! Dac trung luu INT8; giai luong tu truoc buoc SGD (de cuong muc 3).
//! Softmax tru max de khong tran exp khi diem lon.
use libm::{expf, logf};

pub const N_FEATURES: usize = 16;

/// Thong so luong tu hoa: z_fp = (z_i8 - zero) * scale.
#[derive(Clone, Copy)]
pub struct Quant {
    pub scale: [f32; N_FEATURES],
    pub zero: [i8; N_FEATURES],
}

impl Quant {
    /// Gia dinh luong tu hoa doi xung (zero = 0, scale = 1).
    /// Gia tri that lay tu file hieu chuan cua anh (xem hoi 4).
    pub const fn identity() -> Self {
        Self { scale: [1.0f32; N_FEATURES], zero: [0i8; N_FEATURES] }
    }
}

pub struct LinearHead<const C: usize> {
    pub w: [[f32; N_FEATURES]; C],
    pub b: [f32; C],
}

impl<const C: usize> LinearHead<C> {
    pub const fn zeros() -> Self {
        Self { w: [[0.0f32; N_FEATURES]; C], b: [0.0f32; C] }
    }

    pub fn dequant(z_i8: [i8; N_FEATURES], q: &Quant) -> [f32; N_FEATURES] {
        let mut z = [0.0f32; N_FEATURES];
        let mut i = 0;
        while i < N_FEATURES {
            z[i] = (z_i8[i] as f32 - q.zero[i] as f32) * q.scale[i];
            i += 1;
        }
        z
    }

    pub fn scores(&self, z: &[f32; N_FEATURES]) -> [f32; C] {
        let mut s = self.b;
        let mut c = 0;
        while c < C {
            let mut i = 0;
            while i < N_FEATURES {
                s[c] += self.w[c][i] * z[i];
                i += 1;
            }
            c += 1;
        }
        s
    }

    /// Softmax on dinh: tru max. Edge case hai diem bang nhau -> deu 1/C.
    pub fn softmax(scores: [f32; C]) -> [f32; C] {
        let mut m = scores[0];
        let mut c = 1;
        while c < C {
            if scores[c] > m { m = scores[c]; }
            c += 1;
        }
        let mut e = [0.0f32; C];
        let mut sum = 0.0f32;
        c = 0;
        while c < C {
            e[c] = expf(scores[c] - m);
            sum += e[c];
            c += 1;
        }
        c = 0;
        while c < C {
            e[c] /= sum;
            c += 1;
        }
        e
    }

    pub fn predict(&self, z_i8: [i8; N_FEATURES], q: &Quant) -> u8 {
        let z = Self::dequant(z_i8, q);
        let s = self.scores(&z);
        let mut best = 0usize;
        let mut c = 1;
        while c < C {
            if s[c] > s[best] { best = c; }
            c += 1;
        }
        best as u8
    }

    /// Mot buoc SGD, gradient theo logit = p - onehot(label).
    /// Error case: label >= C -> khong cap nhat, tra NaN lo ra o tang tren.
    pub fn sgd_step(&mut self, z_i8: [i8; N_FEATURES], q: &Quant, label: u8, lr: f32) -> f32 {
        if (label as usize) >= C {
            return f32::NAN;
        }
        let z = Self::dequant(z_i8, q);
        let p = Self::softmax(self.scores(&z));
        let loss = -logf(p[label as usize]);
        let mut c = 0;
        while c < C {
            let g = p[c] - if c == label as usize { 1.0 } else { 0.0 };
            self.b[c] -= lr * g;
            let mut i = 0;
            while i < N_FEATURES {
                self.w[c][i] -= lr * g * z[i];
                i += 1;
            }
            c += 1;
        }
        loss
    }

    /// Sao chep trang thai phuc vu do quen (snapshot truoc/sau doi toc do).
    pub fn snapshot(&self) -> Self {
        Self { w: self.w, b: self.b }
    }
}

/// Gia tri vang de cuong muc 4: C=2, W=b=0, mau (1,0) nhan 1, lr=0.1.
/// Ky vong w[1][0]=b[1]=+0.05, w[0][0]=b[0]=-0.05, con lai dung 0, tol 1e-6.
/// Dat lai 0 roi hoc 4 mau lap 200 luot -> phai dung 4/4.
pub fn run_head_golden() -> (bool, f32) {
    let mut head: LinearHead<2> = LinearHead::zeros();
    let q = Quant::identity();
    let mut z_i8 = [0i8; N_FEATURES];
    z_i8[0] = 1; // x1 = 1, x2 = 0, x1*x2 = 0
    head.sgd_step(z_i8, &q, 1, 0.1);
    let mut max_err = 0.0f32;
    let mut check = |got: f32, want: f32| {
        let e = (got - want).abs();
        if e > max_err { max_err = e; }
    };
    check(head.w[1][0], 0.05);
    check(head.b[1], 0.05);
    check(head.w[0][0], -0.05);
    check(head.b[0], -0.05);
    let mut c = 0;
    while c < 2 {
        let mut i = 1;
        while i < N_FEATURES {
            check(head.w[c][i], 0.0);
            i += 1;
        }
        c += 1;
    }

    let mut head2: LinearHead<2> = LinearHead::zeros();
    let data: [([i8; N_FEATURES], u8); 4] = [
        ({ let mut z = [0i8; N_FEATURES]; z }, 0),
        ({ let mut z = [0i8; N_FEATURES]; z[1] = 1; z }, 1),
        ({ let mut z = [0i8; N_FEATURES]; z[0] = 1; z }, 1),
        ({ let mut z = [0i8; N_FEATURES]; z[0] = 1; z[1] = 1; z[2] = 1; z }, 0),
    ];
    let mut epoch = 0;
    while epoch < 200 {
        for (z, y) in data.iter() {
            head2.sgd_step(*z, &q, *y, 0.1);
        }
        epoch += 1;
    }
    let mut correct = 0;
    for (z, y) in data.iter() {
        if head2.predict(*z, &q) == *y { correct += 1; }
    }
    (max_err <= 1e-6 && correct == 4, max_err)
}
