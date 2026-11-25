// Module kiểm thử mạng nơ-ron cơ bản với bài toán XOR
// Đảm bảo không dùng std, chỉ dùng core

pub struct XorNet {
    pub w1: [f32; 13], // Trọng số lớp ẩn (đệm 13 số 0 theo đề cương)
    pub b1: f32,
    pub w0: [f32; 13], // Trọng số lớp ra
    pub b0: f32,
    pub lr: f32,
}

impl XorNet {
    pub fn new() -> Self {
        Self {
            w1: [0.0; 13],
            b1: 0.0,
            w0: [0.0; 13],
            b0: 0.0,
            lr: 0.1,
        }
    }

    // Hàm kích hoạt tanh
    fn tanh(x: f32) -> f32 {
        // Xấp xỉ tanh đơn giản cho no_std (hoặc dùng libm nếu có)
        if x > 10.0 { return 1.0; }
        if x < -10.0 { return -1.0; }
        let e2x = libm::expf(2.0 * x);
        (e2x - 1.0) / (e2x + 1.0)
    }

    // B1: Dự đoán Forward Pass
    pub fn predict(&self, x1: f32, x2: f32) -> f32 {
        // Điểm lớp 1: z = x1*w1[0] + x2*w1[1] + b1
        let z1 = x1 * self.w1[0] + x2 * self.w1[1] + self.b1;
        let h = Self::tanh(z1);
        
        // Điểm lớp 0: z = h*w0[0] + b0
        let z0 = h * self.w0[0] + self.b0;
        Self::tanh(z0)
    }

    // B2: Một bước cập nhật SGD
    pub fn train_step(&mut self, x1: f32, x2: f32, y_true: f32) {
        // Forward
        let z1 = x1 * self.w1[0] + x2 * self.w1[1] + self.b1;
        let h = Self::tanh(z1);
        let z0 = h * self.w0[0] + self.b0;
        let y_pred = Self::tanh(z0);

        // Backward (đạo hàm hàm loss MSE: 0.5*(y_pred - y_true)^2)
        let d_loss = y_pred - y_true;
        let d_z0 = d_loss * (1.0 - y_pred * y_pred); // Đạo hàm tanh
        let d_h = d_z0 * self.w0[0];
        let d_z1 = d_h * (1.0 - h * h); // Đạo hàm tanh

        // Cập nhật trọng số
        self.w0[0] -= self.lr * d_z0 * h;
        self.b0 -= self.lr * d_z0;
        self.w1[0] -= self.lr * d_z1 * x1;
        self.w1[1] -= self.lr * d_z1 * x2;
        self.b1 -= self.lr * d_z1;
    }
}
