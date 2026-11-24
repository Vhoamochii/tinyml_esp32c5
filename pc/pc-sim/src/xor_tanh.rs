use libm::{sqrtf, tanhf};

pub const INPUT_DIM: usize = 2;
pub const HIDDEN_DIM: usize = 8;
pub const TRAIN_EPOCHS: u32 = 2000;
pub const LEARNING_RATE: f32 = 0.2;

pub const XOR_DATA: [([f32; 2], u8); 4] = [
    ([0.0, 0.0], 0),
    ([0.0, 1.0], 1),
    ([1.0, 0.0], 1),
    ([1.0, 1.0], 0),
];

struct XorShift32 { state: u32 }

impl XorShift32 {
    fn new(seed: u32) -> Self { Self { state: if seed == 0 { 0xDEAD_BEEF } else { seed } } }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.state = x; x
    }
    fn next_signed(&mut self) -> f32 {
        let u = self.next_u32() as f32 / u32::MAX as f32;
        2.0 * u - 1.0
    }
}

pub struct XorTanhNet {
    pub w_hidden: [[f32; INPUT_DIM]; HIDDEN_DIM],
    pub b_hidden: [f32; HIDDEN_DIM],
    pub w_output: [f32; HIDDEN_DIM],
    pub b_output: f32,
}

impl XorTanhNet {
    pub fn new(seed: u32) -> Self {
        let mut rng = XorShift32::new(seed);
        let lim_h = sqrtf(6.0 / (INPUT_DIM + HIDDEN_DIM) as f32);
        let lim_o = sqrtf(6.0 / (HIDDEN_DIM + 1) as f32);

        let mut w_hidden = [[0.0f32; INPUT_DIM]; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            for j in 0..INPUT_DIM { w_hidden[i][j] = rng.next_signed() * lim_h; }
        }
        let mut b_hidden = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM { b_hidden[i] = rng.next_signed() * lim_h; }
        let mut w_output = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM { w_output[i] = rng.next_signed() * lim_o; }
        let b_output = rng.next_signed() * lim_o;

        Self { w_hidden, b_hidden, w_output, b_output }
    }

    pub fn zero() -> Self {
        Self {
            w_hidden: [[0.0; INPUT_DIM]; HIDDEN_DIM],
            b_hidden: [0.0; HIDDEN_DIM],
            w_output: [0.0; HIDDEN_DIM],
            b_output: 0.0,
        }
    }

    pub fn forward(&self, x: [f32; 2]) -> (f32, [f32; HIDDEN_DIM]) {
        let x0 = 2.0 * x[0] - 1.0;
        let x1 = 2.0 * x[1] - 1.0;

        let mut h = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let z = self.w_hidden[i][0] * x0 + self.w_hidden[i][1] * x1 + self.b_hidden[i];
            h[i] = tanhf(z);
        }
        let mut z_out = self.b_output;
        for i in 0..HIDDEN_DIM { z_out += self.w_output[i] * h[i]; }
        let y = tanhf(z_out);
        (y, h)
    }

    pub fn predict(&self, x: [f32; 2]) -> u8 {
        let (y, _) = self.forward(x);
        if y > 0.0 { 1 } else { 0 }
    }

    pub fn train_step(&mut self, x: [f32; 2], target_class: u8, lr: f32) -> f32 {
        let x0 = 2.0 * x[0] - 1.0;
        let x1 = 2.0 * x[1] - 1.0;
        let target = if target_class == 0 { -1.0 } else { 1.0 };
        let (y, h) = self.forward(x);
        let err = y - target;
        let loss = err * err;
        let d_out = 2.0 * err * (1.0 - y * y);
        let w_output_old = self.w_output;

        for i in 0..HIDDEN_DIM { self.w_output[i] -= lr * d_out * h[i]; }
        self.b_output -= lr * d_out;

        for i in 0..HIDDEN_DIM {
            let d_h = d_out * w_output_old[i] * (1.0 - h[i] * h[i]);
            self.w_hidden[i][0] -= lr * d_h * x0;
            self.w_hidden[i][1] -= lr * d_h * x1;
            self.b_hidden[i] -= lr * d_h;
        }
        loss
    }
}

pub struct XorTestResult {
    pub epochs: u32,
    pub final_loss: f32,
    pub correct: u8,
    pub total: u8,
    pub outputs: [f32; 4],
}

impl XorTestResult {
    pub fn passed(&self) -> bool { self.correct == self.total }
}

pub fn run_xor_tanh_test<L: FnMut(u32, f32, u8, u8)>(mut log_epoch: L) -> XorTestResult {
    let mut net = XorTanhNet::new(0xC0FF_EE00);
    let mut final_loss = 0.0;

    for epoch in 0..TRAIN_EPOCHS {
        let mut total_loss = 0.0;
        let mut correct = 0u8;
        for (x, target) in XOR_DATA.iter() {
            let loss = net.train_step(*x, *target, LEARNING_RATE);
            total_loss += loss;
            if net.predict(*x) == *target { correct += 1; }
        }
        final_loss = total_loss / 4.0;
        log_epoch(epoch, final_loss, correct, 4);
    }

    let mut outputs = [0.0f32; 4];
    let mut correct = 0u8;
    for (i, (x, target)) in XOR_DATA.iter().enumerate() {
        let (y, _) = net.forward(*x);
        outputs[i] = y;
        if net.predict(*x) == *target { correct += 1; }
    }

    XorTestResult { epochs: TRAIN_EPOCHS, final_loss, correct, total: 4, outputs }
}

pub fn run_xor_tanh_test_silent() -> XorTestResult {
    run_xor_tanh_test(|_, _, _, _| {})
}
