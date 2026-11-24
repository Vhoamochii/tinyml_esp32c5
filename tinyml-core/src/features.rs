// Bộ trích xuất đặc trưng GIAI ĐOẠN 1: 16 đặc trưng tính tay, FP32, không có trọng số.
// Thay tạm cho bộ trích xuất INT8 học được trong đề cương (giai đoạn 2). Giữ nguyên đầu ra 16 số
// nên lớp cuối, mảng ô P/Q/R và lịch học không phải đổi khi thay bộ trích xuất.
// Chạy giống hệt trên Ubuntu và ESP32-C5 (no_std, không heap).
//
// Một cửa sổ = 16 khối x 1024 mẫu = 16384 mẫu = 256 ms ở 64 kHz, đủ dài để thấy ~19 cú gõ của lỗi
// vòng ngoài. Bộ trích xuất nhận từng khối (bo không phải giữ cả cửa sổ), gồm 3 nhóm đặc trưng,
// đều không phụ thuộc độ to của tín hiệu:
//   0-5   hình dạng tín hiệu (có va đập hay không)
//   6-9   năng lượng dải tần tương đối (tỷ lệ trên tổng năng lượng)
//   10-15 phổ bao theo DẢI TẦN CỐ ĐỊNH (nhịp va đập), không cần biết tốc độ quay. Khi đổi tốc độ,
//         nhịp gõ dời sang dải khác: đó là độ lệch mà bo phải tự học lại.

use crate::NUM_FEATURES;

/// Số mẫu của một khối (đơn vị nhận dữ liệu và độ dài FFT)
pub const BLOCK_LEN: usize = 1024;
/// Số khối trong một cửa sổ
pub const BLOCKS_PER_WINDOW: usize = 16;
/// Độ dài cửa sổ: 16384 mẫu = 256 ms ở 64 kHz
pub const WINDOW_LEN: usize = BLOCK_LEN * BLOCKS_PER_WINDOW;
/// Tần số lấy mẫu kênh rung Paderborn
pub const SAMPLE_RATE_HZ: f32 = 64_000.0;

/// Tên 16 đặc trưng theo đúng thứ tự trong vectơ đầu ra
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "crest_factor",
    "kurtosis",
    "skewness",
    "impulse_factor",
    "shape_factor",
    "clearance_factor",
    "log_rel_band_0_2k",
    "log_rel_band_2_8k",
    "log_rel_band_8_16k",
    "log_rel_band_16_32k",
    "env_10_20hz",
    "env_20_35hz",
    "env_35_60hz",
    "env_60_100hz",
    "env_100_170hz",
    "env_170_300hz",
];

/// Độ dài FFT (dùng cho cả khối tín hiệu và đường bao)
const FFT_LEN: usize = 1024;
/// Đường bao: lấy trung bình mỗi 16 mẫu, còn 4000 mẫu/giây
const ENV_DECIMATION: usize = 16;
const ENV_PER_BLOCK: usize = BLOCK_LEN / ENV_DECIMATION;
const ENV_LEN: usize = ENV_PER_BLOCK * BLOCKS_PER_WINDOW;
/// Lọc thông cao trước khi lấy đường bao: chỉ giữ tiếng gõ (cộng hưởng vài kHz)
const HIGHPASS_HZ: f32 = 2000.0;

/// Ranh giới 4 dải tần tương đối theo bin FFT của khối (62,5 Hz/bin): 2 kHz = bin 32, 8 kHz = 128,
/// 16 kHz = 256, 32 kHz = 512 (Nyquist). Bỏ bin 0.
const REL_BAND_EDGES: [usize; 5] = [1, 32, 128, 256, 513];
/// Ranh giới 6 dải phổ bao theo bin FFT đường bao (4000 / 1024 = 3,906 Hz/bin):
/// 10, 20, 35, 60, 100, 170, 300 Hz (làm tròn lên bin kế tiếp)
const ENV_BAND_EDGES: [usize; 7] = [3, 6, 9, 16, 26, 44, 77];

const EPS: f32 = 1e-12;

const _: () = assert!(ENV_LEN == FFT_LEN && BLOCK_LEN == FFT_LEN);

/// Chuẩn hóa tại chỗ: (z - trung bình) / độ lệch chuẩn, với hệ số tính trên tập huấn luyện
pub fn normalize(z: &mut [f32; NUM_FEATURES], mean: &[f32; NUM_FEATURES], std: &[f32; NUM_FEATURES]) {
    for ((v, m), s) in z.iter_mut().zip(mean.iter()).zip(std.iter()) {
        *v = (*v - m) / s;
    }
}

/// Tính 16 đặc trưng của một cửa sổ đầy đủ (tiện cho PC; bo dùng WindowExtractor theo từng khối)
pub fn extract(x: &[f32; WINDOW_LEN]) -> [f32; NUM_FEATURES] {
    let mut extractor = WindowExtractor::new();
    let mut out = [0.0; NUM_FEATURES];
    for block in x.chunks_exact(BLOCK_LEN) {
        if let Some(z) = extractor.push_block(block.try_into().unwrap()) {
            out = z;
        }
    }
    out
}

/// Bộ trích xuất nhận từng khối 1024 mẫu; đủ 16 khối thì trả 16 đặc trưng và tự đặt lại.
/// Mỗi cửa sổ tính độc lập (bộ lọc cũng đặt lại), nên kết quả không phụ thuộc cửa sổ trước.
pub struct WindowExtractor {
    blocks: usize,
    // Thống kê miền thời gian, cộng dồn bằng f64 để tránh sai số khi tính mô-men từ tổng lũy thừa
    sum: f64,
    sum_sq: f64,
    sum_cube: f64,
    sum_quad: f64,
    sum_abs: f64,
    sum_sqrt_abs: f64,
    max: f32,
    min: f32,
    // Năng lượng 4 dải, cộng qua các khối
    rel_bands: [f32; 4],
    // Bộ lọc thông cao bậc 2 (Butterworth): hệ số b0, b1, b2, a1, a2 và trạng thái x1, x2, y1, y2
    hp_coef: [f32; 5],
    hp_state: [f32; 4],
    // Đường bao của cả cửa sổ
    envelope: [f32; ENV_LEN],
}

impl Default for WindowExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowExtractor {
    pub fn new() -> Self {
        // Hệ số lọc thông cao theo RBJ Audio EQ Cookbook, Q = 1/√2
        let w0 = 2.0 * core::f32::consts::PI * HIGHPASS_HZ / SAMPLE_RATE_HZ;
        let (sin_w0, cos_w0) = (libm::sinf(w0), libm::cosf(w0));
        let alpha = sin_w0 / (2.0 * core::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        let hp_coef = [
            (1.0 + cos_w0) / 2.0 / a0,
            -(1.0 + cos_w0) / a0,
            (1.0 + cos_w0) / 2.0 / a0,
            -2.0 * cos_w0 / a0,
            (1.0 - alpha) / a0,
        ];
        Self {
            blocks: 0,
            sum: 0.0,
            sum_sq: 0.0,
            sum_cube: 0.0,
            sum_quad: 0.0,
            sum_abs: 0.0,
            sum_sqrt_abs: 0.0,
            max: f32::NEG_INFINITY,
            min: f32::INFINITY,
            rel_bands: [0.0; 4],
            hp_coef,
            hp_state: [0.0; 4],
            envelope: [0.0; ENV_LEN],
        }
    }

    /// Nạp một khối; trả về Some(16 đặc trưng) khi đã đủ BLOCKS_PER_WINDOW khối
    pub fn push_block(&mut self, x: &[f32; BLOCK_LEN]) -> Option<[f32; NUM_FEATURES]> {
        // --- Thống kê miền thời gian ---
        for &v in x.iter() {
            let d = v as f64;
            let d2 = d * d;
            self.sum += d;
            self.sum_sq += d2;
            self.sum_cube += d2 * d;
            self.sum_quad += d2 * d2;
            self.sum_abs += libm::fabs(d);
            self.sum_sqrt_abs += libm::sqrt(libm::fabs(d));
            self.max = self.max.max(v);
            self.min = self.min.min(v);
        }

        // --- Năng lượng dải tần của khối (bỏ trung bình khối) ---
        let block_mean = x.iter().sum::<f32>() / BLOCK_LEN as f32;
        let mut re = [0.0f32; FFT_LEN];
        let mut im = [0.0f32; FFT_LEN];
        for (r, &v) in re.iter_mut().zip(x.iter()) {
            *r = v - block_mean;
        }
        fft(&mut re, &mut im);
        for (b, e) in self.rel_bands.iter_mut().enumerate() {
            for k in REL_BAND_EDGES[b]..REL_BAND_EDGES[b + 1] {
                *e += re[k] * re[k] + im[k] * im[k];
            }
        }

        // --- Đường bao: lọc thông cao, lấy trị tuyệt đối, trung bình mỗi ENV_DECIMATION mẫu ---
        let [b0, b1, b2, a1, a2] = self.hp_coef;
        let [mut x1, mut x2, mut y1, mut y2] = self.hp_state;
        let base = self.blocks * ENV_PER_BLOCK;
        for (i, chunk) in x.chunks_exact(ENV_DECIMATION).enumerate() {
            let mut acc = 0.0f32;
            for &v in chunk {
                let y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                x2 = x1;
                x1 = v;
                y2 = y1;
                y1 = y;
                acc += y.abs();
            }
            self.envelope[base + i] = acc / ENV_DECIMATION as f32;
        }
        self.hp_state = [x1, x2, y1, y2];

        self.blocks += 1;
        if self.blocks < BLOCKS_PER_WINDOW {
            return None;
        }
        let z = self.finish();
        *self = Self::new();
        Some(z)
    }

    fn finish(&mut self) -> [f32; NUM_FEATURES] {
        // --- Nhóm 1: hình dạng tín hiệu (mô-men trung tâm tính từ tổng lũy thừa) ---
        let n = WINDOW_LEN as f64;
        let eps = EPS as f64;
        let mean = self.sum / n;
        let (e2, e3, e4) = (self.sum_sq / n, self.sum_cube / n, self.sum_quad / n);
        let m2 = (e2 - mean * mean).max(0.0);
        let m3 = e3 - 3.0 * mean * e2 + 2.0 * mean * mean * mean;
        let m4 = e4 - 4.0 * mean * e3 + 6.0 * mean * mean * e2 - 3.0 * mean * mean * mean * mean;
        let std = libm::sqrt(m2);
        let rms = libm::sqrt(e2);
        let peak = self.max.abs().max(self.min.abs()) as f64;
        let mean_abs = self.sum_abs / n;
        let clearance_base = self.sum_sqrt_abs / n;

        // --- Nhóm 2: tỷ lệ năng lượng các dải ---
        let total: f32 = self.rel_bands.iter().sum();
        let rel = |b: usize| libm::log10f(self.rel_bands[b] / (total + EPS) + EPS);

        // --- Nhóm 3: phổ bao. Bỏ trung bình, nhân cửa sổ Hann, FFT, năng lượng trung bình mỗi bin
        // của từng dải chia cho năng lượng trung bình mỗi bin trên toàn vùng 10-300 Hz ---
        let env_mean = self.envelope.iter().sum::<f32>() / ENV_LEN as f32;
        let mut re = [0.0f32; FFT_LEN];
        let mut im = [0.0f32; FFT_LEN];
        for (i, (r, &v)) in re.iter_mut().zip(self.envelope.iter()).enumerate() {
            let hann = 0.5 - 0.5 * libm::cosf(2.0 * core::f32::consts::PI * i as f32 / ENV_LEN as f32);
            *r = (v - env_mean) * hann;
        }
        fft(&mut re, &mut im);
        let band_mean = |lo: usize, hi: usize| {
            (lo..hi).map(|k| re[k] * re[k] + im[k] * im[k]).sum::<f32>() / (hi - lo) as f32
        };
        let env_total = band_mean(ENV_BAND_EDGES[0], ENV_BAND_EDGES[6]);
        let env = |b: usize| {
            libm::log10f(band_mean(ENV_BAND_EDGES[b], ENV_BAND_EDGES[b + 1]) / (env_total + EPS) + EPS)
        };

        [
            (peak / (rms + eps)) as f32,
            (m4 / (m2 * m2 + eps)) as f32,
            (m3 / (m2 * std + eps)) as f32,
            (peak / (mean_abs + eps)) as f32,
            (rms / (mean_abs + eps)) as f32,
            (peak / (clearance_base * clearance_base + eps)) as f32,
            rel(0),
            rel(1),
            rel(2),
            rel(3),
            env(0),
            env(1),
            env(2),
            env(3),
            env(4),
            env(5),
        ]
    }
}

/// FFT cơ số 2, tại chỗ, N = 1024 (Cooley–Tukey lặp)
fn fft(re: &mut [f32; FFT_LEN], im: &mut [f32; FFT_LEN]) {
    let n = FFT_LEN;

    // Hoán vị đảo bit
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Các tầng bướm
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * core::f32::consts::PI / len as f32;
        let (w_im, w_re) = (libm::sinf(angle), libm::cosf(angle));
        let half = len / 2;
        let mut start = 0;
        while start < n {
            let (mut cur_re, mut cur_im) = (1.0f32, 0.0f32);
            for k in 0..half {
                let a = start + k;
                let b = a + half;
                let t_re = re[b] * cur_re - im[b] * cur_im;
                let t_im = re[b] * cur_im + im[b] * cur_re;
                re[b] = re[a] - t_re;
                im[b] = im[a] - t_im;
                re[a] += t_re;
                im[a] += t_im;
                let next_re = cur_re * w_re - cur_im * w_im;
                cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
            }
            start += len;
        }
        len <<= 1;
    }
}
