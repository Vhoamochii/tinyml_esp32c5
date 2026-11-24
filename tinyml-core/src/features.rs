// Bộ trích xuất đặc trưng GIAI ĐOẠN 1: 16 đặc trưng tính tay, FP32, không có trọng số.
// Thay tạm cho bộ trích xuất INT8 học được trong đề cương (giai đoạn 2). Giữ nguyên đầu ra 16 số
// nên lớp cuối, mảng ô P/Q/R và lịch học không phải đổi khi thay bộ trích xuất.
// Chạy giống hệt trên Ubuntu và ESP32-C5 (no_std, không heap).
//
// Một cửa sổ = 16 khối x 1024 mẫu = 16384 mẫu = 256 ms ở 64 kHz, đủ dài để thấy ~19 cú gõ của lỗi
// vòng ngoài. Bộ trích xuất nhận từng khối (bo không phải giữ cả cửa sổ), gồm 3 nhóm đặc trưng,
// đều không phụ thuộc độ to của tín hiệu:
//   0-5   hình dạng tín hiệu (có va đập hay không)
//   6-9   năng lượng dải tần tương đối (tỷ lệ trên tổng năng lượng), ước lượng trên 4 trong 16 khối
//   10-15 phổ bao theo DẢI TẦN CỐ ĐỊNH (nhịp va đập), không cần biết tốc độ quay. Khi đổi tốc độ,
//         nhịp gõ dời sang dải khác: đó là độ lệch mà bo phải tự học lại. Bỏ các bin quanh bội số
//         50 Hz (nhiễu điện lưới trong dữ liệu Paderborn).

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
/// Năng lượng dải tương đối chỉ tính trên các khối 0, 4, 8, 12: bớt 12 FFT mỗi cửa sổ để kịp hạn 256 ms
/// trên ESP32-C5 (không có FPU; đo trên bo: FFT mọi khối làm một cửa sổ mất ~500 ms)
const REL_BAND_BLOCK_STRIDE: usize = 4;
/// Ranh giới 6 dải phổ bao theo bin FFT đường bao (4000 / 1024 = 3,906 Hz/bin):
/// 10, 20, 35, 60, 100, 170, 300 Hz (làm tròn lên bin kế tiếp)
const ENV_BAND_EDGES: [usize; 7] = [3, 6, 9, 16, 26, 44, 77];
/// Tần số điện lưới (Đức). Đỉnh cố định ~50 Hz xuất hiện ở mọi ổ, ở cả hai tốc độ: không phải tín hiệu ổ
const MAINS_HZ: usize = 50;
/// Tần số lấy mẫu đường bao (64000 / 16), dạng số nguyên
const ENV_RATE_HZ: usize = 4000;

/// Bin phổ bao k cách một bội số của 50 Hz dưới 1,5 bin (5,9 Hz): bị loại khỏi đặc trưng.
/// Tần số bin k là k·4000/1024 Hz; nhân cả hai vế với 1024 để tính hoàn toàn bằng số nguyên.
/// Trong vùng 10-300 Hz loại 16 bin: 3 bin quanh mỗi bội 50..250 Hz và bin 76 (296,9 Hz).
fn is_mains_bin(k: usize) -> bool {
    let freq = k * ENV_RATE_HZ;
    let period = MAINS_HZ * FFT_LEN;
    let r = freq % period;
    r.min(period - r) * 2 < 3 * ENV_RATE_HZ
}

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

/// Bảng twiddle cho FFT 1024 điểm: cos và sin của 2πj/1024, j = 0..511
struct Twiddles {
    cos: [f32; FFT_LEN / 2],
    sin: [f32; FFT_LEN / 2],
}

impl Twiddles {
    fn new() -> Self {
        let mut t = Self { cos: [0.0; FFT_LEN / 2], sin: [0.0; FFT_LEN / 2] };
        for j in 0..FFT_LEN / 2 {
            let angle = 2.0 * core::f32::consts::PI * j as f32 / FFT_LEN as f32;
            t.cos[j] = libm::cosf(angle);
            t.sin[j] = libm::sinf(angle);
        }
        t
    }

    /// cos(2πi/1024) với i = 0..1023 (dùng cho cửa sổ Hann)
    fn cos_full(&self, i: usize) -> f32 {
        match i {
            0..512 => self.cos[i],
            512 => -1.0,
            _ => self.cos[FFT_LEN - i],
        }
    }
}

/// Bộ trích xuất nhận từng khối 1024 mẫu; đủ 16 khối thì trả 16 đặc trưng và tự đặt lại.
/// Mỗi cửa sổ tính độc lập (bộ lọc cũng đặt lại), nên kết quả không phụ thuộc cửa sổ trước.
pub struct WindowExtractor {
    blocks: usize,
    // Thống kê miền thời gian: mỗi khối cộng bằng f32, rồi cộng tổng khối vào f64
    sum: f64,
    sum_sq: f64,
    sum_cube: f64,
    sum_quad: f64,
    sum_abs: f64,
    sum_sqrt_abs: f64,
    max: f32,
    min: f32,
    // Năng lượng 4 dải, cộng qua các khối 0, 4, 8, 12
    rel_bands: [f32; 4],
    // Bộ lọc thông cao bậc 2 (Butterworth): hệ số b0, b1, b2, a1, a2 và trạng thái x1, x2, y1, y2
    hp_coef: [f32; 5],
    hp_state: [f32; 4],
    // Đường bao của cả cửa sổ
    envelope: [f32; ENV_LEN],
    // Tính một lần khi tạo, giữ nguyên khi đặt lại
    twiddles: Twiddles,
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
        let mut extractor = Self {
            blocks: 0,
            sum: 0.0,
            sum_sq: 0.0,
            sum_cube: 0.0,
            sum_quad: 0.0,
            sum_abs: 0.0,
            sum_sqrt_abs: 0.0,
            max: 0.0,
            min: 0.0,
            rel_bands: [0.0; 4],
            hp_coef,
            hp_state: [0.0; 4],
            envelope: [0.0; ENV_LEN],
            twiddles: Twiddles::new(),
        };
        extractor.reset();
        extractor
    }

    /// Đặt lại trạng thái cho cửa sổ mới; giữ hệ số lọc và bảng twiddle
    fn reset(&mut self) {
        self.blocks = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.sum_cube = 0.0;
        self.sum_quad = 0.0;
        self.sum_abs = 0.0;
        self.sum_sqrt_abs = 0.0;
        self.max = f32::NEG_INFINITY;
        self.min = f32::INFINITY;
        self.rel_bands = [0.0; 4];
        self.hp_state = [0.0; 4];
        self.envelope = [0.0; ENV_LEN];
    }

    /// Nạp một khối; trả về Some(16 đặc trưng) khi đã đủ BLOCKS_PER_WINDOW khối
    pub fn push_block(&mut self, x: &[f32; BLOCK_LEN]) -> Option<[f32; NUM_FEATURES]> {
        // --- Thống kê miền thời gian: cộng trong khối bằng f32 (1024 số hạng), rồi vào f64 ---
        let (mut s1, mut s2, mut s3, mut s4, mut sa, mut ssa) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for &v in x.iter() {
            let v2 = v * v;
            let a = v.abs();
            s1 += v;
            s2 += v2;
            s3 += v2 * v;
            s4 += v2 * v2;
            sa += a;
            ssa += libm::sqrtf(a);
            self.max = self.max.max(v);
            self.min = self.min.min(v);
        }
        self.sum += s1 as f64;
        self.sum_sq += s2 as f64;
        self.sum_cube += s3 as f64;
        self.sum_quad += s4 as f64;
        self.sum_abs += sa as f64;
        self.sum_sqrt_abs += ssa as f64;

        // --- Năng lượng dải tần của khối (bỏ trung bình khối), chỉ ở các khối 0, 4, 8, 12 ---
        if self.blocks % REL_BAND_BLOCK_STRIDE == 0 {
            let block_mean = s1 / BLOCK_LEN as f32;
            let mut re = [0.0f32; FFT_LEN];
            let mut im = [0.0f32; FFT_LEN];
            for (r, &v) in re.iter_mut().zip(x.iter()) {
                *r = v - block_mean;
            }
            fft(&mut re, &mut im, &self.twiddles);
            for (b, e) in self.rel_bands.iter_mut().enumerate() {
                for k in REL_BAND_EDGES[b]..REL_BAND_EDGES[b + 1] {
                    *e += re[k] * re[k] + im[k] * im[k];
                }
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
        self.reset();
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
        // của từng dải chia cho năng lượng trung bình mỗi bin trên toàn vùng 10-300 Hz.
        // Không tính các bin nhiễu điện lưới (is_mains_bin) ---
        let env_mean = self.envelope.iter().sum::<f32>() / ENV_LEN as f32;
        let mut re = [0.0f32; FFT_LEN];
        let mut im = [0.0f32; FFT_LEN];
        for (i, (r, &v)) in re.iter_mut().zip(self.envelope.iter()).enumerate() {
            let hann = 0.5 - 0.5 * self.twiddles.cos_full(i);
            *r = (v - env_mean) * hann;
        }
        fft(&mut re, &mut im, &self.twiddles);
        let band_mean = |lo: usize, hi: usize| {
            let (mut sum, mut count) = (0.0f32, 0u32);
            for k in (lo..hi).filter(|&k| !is_mains_bin(k)) {
                sum += re[k] * re[k] + im[k] * im[k];
                count += 1;
            }
            sum / count.max(1) as f32
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

/// FFT cơ số 2, tại chỗ, N = 1024 (Cooley–Tukey lặp), hệ số xoay lấy từ bảng twiddle
fn fft(re: &mut [f32; FFT_LEN], im: &mut [f32; FFT_LEN], tw: &Twiddles) {
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

    // Các tầng bướm: hệ số xoay của phần tử k ở tầng dài `len` là e^(-2πi·k/len) = bảng[k·N/len] (lấy liên hợp)
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let (w_re, w_im) = (tw.cos[k * step], -tw.sin[k * step]);
                let a = start + k;
                let b = a + half;
                let t_re = re[b] * w_re - im[b] * w_im;
                let t_im = re[b] * w_im + im[b] * w_re;
                re[b] = re[a] - t_re;
                im[b] = im[a] - t_im;
                re[a] += t_re;
                im[a] += t_im;
            }
            start += len;
        }
        len <<= 1;
    }
}
