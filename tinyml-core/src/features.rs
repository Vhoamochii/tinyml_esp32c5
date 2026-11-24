// Bộ trích xuất đặc trưng GIAI ĐOẠN 1: 16 đặc trưng tính tay, FP32, không có trọng số.
// Thay tạm cho bộ trích xuất INT8 học được trong đề cương (giai đoạn 2). Giữ nguyên đầu ra 16 số
// nên lớp cuối, mảng ô P/Q/R và lịch học không phải đổi khi thay bộ trích xuất.
// Chạy giống hệt trên Ubuntu và ESP32-C5 (no_std, không heap).

use crate::NUM_FEATURES;

/// Độ dài cửa sổ: 1024 mẫu = 16 ms ở 64 kHz
pub const WINDOW_LEN: usize = 1024;
/// Tần số lấy mẫu kênh rung Paderborn
pub const SAMPLE_RATE_HZ: f32 = 64_000.0;

/// Tên 16 đặc trưng theo đúng thứ tự trong vectơ đầu ra
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "rms",
    "std",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "impulse_factor",
    "shape_factor",
    "clearance_factor",
    "log_band_0_1k",
    "log_band_1_2k",
    "log_band_2_4k",
    "log_band_4_8k",
    "log_band_8_16k",
    "log_band_16_32k",
];

/// Ranh giới 6 dải tần theo chỉ số bin FFT (độ phân giải 64000 / 1024 = 62,5 Hz):
/// 1 kHz = bin 16, 2 kHz = bin 32, ... 32 kHz = bin 512 (Nyquist). Bỏ bin 0 (thành phần một chiều).
const BAND_EDGES: [usize; 7] = [1, 16, 32, 64, 128, 256, 513];

const EPS: f32 = 1e-12;

/// Tính 16 đặc trưng của một cửa sổ rung
pub fn extract(x: &[f32; WINDOW_LEN]) -> [f32; NUM_FEATURES] {
    let n = WINDOW_LEN as f32;

    // --- Miền thời gian ---
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut sum_sqrt_abs = 0.0f32;
    let mut max = f32::NEG_INFINITY;
    let mut min = f32::INFINITY;
    for &v in x.iter() {
        sum += v;
        sum_sq += v * v;
        sum_abs += v.abs();
        sum_sqrt_abs += libm::sqrtf(v.abs());
        max = max.max(v);
        min = min.min(v);
    }
    let mean = sum / n;
    let mean_abs = sum_abs / n;
    let rms = libm::sqrtf(sum_sq / n);
    let peak = max.abs().max(min.abs());

    let mut m2 = 0.0f32;
    let mut m3 = 0.0f32;
    let mut m4 = 0.0f32;
    for &v in x.iter() {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
    let std = libm::sqrtf(m2);
    let clearance_base = sum_sqrt_abs / n;

    // --- Miền tần số: năng lượng 6 dải, lấy log10 ---
    let bands = band_energies(x, mean);

    [
        rms,
        std,
        peak,
        max - min,
        peak / (rms + EPS),
        m4 / (m2 * m2 + EPS),
        m3 / (m2 * std + EPS),
        peak / (mean_abs + EPS),
        rms / (mean_abs + EPS),
        peak / (clearance_base * clearance_base + EPS),
        libm::log10f(bands[0] + EPS),
        libm::log10f(bands[1] + EPS),
        libm::log10f(bands[2] + EPS),
        libm::log10f(bands[3] + EPS),
        libm::log10f(bands[4] + EPS),
        libm::log10f(bands[5] + EPS),
    ]
}

/// Năng lượng trung bình mỗi dải tần: tổng |X_k|^2 / N^2 trên các bin của dải, sau khi bỏ trung bình
fn band_energies(x: &[f32; WINDOW_LEN], mean: f32) -> [f32; 6] {
    let mut re = [0.0f32; WINDOW_LEN];
    let mut im = [0.0f32; WINDOW_LEN];
    for (r, &v) in re.iter_mut().zip(x.iter()) {
        *r = v - mean;
    }
    fft(&mut re, &mut im);

    let scale = 1.0 / (WINDOW_LEN as f32 * WINDOW_LEN as f32);
    let mut out = [0.0f32; 6];
    for (b, e) in out.iter_mut().enumerate() {
        for k in BAND_EDGES[b]..BAND_EDGES[b + 1] {
            *e += (re[k] * re[k] + im[k] * im[k]) * scale;
        }
    }
    out
}

/// FFT cơ số 2, tại chỗ, N = 1024 (Cooley–Tukey lặp)
fn fft(re: &mut [f32; WINDOW_LEN], im: &mut [f32; WINDOW_LEN]) {
    let n = WINDOW_LEN;

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
