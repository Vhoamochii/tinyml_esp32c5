// Dữ liệu Paderborn (đề cương, mục 2 và Bảng 1): đường dẫn file, phân tập theo ổ, đọc kênh rung, cắt cửa sổ.
// Mỗi file: <chế độ>_<mã ổ>_<lần đo>.mat, nằm trong thư mục <mã ổ>/, kênh Y.vibration_1 dài 256001 mẫu ở 64 kHz.

use std::path::{Path, PathBuf};

use tinyml_core::features::{extract, WINDOW_LEN};
use tinyml_core::NUM_FEATURES;

use crate::mat::{self, MatValue};

/// 1500 vòng/phút, 0,7 N·m, 1000 N (tốc độ cũ)
pub const OLD_SPEED: &str = "N15_M07_F10";
/// 900 vòng/phút, 0,7 N·m, 1000 N (tốc độ mới)
pub const NEW_SPEED: &str = "N09_M07_F10";
pub const VIBRATION_CHANNEL: &str = "vibration_1";
/// Số lần đo mỗi ổ ở mỗi chế độ
pub const MEASUREMENTS: u32 = 20;
/// Số cửa sổ lấy rải đều từ mỗi bản ghi (trong 250 cửa sổ không chồng nhau)
pub const WINDOWS_PER_RECORD: usize = 32;

/// Bảng 1 của đề cương: phân tập theo ổ vật lý
pub const TRAIN_BEARINGS: [&str; 8] = ["K001", "K002", "K003", "KI04", "KI16", "KI17", "KA04", "KA15"];
pub const CONFIG_BEARINGS: [&str; 3] = ["K004", "KI14", "KA16"];
pub const TEST_BEARINGS: [&str; 6] = ["K005", "K006", "KI18", "KI21", "KA22", "KA30"];

/// Lớp chẩn đoán từ mã ổ: 0 = khỏe (K0..), 1 = lỗi vòng trong (KI..), 2 = lỗi vòng ngoài (KA..).
/// Ổ hỏng cả hai vòng (KB..) không thuộc ba lớp nên trả về None.
pub fn class_of(bearing: &str) -> Option<u8> {
    if bearing.starts_with("KI") {
        Some(1)
    } else if bearing.starts_with("KA") {
        Some(2)
    } else if bearing.starts_with("K0") {
        Some(0)
    } else {
        None
    }
}

/// Thư mục dữ liệu: biến môi trường PADERBORN_DIR, mặc định $HOME/Work/Dataset
pub fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("PADERBORN_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    Path::new(&home).join("Work").join("Dataset")
}

/// Đường dẫn file của một lần đo, ví dụ <gốc>/K001/N15_M07_F10_K001_1.mat
pub fn measurement_path(root: &Path, condition: &str, bearing: &str, measurement: u32) -> PathBuf {
    root.join(bearing).join(format!("{condition}_{bearing}_{measurement}.mat"))
}

/// Đọc kênh rung của một file .mat, đổi sang FP32 (bo cũng tính bằng FP32)
pub fn read_vibration(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("khong doc duoc {}: {e}", path.display()))?;
    let variables = mat::parse(&bytes)?;
    let var = variables.first().ok_or("file khong co bien nao")?;
    let y = var.value.get(0, "Y").ok_or("bien khong co truong Y")?;
    for i in 0..y.element_count() {
        if y.get(i, "Name").and_then(MatValue::as_str) == Some(VIBRATION_CHANNEL) {
            let data = y.get(i, "Data").and_then(MatValue::as_f64).ok_or("kenh rung khong co Data dang so")?;
            return Ok(data.iter().map(|&v| v as f32).collect());
        }
    }
    Err(format!("khong tim thay kenh {VIBRATION_CHANNEL} trong Y"))
}

/// Số cửa sổ WINDOW_LEN mẫu không chồng nhau trong tín hiệu
pub fn window_count(signal: &[f32]) -> usize {
    signal.len() / WINDOW_LEN
}

/// Lấy `count` cửa sổ không chồng nhau, rải đều trong tín hiệu, giữ thứ tự thời gian
pub fn windows(signal: &[f32], count: usize) -> Vec<[f32; WINDOW_LEN]> {
    let total = window_count(signal);
    let count = count.min(total);
    (0..count)
        .map(|i| {
            let w = i * total / count;
            let mut out = [0.0f32; WINDOW_LEN];
            out.copy_from_slice(&signal[w * WINDOW_LEN..(w + 1) * WINDOW_LEN]);
            out
        })
        .collect()
}

/// Một bản ghi (một file .mat) đã cắt cửa sổ và tính 16 đặc trưng (chưa chuẩn hóa)
pub struct Record {
    pub bearing: &'static str,
    pub class: u8,
    pub measurement: u32,
    pub features: Vec<[f32; NUM_FEATURES]>, // WINDOWS_PER_RECORD cửa sổ, theo thứ tự thời gian
}

/// Đọc các lần đo `measurements` của các ổ `bearings` ở chế độ `condition`
pub fn load_records(
    root: &Path,
    condition: &str,
    bearings: &[&'static str],
    measurements: impl IntoIterator<Item = u32> + Clone,
) -> Result<Vec<Record>, String> {
    let mut out = Vec::new();
    for &bearing in bearings {
        let class = class_of(bearing).ok_or_else(|| format!("ma o {bearing} khong thuoc 3 lop"))?;
        let before = out.len();
        for measurement in measurements.clone() {
            let path = measurement_path(root, condition, bearing, measurement);
            let signal = read_vibration(&path)?;
            let features = windows(&signal, WINDOWS_PER_RECORD).iter().map(extract).collect();
            out.push(Record { bearing, class, measurement, features });
        }
        println!("[LOAD] {condition} {bearing}: {} ban ghi", out.len() - before);
    }
    Ok(out)
}
