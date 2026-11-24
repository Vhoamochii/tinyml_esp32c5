// Kiểm tra bộ đọc .mat và bộ trích xuất trên một file Paderborn; so các số in ra với Python.
//   cargo run --release --bin inspect_mat                      # mặc định: K001/N15_M07_F10_K001_1.mat
//   cargo run --release --bin inspect_mat -- <đường dẫn .mat>

use std::path::PathBuf;

use pc_sim::paderborn;
use tinyml_core::features::{extract, FEATURE_NAMES};

fn main() {
    let path = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        paderborn::measurement_path(&paderborn::data_dir(), paderborn::OLD_SPEED, "K001", 1)
    });
    println!("[MAT] file: {}", path.display());

    let x = match paderborn::read_vibration(&path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("[MAT][ERROR] {e}");
            std::process::exit(1);
        }
    };

    // Thống kê tính bằng f64 trên các giá trị đã đổi sang f32
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let std = (x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    let min = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("[MAT] {}: {} mau", paderborn::VIBRATION_CHANNEL, x.len());
    println!("[MAT] mean={mean:.9} std={std:.9} min={min:.9} max={max:.9}");
    println!("[MAT] 5 mau dau: {:.9} {:.9} {:.9} {:.9} {:.9}", x[0], x[1], x[2], x[3], x[4]);

    let windows = paderborn::windows(&x, paderborn::WINDOWS_PER_RECORD);
    println!(
        "[WIN] {} cua so khong chong nhau, lay {} cua so rai deu",
        paderborn::window_count(&x),
        windows.len()
    );

    let z = extract(windows[0]);
    println!("[FEAT] 16 dac trung cua cua so dau tien:");
    for (name, value) in FEATURE_NAMES.iter().zip(z.iter()) {
        println!("[FEAT] {name:<18} {value:.6}");
    }
}
