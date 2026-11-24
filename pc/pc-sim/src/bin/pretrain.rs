// Bước 3: chuẩn hóa 16 đặc trưng + huấn luyện sẵn lớp cuối trên 8 ổ huấn luyện ở tốc độ cũ (N15_M07_F10).
// Kết quả (hệ số chuẩn hóa, trọng số) được ghi vào tinyml-core/src/pretrained.rs: trọng số ban đầu chung
// cho mọi lần chạy thí nghiệm và cho firmware.
//   cargo run --release --bin pretrain

use std::fmt::Write as _;
use std::path::Path;

use pc_sim::paderborn::{self, Record};
use tinyml_core::features::{normalize, FEATURE_NAMES};
use tinyml_core::head::Head;
use tinyml_core::rng::XorShift32;
use tinyml_core::{NUM_CLASSES, NUM_FEATURES};

const TRAIN_MEASUREMENTS: (u32, u32) = (1, 16); // Lần đo dùng để học
const VAL_MEASUREMENTS: (u32, u32) = (17, 20); // Lần đo để kiểm tra (cùng ổ huấn luyện)
const EPOCHS: usize = 50;
const LEARNING_RATE: f32 = 0.05;
const SEED: u32 = 2026;
const CLASS_NAMES: [&str; NUM_CLASSES] = ["khoe", "vong trong", "vong ngoai"];

type Features = [f32; NUM_FEATURES];

fn load(root: &Path, condition: &str, bearings: &[&'static str], range: (u32, u32)) -> Vec<Record> {
    match paderborn::load_records(root, condition, bearings, range.0..=range.1) {
        Ok(records) => records,
        Err(e) => {
            eprintln!("[PRETRAIN][ERROR] {e}");
            std::process::exit(1);
        }
    }
}

/// Trung bình và độ lệch chuẩn của từng đặc trưng trên tập huấn luyện
fn fit_normalization(records: &[Record]) -> (Features, Features) {
    let mut sum = [0.0f64; NUM_FEATURES];
    let mut sum_sq = [0.0f64; NUM_FEATURES];
    let mut n = 0.0f64;
    for z in records.iter().flat_map(|r| r.features.iter()) {
        for f in 0..NUM_FEATURES {
            sum[f] += z[f] as f64;
            sum_sq[f] += (z[f] as f64).powi(2);
        }
        n += 1.0;
    }
    let mut mean = [0.0f32; NUM_FEATURES];
    let mut std = [1.0f32; NUM_FEATURES];
    for f in 0..NUM_FEATURES {
        let m = sum[f] / n;
        mean[f] = m as f32;
        std[f] = ((sum_sq[f] / n - m * m).max(0.0).sqrt() as f32).max(1e-6);
    }
    (mean, std)
}

fn normalized(z: &Features, mean: &Features, std: &Features) -> Features {
    let mut out = *z;
    normalize(&mut out, mean, std);
    out
}

/// BA theo đề cương: độ nhạy của từng ổ, trung bình các ổ trong cùng lớp, rồi trung bình đều các lớp.
/// Trả về (BA %, danh sách (ổ, độ nhạy %)).
fn evaluate(head: &Head<NUM_CLASSES>, records: &[Record], mean: &Features, std: &Features) -> (f32, Vec<(&'static str, f32)>) {
    let mut per_bearing: Vec<(&'static str, u8, u32, u32)> = Vec::new(); // (ổ, lớp, đúng, tổng)
    for r in records {
        let idx = match per_bearing.iter().position(|b| b.0 == r.bearing) {
            Some(i) => i,
            None => {
                per_bearing.push((r.bearing, r.class, 0, 0));
                per_bearing.len() - 1
            }
        };
        for z in &r.features {
            let ok = head.predict(&normalized(z, mean, std)) == r.class as usize;
            per_bearing[idx].2 += ok as u32;
            per_bearing[idx].3 += 1;
        }
    }
    let recalls: Vec<(&'static str, f32)> =
        per_bearing.iter().map(|b| (b.0, b.2 as f32 / b.3 as f32 * 100.0)).collect();
    let mut class_scores = Vec::new();
    for c in 0..NUM_CLASSES as u8 {
        let rs: Vec<f32> = per_bearing.iter().zip(&recalls).filter(|(b, _)| b.1 == c).map(|(_, r)| r.1).collect();
        if !rs.is_empty() {
            class_scores.push(rs.iter().sum::<f32>() / rs.len() as f32);
        }
    }
    let ba = class_scores.iter().sum::<f32>() / class_scores.len().max(1) as f32;
    (ba, recalls)
}

fn print_eval(name: &str, head: &Head<NUM_CLASSES>, records: &[Record], mean: &Features, std: &Features) {
    let (ba, recalls) = evaluate(head, records, mean, std);
    let detail: Vec<String> = recalls.iter().map(|(b, r)| format!("{b} {r:.1}%")).collect();
    println!("[EVAL] {name:<32} BA = {ba:6.2}% | {}", detail.join(", "));
}

/// Ghi hệ số chuẩn hóa và trọng số thành mã Rust (định dạng {:?} của f32 đọc lại được chính xác)
fn write_pretrained(head: &Head<NUM_CLASSES>, mean: &Features, std: &Features) -> std::io::Result<std::path::PathBuf> {
    let mut s = String::new();
    let _ = writeln!(s, "// FILE SINH TỰ ĐỘNG bởi `cargo run --release --bin pretrain` trong pc/pc-sim. Không sửa tay.");
    let _ = writeln!(s, "// Hệ số chuẩn hóa 16 đặc trưng và trọng số lớp cuối huấn luyện sẵn trên 8 ổ huấn luyện ở N15_M07_F10.");
    let _ = writeln!(
        s,
        "// Lần đo {}-{}, {} cửa sổ mỗi bản ghi, {} lượt, lr = {}, hạt giống {}.",
        TRAIN_MEASUREMENTS.0, TRAIN_MEASUREMENTS.1, paderborn::WINDOWS_PER_RECORD, EPOCHS, LEARNING_RATE, SEED
    );
    let _ = writeln!(s);
    let _ = writeln!(s, "use crate::head::Head;");
    let _ = writeln!(s, "use crate::{{NUM_CLASSES, NUM_FEATURES}};");
    let _ = writeln!(s);
    let _ = writeln!(s, "pub const AVAILABLE: bool = true;");
    let _ = writeln!(s, "pub const FEATURE_MEAN: [f32; NUM_FEATURES] = {:?};", mean);
    let _ = writeln!(s, "pub const FEATURE_STD: [f32; NUM_FEATURES] = {:?};", std);
    let _ = writeln!(s, "pub const WEIGHTS: [[f32; NUM_FEATURES]; NUM_CLASSES] = {:?};", head.weights);
    let _ = writeln!(s, "pub const BIAS: [f32; NUM_CLASSES] = {:?};", head.bias);
    let _ = writeln!(s);
    let _ = writeln!(s, "/// Lớp cuối với trọng số huấn luyện sẵn (trọng số ban đầu chung cho mọi lần chạy)");
    let _ = writeln!(s, "pub fn head() -> Head<NUM_CLASSES> {{");
    let _ = writeln!(s, "    Head {{ weights: WEIGHTS, bias: BIAS }}");
    let _ = writeln!(s, "}}");

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tinyml-core/src/pretrained.rs");
    std::fs::write(&path, s)?;
    Ok(path)
}

fn main() {
    let root = paderborn::data_dir();
    println!("[PRETRAIN] Du lieu: {}", root.display());

    let train = load(&root, paderborn::OLD_SPEED, &paderborn::TRAIN_BEARINGS, TRAIN_MEASUREMENTS);
    let val = load(&root, paderborn::OLD_SPEED, &paderborn::TRAIN_BEARINGS, VAL_MEASUREMENTS);
    let config_old = load(&root, paderborn::OLD_SPEED, &paderborn::CONFIG_BEARINGS, (1, paderborn::MEASUREMENTS));
    let config_new = load(&root, paderborn::NEW_SPEED, &paderborn::CONFIG_BEARINGS, (1, paderborn::MEASUREMENTS));

    // Chuẩn hóa: hệ số chỉ lấy từ tập học
    let (mean, std) = fit_normalization(&train);
    let mut samples: Vec<(Features, u8)> = train
        .iter()
        .flat_map(|r| r.features.iter().map(move |z| (normalized(z, &mean, &std), r.class)))
        .collect();
    println!("[PRETRAIN] {} cua so hoc, {} luot, lr = {}", samples.len(), EPOCHS, LEARNING_RATE);

    // Huấn luyện lớp cuối: SGD từng mẫu, xáo trộn mỗi lượt
    let mut head = Head::<NUM_CLASSES>::zeros();
    let mut rng = XorShift32::new(SEED);
    for epoch in 1..=EPOCHS {
        for i in (1..samples.len()).rev() {
            let j = rng.below(i as u32 + 1) as usize;
            samples.swap(i, j);
        }
        let mut loss = 0.0f32;
        for (z, label) in &samples {
            loss += head.sgd_step(z, *label, LEARNING_RATE);
        }
        if epoch == 1 || epoch % 10 == 0 {
            println!("[PRETRAIN] luot {epoch:>3}: loss = {:.4}", loss / samples.len() as f32);
        }
    }

    println!();
    print_eval("Hoc: 8 o, N15, lan do 1-16", &head, &train, &mean, &std);
    print_eval("Kiem tra: 8 o, N15, lan do 17-20", &head, &val, &mean, &std);
    print_eval("O cau hinh moi, N15 (toc do cu)", &head, &config_old, &mean, &std);
    print_eval("O cau hinh moi, N09 (toc do moi)", &head, &config_new, &mean, &std);

    // Trung bình đặc trưng đã chuẩn hóa theo lớp: đặc trưng nào tách được các lớp
    println!("\n[FEAT] Trung binh dac trung da chuan hoa tren tap hoc, theo lop:");
    println!("[FEAT] {:<18} {:>10} {:>10} {:>10}", "dac trung", CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]);
    for (f, name) in FEATURE_NAMES.iter().enumerate() {
        let mut sums = [0.0f64; NUM_CLASSES];
        let mut counts = [0u32; NUM_CLASSES];
        for (z, label) in &samples {
            sums[*label as usize] += z[f] as f64;
            counts[*label as usize] += 1;
        }
        let m: Vec<f64> = (0..NUM_CLASSES).map(|c| sums[c] / counts[c].max(1) as f64).collect();
        println!("[FEAT] {name:<18} {:>10.3} {:>10.3} {:>10.3}", m[0], m[1], m[2]);
    }

    match write_pretrained(&head, &mean, &std) {
        Ok(path) => println!("\n[SAVE] Da ghi {}", path.display()),
        Err(e) => {
            eprintln!("[SAVE][ERROR] {e}");
            std::process::exit(1);
        }
    }
}
