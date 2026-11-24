// Bước 3: chuẩn hóa 16 đặc trưng + huấn luyện sẵn lớp cuối trên các ổ huấn luyện ở tốc độ cũ (N15_M07_F10).
// Kết quả (hệ số chuẩn hóa, trọng số) được ghi vào tinyml-core/src/pretrained.rs: trọng số ban đầu chung
// cho mọi lần chạy thí nghiệm và cho firmware.
// Chẩn đoán kèm theo: ma trận nhầm lẫn theo từng ổ, và kiểm tra chéo bỏ-một-ổ (khả năng tổng quát sang ổ mới).
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
/// Vị trí đặc trưng phổ bao đầu tiên trong vectơ z (6 đặc trưng cuối)
const ENV_FIRST: usize = 10;

type Features = [f32; NUM_FEATURES];
type Samples = Vec<(Features, u8)>;

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
fn fit_normalization<'a>(records: impl IntoIterator<Item = &'a Record>) -> (Features, Features) {
    let mut sum = [0.0f64; NUM_FEATURES];
    let mut sum_sq = [0.0f64; NUM_FEATURES];
    let mut n = 0.0f64;
    for z in records.into_iter().flat_map(|r| r.features.iter()) {
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

/// Các cặp (đặc trưng đã chuẩn hóa, nhãn) để học
fn to_samples<'a>(records: impl IntoIterator<Item = &'a Record>, mean: &Features, std: &Features) -> Samples {
    records
        .into_iter()
        .flat_map(|r| r.features.iter().map(move |z| (normalized(z, mean, std), r.class)))
        .collect()
}

/// Huấn luyện lớp cuối từ trọng số 0: SGD từng mẫu, xáo trộn mỗi lượt
fn train_head(samples: &mut Samples, print_loss: bool) -> Head<NUM_CLASSES> {
    let mut head = Head::<NUM_CLASSES>::zeros();
    let mut rng = XorShift32::new(SEED);
    for epoch in 1..=EPOCHS {
        for i in (1..samples.len()).rev() {
            let j = rng.below(i as u32 + 1) as usize;
            samples.swap(i, j);
        }
        let mut loss = 0.0f32;
        for (z, label) in samples.iter() {
            loss += head.sgd_step(z, *label, LEARNING_RATE);
        }
        if print_loss && (epoch == 1 || epoch % 10 == 0) {
            println!("[PRETRAIN] luot {epoch:>3}: loss = {:.4}", loss / samples.len() as f32);
        }
    }
    head
}

/// Kết quả trên một ổ: số cửa sổ bị đoán thành từng lớp (một hàng của ma trận nhầm lẫn)
struct BearingResult {
    bearing: &'static str,
    class: u8,
    predicted: [u32; NUM_CLASSES],
}

impl BearingResult {
    fn share(&self, c: usize) -> f32 {
        self.predicted[c] as f32 / self.predicted.iter().sum::<u32>().max(1) as f32 * 100.0
    }

    /// Độ nhạy: % cửa sổ đoán đúng lớp của ổ
    fn recall(&self) -> f32 {
        self.share(self.class as usize)
    }

    fn describe(&self) -> String {
        let shares: Vec<String> = (0..NUM_CLASSES).map(|c| format!("{} {:5.1}%", CLASS_NAMES[c], self.share(c))).collect();
        format!(
            "{} ({:<10}): dung {:5.1}% | bi doan thanh: {}",
            self.bearing,
            CLASS_NAMES[self.class as usize],
            self.recall(),
            shares.join(", ")
        )
    }
}

fn per_bearing<'a>(
    head: &Head<NUM_CLASSES>,
    records: impl IntoIterator<Item = &'a Record>,
    mean: &Features,
    std: &Features,
) -> Vec<BearingResult> {
    let mut out: Vec<BearingResult> = Vec::new();
    for r in records {
        let idx = match out.iter().position(|b| b.bearing == r.bearing) {
            Some(i) => i,
            None => {
                out.push(BearingResult { bearing: r.bearing, class: r.class, predicted: [0; NUM_CLASSES] });
                out.len() - 1
            }
        };
        for z in &r.features {
            out[idx].predicted[head.predict(&normalized(z, mean, std))] += 1;
        }
    }
    out
}

/// BA theo đề cương: độ nhạy của từng ổ, trung bình các ổ trong cùng lớp, rồi trung bình đều các lớp
fn balanced_accuracy(results: &[BearingResult]) -> f32 {
    let mut class_scores = Vec::new();
    for c in 0..NUM_CLASSES as u8 {
        let rs: Vec<f32> = results.iter().filter(|b| b.class == c).map(BearingResult::recall).collect();
        if !rs.is_empty() {
            class_scores.push(rs.iter().sum::<f32>() / rs.len() as f32);
        }
    }
    class_scores.iter().sum::<f32>() / class_scores.len().max(1) as f32
}

fn print_eval(name: &str, head: &Head<NUM_CLASSES>, records: &[Record], mean: &Features, std: &Features) {
    let results = per_bearing(head, records, mean, std);
    println!("[EVAL] {name:<32} BA = {:6.2}%", balanced_accuracy(&results));
    for b in &results {
        println!("[EVAL]     {}", b.describe());
    }
}

/// Trung bình 6 đặc trưng phổ bao (chưa chuẩn hóa) theo từng ổ: ổ nào có đỉnh nhịp gõ ở dải nào.
/// Giá trị > 0 nghĩa là dải đó cao hơn mức trung bình của vùng 10-300 Hz.
fn print_envelope_by_bearing(name: &str, records: &[Record]) {
    println!("[ENV] {name}");
    let mut bearings: Vec<(&'static str, u8)> = Vec::new();
    for r in records {
        if !bearings.iter().any(|b| b.0 == r.bearing) {
            bearings.push((r.bearing, r.class));
        }
    }
    for (bearing, class) in bearings {
        let mut sums = [0.0f64; NUM_FEATURES - ENV_FIRST];
        let mut n = 0u32;
        for z in records.iter().filter(|r| r.bearing == bearing).flat_map(|r| r.features.iter()) {
            for (s, v) in sums.iter_mut().zip(&z[ENV_FIRST..]) {
                *s += *v as f64;
            }
            n += 1;
        }
        let cols: String = sums.iter().map(|s| format!("{:>10.2}", s / n.max(1) as f64)).collect();
        println!("[ENV]   {bearing} ({:<10}){cols}", CLASS_NAMES[class as usize]);
    }
}

/// Ghi hệ số chuẩn hóa và trọng số thành mã Rust (định dạng {:?} của f32 đọc lại được chính xác)
fn write_pretrained(head: &Head<NUM_CLASSES>, mean: &Features, std: &Features) -> std::io::Result<std::path::PathBuf> {
    let mut s = String::new();
    let _ = writeln!(s, "// FILE SINH TỰ ĐỘNG bởi `cargo run --release --bin pretrain` trong pc/pc-sim. Không sửa tay.");
    let _ = writeln!(
        s,
        "// Hệ số chuẩn hóa 16 đặc trưng và trọng số lớp cuối huấn luyện sẵn trên {} ở N15_M07_F10.",
        paderborn::TRAIN_BEARINGS.join(", ")
    );
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
    let mut samples = to_samples(&train, &mean, &std);
    println!("[PRETRAIN] {} cua so hoc, {} luot, lr = {}", samples.len(), EPOCHS, LEARNING_RATE);
    let head = train_head(&mut samples, true);

    // Ma trận nhầm lẫn theo từng ổ: mỗi ổ bị đoán thành lớp nào
    println!();
    let n = paderborn::TRAIN_BEARINGS.len();
    print_eval(&format!("Hoc: {n} o, N15, lan do 1-16"), &head, &train, &mean, &std);
    print_eval(&format!("Kiem tra: {n} o, N15, lan do 17-20"), &head, &val, &mean, &std);
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

    // Phổ bao theo từng ổ: vòng ngoài N15 phải có đỉnh ở 60-100 Hz (BPFO ~76 Hz), N09 ở 35-60 Hz (~46 Hz)
    let header: String = FEATURE_NAMES[ENV_FIRST..].iter().map(|f| format!("{:>10}", f.trim_start_matches("env_"))).collect();
    println!("
[ENV] Trung binh dac trung pho bao (chua chuan hoa, > 0 = cao hon muc trung binh) theo tung o:");
    println!("[ENV]   {:<17}{header}", "o");
    print_envelope_by_bearing("O hoc, N15, lan do 1-16", &train);
    print_envelope_by_bearing("O cau hinh, N15", &config_old);
    print_envelope_by_bearing("O cau hinh, N09", &config_new);

    // Kiểm tra chéo bỏ-một-ổ: học trên 7 ổ (chuẩn hóa cũng chỉ từ 7 ổ đó), chấm ổ thứ 8 chưa từng thấy
    println!("\n[LOBO] Kiem tra cheo bo-mot-o: hoc {} o (N15, lan do 1-16), cham o bi bo ra (N15, lan do 1-20)", paderborn::TRAIN_BEARINGS.len() - 1);
    let mut held_results = Vec::new();
    for &held in &paderborn::TRAIN_BEARINGS {
        let fold_train = || train.iter().filter(move |r| r.bearing != held);
        let (fold_mean, fold_std) = fit_normalization(fold_train());
        let mut fold_samples = to_samples(fold_train(), &fold_mean, &fold_std);
        let fold_head = train_head(&mut fold_samples, false);
        let held_out = train.iter().chain(val.iter()).filter(|r| r.bearing == held);
        for result in per_bearing(&fold_head, held_out, &fold_mean, &fold_std) {
            println!("[LOBO]     bo {}", result.describe());
            held_results.push(result);
        }
    }
    println!("[LOBO] BA bo-mot-o = {:.2}%", balanced_accuracy(&held_results));

    match write_pretrained(&head, &mean, &std) {
        Ok(path) => println!("\n[SAVE] Da ghi {}", path.display()),
        Err(e) => {
            eprintln!("[SAVE][ERROR] {e}");
            std::process::exit(1);
        }
    }
}
