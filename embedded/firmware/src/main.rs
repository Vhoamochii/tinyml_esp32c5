// Điểm vào firmware ESP32-C5 (no_std, không heap).
// 1. Chạy hai phép thử bắt buộc của đề cương (XOR và ghép nhãn qua ranh giới epoch) bằng tinyml-core,
//    cùng mã với `cargo run --bin selftest` trên Ubuntu. Log in qua UART phải khớp từng dòng với Ubuntu.
// 2. Bước 5, đo thời gian trên bo (chưa có UART nhận dữ liệu): bộ trích xuất 16 đặc trưng trên tín hiệu
//    giả tự sinh, và một bước học đầy đủ với cấu hình đã khóa (K = 25, r = 0%, ưu tiên Q, lr = 0,2,
//    Case 3 reservoir là case nặng nhất vì phải băm ID khi thu hồi). Ghi log UART nằm ngoài khoảng đo.
#![no_std]
#![no_main]

use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::Instant;
use esp_println::println;
use tinyml_core::event::Event;
use tinyml_core::features::{normalize, WindowExtractor, BLOCKS_PER_WINDOW, BLOCK_LEN, SAMPLE_RATE_HZ};
use tinyml_core::learner::{Config, Learner, MANAGEMENT_BYTES};
use tinyml_core::memsim::{slots_for_budget, SLOT_SIZE_BYTES};
use tinyml_core::policy::Retention;
use tinyml_core::pretrained;
use tinyml_core::rng::XorShift32;
use tinyml_core::schedule::Schedule;
use tinyml_core::{selftest, NUM_CLASSES, NUM_FEATURES};

// Bắt buộc với esp-hal 1.x: espflash cần app descriptor để nạp firmware
esp_bootloader_esp_idf::esp_app_desc!();

const MEMORY_BYTES: usize = 2000; // M, giống pc-sim
/// Cấu hình đã khóa trên tập chọn cấu hình (pc-sim): K = 25 ô
const LOCKED_K: usize = 25;
/// Số cửa sổ dùng để đo bộ trích xuất (mỗi cửa sổ 16 khối)
const BENCH_WINDOWS: usize = 4;
/// Số bước học dùng để đo
const BENCH_STEPS: usize = 200;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("[FW][PANIC] {}", info);
    loop {}
}

#[main]
fn main() -> ! {
    // Chạy CPU ở xung nhịp cao nhất; đề cương yêu cầu ghi xung nhịp bo kèm số đo thời gian
    let _peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    println!(
        "[FW][BOOT] s = {} byte/o, M = {} byte, H = {} byte -> K = {} o | CPU = CpuClock::max()",
        SLOT_SIZE_BYTES,
        MEMORY_BYTES,
        MANAGEMENT_BYTES,
        slots_for_budget(MEMORY_BYTES, MANAGEMENT_BYTES)
    );

    let mut out = |a: core::fmt::Arguments<'_>| println!("{}", a);
    let xor_ok = selftest::run_xor(&mut out);
    let epoch_ok = selftest::run_epoch_matching(&mut out);
    println!(
        "[SELFTEST] XOR={} EPOCH={}",
        if xor_ok { "PASS" } else { "FAIL" },
        if epoch_ok { "PASS" } else { "FAIL" }
    );

    println!(
        "[BENCH][RAM] WindowExtractor = {} byte | Learner<{}, {}> = {} byte | pretrain co san = {}",
        core::mem::size_of::<WindowExtractor>(),
        LOCKED_K,
        NUM_CLASSES,
        core::mem::size_of::<Learner<LOCKED_K, NUM_CLASSES>>(),
        pretrained::AVAILABLE
    );
    let z = bench_extractor();
    bench_learner(z);
    println!("[BENCH] XONG");

    loop {}
}

/// Số µs từ `t0` đến giờ
fn micros_since(t0: Instant) -> u32 {
    t0.elapsed().as_micros() as u32
}

/// (trung vị, P95 theo hạng gần nhất, lớn nhất) của các số đo µs; sắp xếp tại chỗ
fn stats(v: &mut [u32]) -> (u32, u32, u32) {
    v.sort_unstable();
    let p95 = v[(v.len() * 95).div_ceil(100) - 1];
    (v[v.len() / 2], p95, v[v.len() - 1])
}

/// Tín hiệu giả để đo thời gian: nhiễu trắng + tiếng gõ ~76 Hz tắt dần.
/// Thời gian tính gần như không phụ thuộc giá trị mẫu, chỉ cần tín hiệu có biên độ thực tế.
struct FakeSignal {
    rng: XorShift32,
    n: u32,
    amp: f32,
}

impl FakeSignal {
    fn fill(&mut self, block: &mut [f32; BLOCK_LEN]) {
        for v in block.iter_mut() {
            if self.n % 838 == 0 {
                self.amp = 2.0; // 64000 / 838 ≈ 76,4 cú gõ mỗi giây
            }
            self.amp *= 0.97;
            let ring = if self.n % 2 == 0 { self.amp } else { -self.amp };
            *v = 0.1 * self.rng.range_f32(-1.0, 1.0) + ring;
            self.n += 1;
        }
    }
}

/// Đo bộ trích xuất theo từng khối như khi nhận luồng thật. Hạn: mỗi khối 1024 mẫu đến cách nhau 16 ms,
/// mỗi cửa sổ 16 khối cách nhau 256 ms (đề cương: T = bước trượt / 64000 giây). Trả về z của cửa sổ cuối.
fn bench_extractor() -> [f32; NUM_FEATURES] {
    let mut extractor = WindowExtractor::new();
    let mut signal = FakeSignal { rng: XorShift32::new(1), n: 0, amp: 0.0 };
    let mut block = [0.0f32; BLOCK_LEN];
    let mut normal = [0u32; BENCH_WINDOWS * (BLOCKS_PER_WINDOW - 1)];
    let mut last = [0u32; BENCH_WINDOWS];
    let mut window_total = [0u32; BENCH_WINDOWS];
    let mut z = [0.0f32; NUM_FEATURES];

    for w in 0..BENCH_WINDOWS {
        for b in 0..BLOCKS_PER_WINDOW {
            signal.fill(&mut block);
            let t0 = Instant::now();
            let result = extractor.push_block(&block);
            let dt = micros_since(t0);
            window_total[w] += dt;
            match result {
                Some(features) => {
                    last[w] = dt;
                    z = features;
                }
                None => normal[w * (BLOCKS_PER_WINDOW - 1) + b] = dt,
            }
        }
    }

    let block_budget = (BLOCK_LEN as f32 / SAMPLE_RATE_HZ * 1e6) as u32;
    let window_budget = block_budget * BLOCKS_PER_WINDOW as u32;
    let (med, p95, max) = stats(&mut normal);
    println!(
        "[BENCH][EXTRACT] khoi thuong ({} khoi): trung vi {} us | P95 {} us | max {} us | han moi khoi {} us",
        normal.len(),
        med,
        p95,
        max,
        block_budget
    );
    let (med, _, max) = stats(&mut last);
    println!("[BENCH][EXTRACT] khoi cuoi (them FFT pho bao): trung vi {} us | max {} us", med, max);
    let (med, _, max) = stats(&mut window_total);
    println!(
        "[BENCH][EXTRACT] ca cua so ({} khoi): trung vi {} us | max {} us | han {} us -> {}",
        BLOCKS_PER_WINDOW,
        med,
        max,
        window_budget,
        if max <= window_budget { "KIP" } else { "KHONG KIP" }
    );
    println!("[BENCH][EXTRACT] z = {:?}", z);
    z
}

/// Đo một bước học đầy đủ: chuẩn hóa + on_window (dự đoán, cấp ô) + on_label + train(2 lượt).
/// Nhãn trễ 4 bước nên sau ~25 bước mảng đầy và mỗi bước đều phải thu hồi ô (reservoir băm ID).
fn bench_learner(z: [f32; NUM_FEATURES]) {
    let cfg = Config {
        r_protected: 0,
        q_max: 9,
        schedule: Schedule::QPriority,
        replay_enabled: true,
        lr: 0.2,
        seed: 11,
        retention: Retention::Reservoir,
    };
    let mut learner: Learner<LOCKED_K, NUM_CLASSES> = Learner::new(&cfg);
    learner.head = pretrained::head();
    let mut ignore = |_e: Event| {};
    let mut times = [0u32; BENCH_STEPS];

    for step in 0..BENCH_STEPS as u32 {
        let mut features = z;
        features[0] += step as f32 * 0.001; // mỗi bước một vectơ hơi khác
        let t0 = Instant::now();
        normalize(&mut features, &pretrained::FEATURE_MEAN, &pretrained::FEATURE_STD);
        learner.on_window(step, step, features, &mut ignore);
        if step >= 4 {
            learner.on_label(step, step - 4, ((step - 4) % NUM_CLASSES as u32) as u8, &mut ignore);
        }
        learner.train(step, 2, &mut ignore);
        times[step as usize] = micros_since(t0);
    }

    let (med, p95, max) = stats(&mut times);
    println!(
        "[BENCH][LEARN] 1 buoc (chuan hoa + du doan + cap o + nhan + 2 luot hoc), {} buoc: trung vi {} us | P95 {} us | max {} us",
        BENCH_STEPS, med, p95, max
    );
}
