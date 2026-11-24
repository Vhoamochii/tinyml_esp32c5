// Điểm vào firmware ESP32-C5 (no_std, không heap).
// Hiện mới là bài tự kiểm tra: chạy một vòng P -> Q -> học -> R bằng tinyml-core (cùng mã với pc-sim)
// và in kết quả qua UART. Chưa nhận luồng dữ liệu từ PC.
#![no_std]
#![no_main]

use esp_hal::main;
use esp_println::println;
use tinyml_core::head::Head;
use tinyml_core::memsim::{slots_for_budget, MemoryPool, SlotState, SLOT_SIZE_BYTES};
use tinyml_core::policy::SimplePolicy;
use tinyml_core::NUM_FEATURES;

// Bắt buộc với esp-hal 1.x: espflash cần app descriptor để nạp firmware
esp_bootloader_esp_idf::esp_app_desc!();

const MEMORY_BUDGET_BYTES: usize = 2000;
const N: usize = slots_for_budget(MEMORY_BUDGET_BYTES);
const PENDING_TTL: u32 = 10;
const LEARNING_RATE: f32 = 0.1;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("[FW][PANIC] {}", info);
    loop {}
}

#[main]
fn main() -> ! {
    let _peripherals = esp_hal::init(esp_hal::Config::default());
    println!(
        "[FW][BOOT] Moi o {} byte, ngan sach {} byte -> {} o",
        SLOT_SIZE_BYTES, MEMORY_BUDGET_BYTES, N
    );

    let mut pool: MemoryPool<N> = MemoryPool::new();
    let mut policy = SimplePolicy::new(N / 2, PENDING_TTL, 43);
    let mut head = Head::zeros();

    // Tự kiểm tra: một mẫu cố định, nhãn lớp 1, đi hết vòng đời một ô
    let mut x = [0.0f32; NUM_FEATURES];
    x[1] = 1.0;
    let before = head.predict(&x);

    let result = pool
        .alloc_p(1, x, 0)
        .and_then(|_| pool.assign_label(1, 1, 1));
    match result {
        Ok(idx) => {
            for _ in 0..20 {
                head.sgd_step(&[(&x, 1)], LEARNING_RATE);
            }
            policy.after_first_training(&mut pool, idx, 2);
        }
        Err(e) => println!("{}", e.as_str()),
    }

    println!(
        "[FW][SELFTEST] Du doan truoc={} sau={} (mong doi 1) | P={} Q={} R={} | {}/{} bytes",
        before,
        head.predict(&x),
        pool.count(SlotState::P),
        pool.count(SlotState::Q),
        pool.count(SlotState::R),
        pool.usage_bytes(),
        pool.capacity_bytes()
    );

    loop {}
}
