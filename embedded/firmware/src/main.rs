// Điểm vào firmware ESP32-C5 (no_std, không heap).
// Chạy hai phép thử bắt buộc của đề cương (XOR và ghép nhãn qua ranh giới epoch) bằng tinyml-core,
// cùng mã với `cargo run --bin selftest` trên Ubuntu. Log in qua UART phải khớp từng dòng với Ubuntu.
// Chưa nhận luồng cửa sổ tín hiệu từ máy tính.
#![no_std]
#![no_main]

use esp_hal::main;
use esp_println::println;
use tinyml_core::learner::MANAGEMENT_BYTES;
use tinyml_core::memsim::{slots_for_budget, SLOT_SIZE_BYTES};
use tinyml_core::selftest;

// Bắt buộc với esp-hal 1.x: espflash cần app descriptor để nạp firmware
esp_bootloader_esp_idf::esp_app_desc!();

const MEMORY_BYTES: usize = 2000; // M, giống pc-sim

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("[FW][PANIC] {}", info);
    loop {}
}

#[main]
fn main() -> ! {
    let _peripherals = esp_hal::init(esp_hal::Config::default());
    println!(
        "[FW][BOOT] s = {} byte/o, M = {} byte, H = {} byte -> K = {} o",
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

    loop {}
}
