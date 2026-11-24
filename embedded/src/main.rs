#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{
    main,
    init,
    Config,
};
use esp_println::println;

mod xor_tanh;

#[main]
fn main() -> ! {
    // Khai bao App Descriptor de espflash va bootloader xac nhan firmware hop le
    esp_bootloader_esp_idf::esp_app_desc!();

    let _peripherals = init(Config::default());

    println!("\n[EMBEDDED][START] ESP32-C5 XOR Tanh Test");

    let result = xor_tanh::run_xor_tanh_test(|epoch, loss, correct, total| {
        if epoch % 100 == 0 || epoch == 1999 {
            println!("[EMBEDDED][EPOCH {}] loss={:.6} correct={}/{}", epoch, loss, correct, total);
        }
    });

    println!(
        "[EMBEDDED][END] correct={}/{} final_loss={:.6} outputs=[{:.4},{:.4},{:.4},{:.4}] passed={}",
        result.correct, result.total, result.final_loss,
        result.outputs[0], result.outputs[1], result.outputs[2], result.outputs[3],
        result.passed()
    );

    println!("[EMBEDDED][DONE] Nhan Reset de chay lai.");

    loop {}
}
