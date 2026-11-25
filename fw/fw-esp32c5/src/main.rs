#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{
    prelude::*,
    uart::{config::Config, Uart},
};
use esp_println::println;

mod algo;

#[entry]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default());

    // Cấu hình UART để in log
    let config = Config::default().with_baudrate(115200);
    let mut uart = Uart::new(peripherals.UART0, config).unwrap();
    let (mut tx, mut rx) = uart.split();

    println!("=== [B1-B3] BẮT ĐẦU TEST XOR ===");
    
    let mut net = algo::xor::XorNet::new();
    
    // B1: Test dự đoán (giả sử trọng số đã được set thủ công để test)
    net.w1[0] = 1.0; net.w1[1] = 1.0; net.b1 = -1.5;
    net.w0[0] = 1.0; net.b0 = 0.0;
    
    let inputs = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)];
    let mut pass_count = 0;
    
    for (x1, x2, y_true) in inputs {
        let y_pred = net.predict(x1, x2);
        let is_pass = (y_pred > 0.5) == (y_true > 0.5);
        if is_pass { pass_count += 1; }
        println!("Input: ({}, {}) | True: {} | Pred: {:.4} | {}", x1, x2, y_true, y_pred, if is_pass { "PASS" } else { "FAIL" });
    }
    println!("[B1] Kết quả: {}/4 PASS", pass_count);

    // B2: Test cập nhật SGD (1 bước)
    let mut net_sgd = algo::xor::XorNet::new();
    net_sgd.train_step(1.0, 0.0, 1.0);
    println!("[B2] Sau 1 bước SGD (1,0)->1: w1[0]={:.4}, b1={:.4}, w0[0]={:.4}, b0={:.4}", net_sgd.w1[0], net_sgd.b1, net_sgd.w0[0], net_sgd.b0);

    // B5: Kiểm tra kích thước Pool
    let pool = algo::pool::Pool::new(320, 32 * 1024); // Giả sử 320KB RAM, trừ 32KB heap
    println!("[B5] Kích thước 1 ô (s): {} bytes", algo::pool::PoolItem::SIZE);
    println!("[B5] Số ô tối đa (K): {}", pool.k);

    println!("=== HOÀN TẤT TEST ===");
    loop {}
}
