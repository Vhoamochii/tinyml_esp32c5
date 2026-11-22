#![no_std]
#![no_main]

use esp_hal::uart::{Uart, Config as UartConfig};
use esp_hal::main;
use esp_println::println;

// Trọng số W1 (Shape: 2x8, flatten theo hàng)
const W1: [f32; 16] = [
    0.304341, -0.024730, 0.470560, 2.469548, 0.467082, -1.446087, 2.220913, 0.822179,
    -0.001364, 0.233778, -0.082806, -1.791276, 0.513829, -1.611619, -2.822321, -0.217625
];

// Bias B1 (Shape: 1x8)
const B1: [f32; 8] = [
    -0.275397, 0.086738, -0.152390, 0.782191, 0.676296, 0.061371, -1.023316, -0.089177
];

// Trọng số W2 (Shape: 8x1)
const W2: [f32; 8] = [
    -0.563520, 0.099843, -0.507717, -2.857987, 1.067262, -1.832937, 3.475092, -0.857385
];

// Bias B2 (Shape: 1x1)
const B2: [f32; 1] = [
    0.844314
];

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("[RUST][PANIC] {}", info);
    loop {}
}

// Hàm thực hiện Forward Pass
fn forward_pass(in1: f32, in2: f32) -> f32 {
    let mut a1 = [0.0f32; 8];
    
    // Lớp ẩn: Z1 = W1 * X + B1, A1 = tanh(Z1)
    for j in 0..8 {
        // W1 flatten: index = i * 8 + j (với i là 0 hoặc 1)
        let z1 = B1[j] + W1[0 * 8 + j] * in1 + W1[1 * 8 + j] * in2;
        // libm::tanhf dùng cho môi trường no_std không có sẵn hàm tanh
        a1[j] = libm::tanhf(z1);
    }
    
    // Lớp ra: Z2 = W2 * A1 + B2, A2 = tanh(Z2)
    let mut z2 = B2[0];
    for j in 0..8 {
        z2 += a1[j] * W2[j];
    }
    
    libm::tanhf(z2)
}

#[main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default());
    let config = UartConfig::default().with_baudrate(115200);
    let mut uart0 = Uart::new(peripherals.UART0, config).unwrap();

    println!("[RUST][BOOT] ready for XOR test");

    let mut rx_buf = [0u8; 2]; // Nhận 2 byte (in1, in2)

    loop {
        // Đọc đủ 2 byte
        match uart0.read(&mut rx_buf) {
            Ok(_) => {
                // Ép kiểu u8 sang f32 (giả định PC chỉ gửi 0.0 hoặc 1.0)
                let in1 = rx_buf[0] as f32;
                let in2 = rx_buf[1] as f32;
                
                // Tính toán
                let out = forward_pass(in1, in2);
                
                // Gửi trả lại 4 byte (f32 little-endian)
                let tx_bytes = out.to_le_bytes();
                let _ = uart0.write(&tx_bytes);
            }
            Err(_) => {
                // Lỗi UART
            }
        }
    }
}
