#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_println::println;
use fw_esp32c5::algo::{LinearHead, Quant};

esp_bootloader_esp_idf::esp_app_desc!();

const FIXTURE: &[(u8, [i8; 16])] = &[
    (0, [-34,-56,-57,-54,-53,7,-25,-33,-26,-15,-43,-34,0,-31,19,-7]),
    (0, [-41,-50,-50,-48,-48,-2,-24,-28,-22,-16,-44,-31,0,-28,15,-1]),
    (0, [-35,-77,-78,-54,-53,-2,18,13,-25,-16,-52,-38,0,11,127,-30]),
    (0, [-32,-73,-74,-43,-43,-15,70,42,-26,-16,-53,-39,0,49,127,-30]),
    (1, [31,-38,-38,-46,-45,-4,-28,-37,-6,-14,-47,-20,1,-35,-8,13]),
    (1, [26,-31,-31,-36,-36,11,-26,-24,-11,-16,-43,-10,1,-25,-2,17]),
    (1, [23,-72,-71,-45,-46,-70,30,24,-13,-16,-53,-38,1,15,127,-24]),
    (1, [25,-63,-62,-38,-35,-65,61,22,-12,-15,-53,-36,1,21,105,-21]),
    (2, [50,127,127,127,127,-69,32,16,123,112,108,85,1,24,-42,-127]),
    (2, [22,127,127,127,127,35,13,22,36,22,58,127,1,21,-39,-127]),
    (2, [33,-32,-32,-38,-39,-14,-19,-27,39,18,-37,-25,1,-24,-3,11]),
    (2, [29,-17,-17,-32,-34,-31,-22,-30,75,38,-30,-23,1,-28,-16,23]),
];

const LR: f32 = 0.1;

// Tăng delay lên 1 triệu vòng lặp (~4ms) để UART gửi xong
fn uart_gap() {
    for _ in 0..1_000_000 {
        core::hint::spin_loop();
    }
}

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    println!("BOARD_START");
    uart_gap();

    let q = Quant::identity();
    let mut head = LinearHead::<3>::zeros();

    for (step, &(label, z)) in FIXTURE.iter().enumerate() {
        let pred = head.predict(z, &q);
        println!("S {} L {} P {}", step, label, pred);
        uart_gap();

        head.sgd_step(z, &q, label, LR);

        // In từng giá trị trên một dòng riêng để tránh dính
        println!("W00 {}", head.w[0][0]);
        uart_gap();
        println!("W0_15 {}", head.w[0][15]);
        uart_gap();
        println!("W10 {}", head.w[1][0]);
        uart_gap();
        println!("W1_15 {}", head.w[1][15]);
        uart_gap();
        println!("W20 {}", head.w[2][0]);
        uart_gap();
        println!("W2_15 {}", head.w[2][15]);
        uart_gap();
        println!("b0 {}", head.b[0]);
        uart_gap();
        println!("b1 {}", head.b[1]);
        uart_gap();
        println!("b2 {}", head.b[2]);
        uart_gap();
    }

    println!("BOARD_DONE");
    loop {}
}
