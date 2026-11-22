# SPECS — Phân bổ bộ nhớ khi nhãn đến trễ trên ESP32-C5

## Trạng thái hiện tại (Cập nhật: 22/11/2026)
- Đã hoàn thành Test Case 01 (Rust + Python giao tiếp UART cơ bản).
- Đã hoàn thành Test Case 02 (Rust tính toán XOR 2-8-1 dùng Tanh, Python huấn luyện).
- Firmware Rust đã build thành công, chưa nạp xuống board thực tế.

## Stack
- **Firmware:** Rust (no_std), esp-hal 1.2.0, esp-println 0.18.0 (UART log), libm (toán học FP32).
- **PC:** Python 3.10+, NumPy (huấn luyện), PySerial (giao tiếp).
- **Phần cứng:** ESP32-C5 (RISC-V 240 MHz).

## Giao thức UART (Test 02)
- Baudrate: 115200.
- PC gửi xuống: 2 bytes (giá trị 0 hoặc 1).
- Firmware trả về: 4 bytes (số thực f32 little-endian, kết quả forward pass).

## Cách chạy
1. Build firmware: `cd embedded/firmware && cargo build --release`
2. Flash: `espflash flash --monitor --port /dev/ttyACM0 target/riscv32imac-unknown-none-elf/release/esp32c5_firmware`
3. Chạy test: `cd pc && conda activate tinyml_pc && python test_02_xor.py --port /dev/ttyACM0`
