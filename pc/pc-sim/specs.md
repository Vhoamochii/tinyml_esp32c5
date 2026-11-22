# SPECS: PC Simulator cho TinyML với Nhãn Trễ (ESP32-C5)

## 1. Kiến trúc tổng thể
Dự án mô phỏng chính sách phân bổ bộ nhớ động (P/Q/R) cho bài toán Continual Learning với nhãn đến trễ. 
Hệ thống được thiết kế để chạy trên PC (bằng Rust) nhằm đánh giá thuật toán trước khi triển khai xuống vi điều khiển ESP32-C5.

- **P (Pending):** Chứa các mẫu tín hiệu đã dự đoán nhưng chưa có nhãn.
- **Q (Queue):** Chứa các mẫu đã có nhãn, chờ được đưa vào huấn luyện (SGD).
- **R (Replay):** Chứa các mẫu cũ đã học để ôn lại, tránh hiện tượng quên lãng (Catastrophic Forgetting).

## 2. Tech Stack
- **Ngôn ngữ:** Rust (Edition 2021).
- **Thư viện Neural Network:** ruv-fann (Phiên bản 0.2.3).
- **Thư viện Benchmark:** criterion (đã cấu hình, sẽ dùng sau).
- **Môi trường:** Ubuntu + Vim.

## 3. Các quyết định thiết kế quan trọng
- **Quản lý bộ nhớ (memsim.rs):** Ước tính 1 mẫu = 80 bytes (bao gồm ID, 16 features, label, status, timestamp). Tổng bộ nhớ được giới hạn (ví dụ: 2000 bytes) để mô phỏng RAM hạn chế của vi điều khiển.
- **Mạng Nơ-ron (main.rs):** Sử dụng cấu trúc 16 Input - 8 Hidden - 3 Output. Hàm kích hoạt `Tanh` cho lớp ẩn và `Sigmoid` cho lớp Output để ổn định quá trình lan truyền ngược.
- **Kiểm thử độc lập (tests/memsim_test.rs):** Tuân thủ nguyên tắc tách biệt logic quản lý bộ nhớ và kiểm thử các edge case (Happy path, Overflow, Invalid ID) trước khi tích hợp.

## 4. Cách cài đặt & chạy dự án
Di chuyển vào thư mục `pc/pc-sim` và chạy các lệnh sau:
- Build dự án: `cargo build`
- Chạy mô phỏng luồng dữ liệu: `cargo run`
- Chạy kiểm thử tự động: `cargo test`
