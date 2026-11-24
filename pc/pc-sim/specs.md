# SPECS: PC Simulator cho TinyML với Nhãn Trễ (ESP32-C5)

## 1. Kiến trúc tổng thể
Dự án mô phỏng chính sách phân bổ bộ nhớ động (P/Q/R) cho bài toán Continual Learning với nhãn đến trễ. 
Hệ thống được thiết kế để chạy trên PC (bằng Rust) nhằm đánh giá thuật toán trước khi triển khai xuống vi điều khiển ESP32-C5.

- **P (Pending):** Chứa các mẫu tín hiệu đã dự đoán nhưng chưa có nhãn.
- **Q (Queue):** Chứa các mẫu đã có nhãn, chờ được đưa vào huấn luyện (SGD).
- **R (Replay):** Chứa các mẫu cũ đã học để ôn lại, tránh hiện tượng quên lãng (Catastrophic Forgetting).

## 2. Tech Stack
- **Ngôn ngữ:** Rust (Edition 2021).
- **Lõi dùng chung:** crate `tinyml-core` (`../../tinyml-core`, `no_std`, không heap), cùng mã với firmware. Không dùng thư viện mạng nơ-ron ngoài.
- **Thư viện Benchmark:** criterion (dev-dependency, sẽ dùng sau).
- **Môi trường:** Ubuntu + Vim.

## 3. Các quyết định thiết kế quan trọng
- **Chia vai:** `pc-sim` chỉ là phía PC (sinh dữ liệu, giữ nhãn đúng, hẹn giờ trả nhãn, chấm điểm). Phía thiết bị (vùng nhớ, chính sách, lớp cuối) nằm trong `tinyml-core` để PC và bo chạy cùng một mã.
- **Vùng nhớ (`tinyml-core/src/memsim.rs`):** N ô cố định, mỗi mẫu chiếm một ô, trạng thái ô đổi tại chỗ Trống → P → Q → R → Trống. Kích thước ô lấy bằng `size_of` (hiện 80 byte); 2000 byte → 25 ô.
- **Chính sách (`tinyml-core/src/policy.rs`):** tạm thời (TTL cho P, R tối đa nửa số ô + reservoir sampling). Đối chứng và chính sách thích ứng của báo cáo sẽ là các module cùng vai trò.
- **Lớp cuối (`tinyml-core/src/head.rs`):** chỉ học lớp cuối 16 → 3 (softmax + cross-entropy, SGD trên mẫu mới + replay); bộ trích xuất giữ cố định.
- **Kiểm thử:** `tinyml-core/tests/core_test.rs` (happy path, overflow, invalid ID, giới hạn R, TTL, lớp cuối, bộ sinh ngẫu nhiên).

## 4. Cách cài đặt & chạy dự án
- Chạy mô phỏng: `cd pc/pc-sim && cargo run --release`
- Chạy kiểm thử lõi: `cd tinyml-core && cargo test`
