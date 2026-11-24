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
Theo `src/De_cuong_phuong_phap_Rust_ESP32-C5.docx` (đề cương phương pháp mới).
- **Chia vai:** `pc-sim` chỉ là phía máy tính (sinh luồng, giữ nhãn thật, trả nhãn theo lịch, chấm điểm). Phía thiết bị nằm trong `tinyml-core` để Ubuntu và bo chạy cùng một mã.
- **Mảng ô (`memsim.rs`):** K = ⌊(M − H)/s⌋ ô cố định; s = `size_of` của ô (hiện 76 byte), H = byte quản lý ngoài mảng. Mỗi mẫu chiếm một ô, trạng thái đổi tại chỗ Trống → P → Q → R → Trống.
- **Chính sách bộ nhớ (`policy.rs`):** dùng ô trống trước; mảng đầy thì thu hồi R vào sớm nhất khi R có hơn r mẫu, sau đó P chờ lâu nhất; Q được bảo vệ và có giới hạn; mọi lần bỏ mẫu đều ghi ID và lý do.
- **Lịch học (`schedule.rs`):** xen kẽ hoặc ưu tiên Q; mỗi lần học đầu cho một lượt học lại (tối đa K); chọn đều trong R, bỏ qua ID vừa học.
- **Lớp cuối (`head.rs`):** Wz + b, softmax, entropy chéo, một bước SGD FP32 trên một mẫu; số lớp là tham số (3 cho ổ lăn, 2 cho XOR).
- **Nhật ký (`event.rs`):** mỗi dự đoán, ghép nhãn, bỏ mẫu và lượt học là một dòng `[EVT]`, cùng định dạng trên Ubuntu và bo.
- **Phép thử bắt buộc (`selftest.rs`):** XOR (điểm cố định, một bước SGD, 200 lượt) và ghép nhãn qua ranh giới epoch (10 ID, nhãn trễ 2 bước).
- **Thí nghiệm (`pc-sim/src/main.rs`):** dữ liệu GIẢ LẬP hai tốc độ; r = 0/25/50% K và đối chứng tắt học lại; lịch trả nhãn rải/theo đợt; lịch học xen kẽ/ưu tiên Q; 5 hạt giống.

## 4. Cách cài đặt & chạy dự án
- Kiểm thử lõi: `cd tinyml-core && cargo test`
- Hai phép thử bắt buộc trên Ubuntu: `cd pc/pc-sim && cargo run --release --bin selftest`
- Thí nghiệm giả lập: `cd pc/pc-sim && cargo run --release --bin pc-sim`
- Trên bo: `cd embedded/firmware && cargo run --release` (in cùng các dòng `[XOR]`, `[EVT]`, `[EPOCH]`)
