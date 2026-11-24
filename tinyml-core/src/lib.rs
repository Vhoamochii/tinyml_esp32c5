// Lõi dùng chung cho bộ mô phỏng PC (pc/pc-sim) và firmware ESP32-C5 (embedded/firmware).
// no_std, không dùng heap: mọi vùng nhớ có dung lượng cố định lúc biên dịch (YC2 trong báo cáo).
// PC và bo chạy cùng một mã này nên cho cùng kết quả trên cùng luồng dữ liệu (YC6).
#![no_std]

pub mod head;
pub mod memsim;
pub mod policy;
pub mod rng;

/// Số đặc trưng do bộ trích xuất INT8 tạo ra cho mỗi cửa sổ
pub const NUM_FEATURES: usize = 16;
/// Ba lớp: ổ khỏe, lỗi vòng trong, lỗi vòng ngoài
pub const NUM_CLASSES: usize = 3;
