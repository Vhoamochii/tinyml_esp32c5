// Lõi dùng chung cho bộ mô phỏng PC (pc/pc-sim) và firmware ESP32-C5 (embedded/firmware),
// theo "Đề cương phương pháp nghiên cứu" (src/De_cuong_phuong_phap_Rust_ESP32-C5.docx).
// no_std, không dùng heap: vùng nhớ có dung lượng cố định lúc biên dịch.
// Ubuntu và bo chạy cùng một mã này nên phải cho cùng ID, nhãn và thứ tự học trên cùng luồng.
#![no_std]

pub mod event;
pub mod features;
pub mod head;
pub mod learner;
pub mod memsim;
pub mod policy;
pub mod pretrained;
pub mod rng;
pub mod schedule;
pub mod selftest;

/// Số đặc trưng bộ trích xuất tạo ra cho mỗi cửa sổ (vectơ z)
pub const NUM_FEATURES: usize = 16;
/// Ba lớp chẩn đoán: ổ khỏe, lỗi vòng trong, lỗi vòng ngoài
pub const NUM_CLASSES: usize = 3;
