//! Fixture epoch 10 ID de kiem tra ghep nhan tre qua ranh gioi epoch.
//! Theo huong dan thuc nghiem Bang 3: 10 mau, moi nhan tre 2 buoc.
//!
//! Bang 3 (Bang 3 trong huong dan):
//! Buoc | Doan phat | ID du doan | ID nhan nhan | ID hoc lan dau
//! 0    | Epoch 1   | 0          | --           | --
//! 1    | Epoch 1   | 1          | --           | --
//! 2    | Epoch 1   | 2          | 0            | 0
//! ...
//! 10   | Tra not   | --         | 8            | 8
//! 11   | Tra not   | --         | 9            | 9

/// 10 dac trung INT8 16 chieu, moi dac trung la mot hang don gian.
/// Dung de kiem tra MemPool khong phu thuoc vao noi dung dac trung.
pub const EPOCH_FEATURES: [[i8; 16]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  // ID 0
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // ID 1
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  // ID 2
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  // ID 3
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  // ID 4
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  // ID 5
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  // ID 6
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],  // ID 7
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],  // ID 8
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],  // ID 9
];

/// Lich tra nhan: moi buoc tra nhan cho ID nao (None = khong tra).
/// Theo Bang 3: buoc 2 tra ID 0, buoc 3 tra ID 1, ..., buoc 11 tra ID 9.
pub const LABEL_SCHEDULE: [Option<u32>; 12] = [
    None,    // buoc 0: chi predict ID 0
    None,    // buoc 1: chi predict ID 1
    Some(0), // buoc 2: predict ID 2, tra nhan ID 0
    Some(1), // buoc 3: predict ID 3, tra nhan ID 1
    Some(2), // buoc 4: predict ID 4, tra nhan ID 2
    Some(3), // buoc 5: predict ID 5, tra nhan ID 3 (Epoch 2 bat dau)
    Some(4), // buoc 6: predict ID 6, tra nhan ID 4
    Some(5), // buoc 7: predict ID 7, tra nhan ID 5
    Some(6), // buoc 8: predict ID 8, tra nhan ID 6
    Some(7), // buoc 9: predict ID 9, tra nhan ID 7
    Some(8), // buoc 10: tra nhan ID 8 (khong predict)
    Some(9), // buoc 11: tra nhan ID 9 (khong predict)
];

/// So buoc co predict (0..9)
pub const PREDICT_STEPS: usize = 10;
/// Tong so buoc (ke ca 2 buoc tra not)
pub const TOTAL_STEPS: usize = 12;

/// Nhan lop cho 10 mau epoch (0=khoe, 1=vong trong, 2=vong ngoai).
/// PC va bo doc cung bang nay de log doi chieu duoc y het.
pub const EPOCH_LABELS: [u8; 10] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0];
