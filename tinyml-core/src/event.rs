// Nhật ký theo ID (đề cương, mục "Hồ sơ kết quả"): mọi lần dự đoán, ghép nhãn, bỏ mẫu và học
// đều phát ra một sự kiện. Ubuntu và bo in cùng định dạng nên có thể so từng dòng.

use core::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Predicted(u8),  // Lớp dự đoán, ghi trước khi dùng nhãn của mẫu
    Admitted,       // Mẫu được cấp ô P
    EvictedR,       // Thu hồi ô R vào sớm nhất để nhường chỗ (R đang có hơn r mẫu)
    EvictedP,       // Thu hồi ô P chờ lâu nhất để nhường chỗ; nhãn về sau sẽ ghép trượt
    DroppedNoSlot,  // Mẫu mới không có ô: mọi ô đều là Q hoặc R được bảo vệ
    LabelMatched,   // Nhãn khớp ID trong P
    DroppedQFull,   // Nhãn đã khớp nhưng Q đầy: bỏ mẫu trước khi học
    LabelMissed,    // Nhãn về nhưng ID không còn trong bộ nhớ (ghép trượt)
    LabelDuplicate, // Nhãn trùng cho ID đang ở Q hoặc R: không tạo lần học đầu thứ hai
    FirstLearn,     // Học lần đầu (Q -> R)
    Replay,         // Học lại một mẫu ở R
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub step: u32,
    pub id: u32,
    pub kind: EventKind,
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[EVT] step={} id={} ", self.step, self.id)?;
        match self.kind {
            EventKind::Predicted(c) => write!(f, "PRED {}", c),
            EventKind::Admitted => f.write_str("ADMIT_P"),
            EventKind::EvictedR => f.write_str("EVICT_R"),
            EventKind::EvictedP => f.write_str("EVICT_P"),
            EventKind::DroppedNoSlot => f.write_str("DROP_NO_SLOT"),
            EventKind::LabelMatched => f.write_str("LABEL_MATCH"),
            EventKind::DroppedQFull => f.write_str("DROP_Q_FULL"),
            EventKind::LabelMissed => f.write_str("LABEL_MISS"),
            EventKind::LabelDuplicate => f.write_str("LABEL_DUP"),
            EventKind::FirstLearn => f.write_str("LEARN_FIRST"),
            EventKind::Replay => f.write_str("REPLAY"),
        }
    }
}
