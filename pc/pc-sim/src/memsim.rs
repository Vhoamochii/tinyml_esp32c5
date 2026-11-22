// Mô phỏng phân bổ bộ nhớ cho TinyML với nhãn đến trễ
// Các trạng thái: P (Chờ nhãn), Q (Chờ học), R (Học lại)

// Ước tính kích thước một mẫu (tính bằng byte)
// Giả định: ID (4 bytes), Features (16 * 4 bytes = 64 bytes), Label (1 byte), Timestamp (8 bytes)
// Tổng cộng: 77 bytes. Cộng thêm overhead căn chỉnh (alignment) -> làm tròn lên 80 bytes.
pub const SAMPLE_SIZE_BYTES: usize = 80; 

#[derive(Debug, Clone, PartialEq)]
pub enum SampleStatus {
    P, // Chờ nhãn
    Q, // Chờ học
    R, // Học lại (Replay)
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub id: u32,
    pub features: [f32; 16], // 16 đặc trưng như trong đề cương
    pub label: Option<u8>,
    pub status: SampleStatus,
    pub timestamp: u64,
}

impl Sample {
    pub fn new(id: u32, features: [f32; 16], timestamp: u64) -> Self {
        Self {
            id,
            features,
            label: None,
            status: SampleStatus::P,
            timestamp,
        }
    }
}

pub struct MemoryManager {
    pub p_queue: Vec<Sample>,
    pub q_queue: Vec<Sample>,
    pub r_queue: Vec<Sample>,
    pub max_memory_bytes: usize,
}

impl MemoryManager {
    pub fn new(max_memory_bytes: usize) -> Self {
        Self {
            p_queue: Vec::new(),
            q_queue: Vec::new(),
            r_queue: Vec::new(),
            max_memory_bytes,
        }
    }

    // Tính tổng bộ nhớ đang sử dụng
    pub fn current_usage_bytes(&self) -> usize {
        (self.p_queue.len() + self.q_queue.len() + self.r_queue.len()) * SAMPLE_SIZE_BYTES
    }

    // Thêm mẫu vào P (Chờ nhãn)
    pub fn add_to_p(&mut self, sample: Sample) -> Result<(), &'static str> {
        if self.current_usage_bytes() + SAMPLE_SIZE_BYTES > self.max_memory_bytes {
            return Err("[MEMSIM][ERROR] Khong du bo nho de them vao P");
        }
        self.p_queue.push(sample);
        Ok(())
    }

    // Nhãn đến, chuyển mẫu từ P sang Q
    pub fn assign_label(&mut self, id: u32, label: u8) -> Result<(), &'static str> {
        if let Some(pos) = self.p_queue.iter().position(|s| s.id == id) {
            let mut sample = self.p_queue.remove(pos);
            sample.label = Some(label);
            sample.status = SampleStatus::Q;
            self.q_queue.push(sample);
            Ok(())
        } else {
            Err("[MEMSIM][ERROR] Khong tim thay ID trong P")
        }
    }

    // Lấy mẫu từ Q để train, sau đó đưa vào R
    pub fn pop_for_training(&mut self) -> Option<Sample> {
        if !self.q_queue.is_empty() {
            let mut sample = self.q_queue.remove(0);
            sample.status = SampleStatus::R;
            self.r_queue.push(sample.clone());
            Some(sample)
        } else {
            None
        }
    }
}
