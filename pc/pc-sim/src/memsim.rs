// Mô phỏng phân bổ bộ nhớ cho TinyML với nhãn đến trễ
// Các trạng thái: P (Chờ nhãn), Q (Chờ học), R (Học lại)

use std::collections::VecDeque;

// Ước tính kích thước một mẫu (tính bằng byte)
// ID (4) + Features (16 * 4 = 64) + Label Option<u8> (2) + Status (1) + Timestamp (8) = 79 bytes.
// Căn chỉnh theo u64 (8 bytes) -> 80 bytes.
pub const SAMPLE_SIZE_BYTES: usize = 80;

// Kiểm tra lúc biên dịch: nếu sửa struct Sample thì phải cập nhật SAMPLE_SIZE_BYTES
const _: () = assert!(std::mem::size_of::<Sample>() == SAMPLE_SIZE_BYTES);

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
    pub q_queue: VecDeque<Sample>,
    pub r_queue: Vec<Sample>,
    pub max_memory_bytes: usize,
    pub r_capacity: usize, // Số mẫu tối đa được giữ trong R
    pub r_seen: u64,       // Tổng số mẫu đã học (dùng cho reservoir sampling)
    rng_state: u32,        // Trạng thái bộ sinh số ngẫu nhiên xorshift32 (không cần thư viện, port được xuống MCU)
}

impl MemoryManager {
    // Mặc định: R được dùng tối đa một nửa ngân sách, phần còn lại dành cho P và Q
    pub fn new(max_memory_bytes: usize) -> Self {
        Self::with_r_capacity(max_memory_bytes, max_memory_bytes / SAMPLE_SIZE_BYTES / 2)
    }

    pub fn with_r_capacity(max_memory_bytes: usize, r_capacity: usize) -> Self {
        Self {
            p_queue: Vec::new(),
            q_queue: VecDeque::new(),
            r_queue: Vec::new(),
            max_memory_bytes,
            r_capacity,
            r_seen: 0,
            rng_state: 0x9E37_79B9,
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
            self.q_queue.push_back(sample);
            Ok(())
        } else {
            Err("[MEMSIM][ERROR] Khong tim thay ID trong P")
        }
    }

    // Loại các mẫu chờ nhãn quá lâu (nhãn có thể không bao giờ đến). Trả về số mẫu bị loại.
    pub fn expire_pending(&mut self, now: u64, ttl: u64) -> usize {
        let before = self.p_queue.len();
        self.p_queue.retain(|s| now.saturating_sub(s.timestamp) <= ttl);
        before - self.p_queue.len()
    }

    // Lấy mẫu từ Q (FIFO) để train, sau đó đưa vào R
    pub fn pop_for_training(&mut self) -> Option<Sample> {
        let mut sample = self.q_queue.pop_front()?;
        sample.status = SampleStatus::R;
        self.store_in_replay(sample.clone());
        Some(sample)
    }

    // Reservoir sampling (Algorithm R): R không vượt quá r_capacity,
    // và mỗi mẫu đã học đều có cùng xác suất r_capacity / r_seen được giữ lại
    fn store_in_replay(&mut self, sample: Sample) {
        self.r_seen += 1;
        if self.r_queue.len() < self.r_capacity {
            self.r_queue.push(sample);
        } else if self.r_capacity > 0 {
            let j = (self.next_rand() as u64 % self.r_seen) as usize;
            if j < self.r_capacity {
                self.r_queue[j] = sample;
            }
        }
    }

    // xorshift32
    fn next_rand(&mut self) -> u32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng_state = x;
        x
    }
}
