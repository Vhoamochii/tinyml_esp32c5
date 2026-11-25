use std::collections::VecDeque;

#[derive(Clone, Debug)]
struct Item {
    id: usize,
    true_label: Option<u8>,
    pred_label: Option<u8>,
}

struct PQRScheduler {
    p_queue: VecDeque<Item>, // Pending
    q_queue: Vec<Item>,      // Ready
    r_queue: Vec<Item>,      // Replay
    q_max_size: usize,
    learned_ids: Vec<usize>,
}

impl PQRScheduler {
    fn new(q_max_size: usize) -> Self {
        Self {
            p_queue: VecDeque::new(),
            q_queue: Vec::new(),
            r_queue: Vec::new(),
            q_max_size,
            learned_ids: Vec::new(),
        }
    }

    fn receive_data(&mut self, id: usize) {
        let item = Item {
            id,
            true_label: None,
            pred_label: Some(0),
        };
        println!("[PREDICT] ID: {} | Dự đoán: {:?} (Ghi log trước khi học)", item.id, item.pred_label.unwrap());
        self.p_queue.push_back(item);
    }

    fn receive_label(&mut self, id: usize, label: u8, is_shuffled: bool) {
        if self.learned_ids.contains(&id) {
            println!("[REJECT] ID: {} | Lý do: Nhãn trùng (đã học)", id);
            return;
        }

        // Nếu Q đầy, từ chối nhãn (để test edge case Q đầy)
        if self.q_queue.len() >= self.q_max_size {
            println!("[REJECT] ID: {} | Lý do: Q đầy (Max: {}) - Giữ lại trong P", id, self.q_max_size);
            return; 
        }

        if let Some(pos) = self.p_queue.iter().position(|x| x.id == id) {
            let mut item = self.p_queue.remove(pos).unwrap();
            item.true_label = Some(label);
            self.q_queue.push(item);
            let suffix = if is_shuffled { " (Nhãn xáo trộn)" } else { "" };
            println!("[MATCH] Ghép nhãn ID: {} -> Lớp {}{}. Chuyển sang Q.", id, label, suffix);
        } else {
            println!("[REJECT] ID: {} | Lý do: Không tìm thấy trong P", id);
        }
    }

    fn flush_epoch(&mut self) {
        if self.q_queue.is_empty() {
            return;
        }
        
        println!("--- [EPOCH] Ranh giới! Bắt đầu học Q ({} mục) ---", self.q_queue.len());
        
        // Sắp xếp Q theo ID để đảm bảo thứ tự học
        self.q_queue.sort_by_key(|x| x.id);
        
        for item in self.q_queue.drain(..) {
            if self.learned_ids.contains(&item.id) {
                println!("  [LEARN] ID: {} | BỎ QUA (Đã học)", item.id);
                continue;
            }
            println!("  [LEARN] ID: {} | Nhãn: {:?}", item.id, item.true_label.unwrap());
            self.learned_ids.push(item.id);
            self.r_queue.push(item);
        }
        println!("--- [EPOCH] Kết thúc ---");
    }

    fn flush_final(&mut self) {
        println!("--- [FLUSH FINAL] Dọn dẹp P ---");
        while let Some(item) = self.p_queue.pop_front() {
            println!("  [DROP] ID: {} | Lý do: Hết epoch, không có nhãn", item.id);
        }
    }
}

fn main() {
    println!("=== BẮT ĐẦU TEST B4: GHÉP NHÃN QUA RANH GIỚI EPOCH ===");
    
    // Tăng q_max_size lên 5 để chứa đủ 1 epoch
    let mut scheduler = PQRScheduler::new(5);

    let epoch_size = 5;
    let delay = 2;

    for step in 1..=12 {
        println!("\n--- BƯỚC {} ---", step);
        
        // 1. Nhận dữ liệu mới (chỉ nhận đến bước 11)
        if step <= 11 {
            scheduler.receive_data(step);
        }

        // 2. Nhận nhãn đến trễ
        if step > delay {
            let labeled_id = step - delay;
            
            if step == 6 {
                // Test xáo trộn: nhãn ID 4 đến trước ID 3
                scheduler.receive_label(4, 1, true);
                scheduler.receive_label(3, 2, true); // Nhãn trùng của ID 3
            } 
            else if step == 7 {
                // Test nhãn trùng
                scheduler.receive_label(2, 1, false); // Nhãn trùng của ID 2
                scheduler.receive_label(5, 0, false);
            }
            else {
                let label = (labeled_id % 3) as u8;
                scheduler.receive_label(labeled_id, label, false);
            }
        }

        // 3. Ranh giới Epoch
        if step % epoch_size == 0 {
            scheduler.flush_epoch();
        }
    }

    // 4. Flush đợt cuối để học nốt ID 9, 10
    scheduler.flush_epoch();
    scheduler.flush_final();

    println!("\n=== KẾT QUẢ CUỐI CÙNG ===");
    println!("Các ID đã học: {:?}", scheduler.learned_ids);
    
    // Kiểm tra 10 ID đầu
    let first_10: Vec<usize> = scheduler.learned_ids.iter().filter(|&&id| id <= 10).cloned().collect();
    assert_eq!(first_10.len(), 10, "Lỗi: Không học đủ 10 ID!");
    println!("[TEST][B4][PASS] 10 ID đầu tiên đã được học đúng 1 lần.");
}
