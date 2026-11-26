//! Test fixture epoch 10 ID tren MemPool that.
//! Ghi log su kien ra stdout de doi chieu voi firmware sau nay.
use pc_sim::mem_pool::{CellState, MemPool};
use tinyol_core::epoch_fixture::{EPOCH_FEATURES, LABEL_SCHEDULE, PREDICT_STEPS, TOTAL_STEPS};

#[test]
fn epoch_10_id_tren_mempool() {
    let mut pool: MemPool<16> = MemPool::new(0);  // K=16, r=0
    let mut log_lines: Vec<String> = Vec::new();
    
    log_lines.push(format!("step,event,id,label,state_after"));
    
    for step in 0..TOTAL_STEPS {
        // 1. Predict (neu con trong PREDICT_STEPS)
        if step < PREDICT_STEPS {
            let z = EPOCH_FEATURES[step];
            match pool.push(z) {
                Ok(id) => {
                    log_lines.push(format!("{},predict,{},,P={}", 
                        step, id, pool.count_state(CellState::Pending)));
                }
                Err(_) => {
                    log_lines.push(format!("{},predict_fail,,,P={}", 
                        step, pool.count_state(CellState::Pending)));
                }
            }
        }
        
        // 2. Tra nhan (neu co trong schedule)
        if let Some(label_id) = LABEL_SCHEDULE[step] {
            let label = 0u8;  // Tat ca nhan = 0 cho don gian
            match pool.on_label(label_id, label) {
                Ok(()) => {
                    log_lines.push(format!("{},label,{},{},P={} Q={} R={}", 
                        step, label_id, label,
                        pool.count_state(CellState::Pending),
                        pool.count_state(CellState::Ready),
                        pool.count_state(CellState::Replay)));
                }
                Err(e) => {
                    log_lines.push(format!("{},label_fail,{},{:?},P={} Q={} R={}", 
                        step, label_id, e,
                        pool.count_state(CellState::Pending),
                        pool.count_state(CellState::Ready),
                        pool.count_state(CellState::Replay)));
                }
            }
        }
        
        // 3. Hoc lan dau (pop tu Q, hoc, promote sang R)
        while let Some(cell) = pool.pop_ready() {
            let learned_id = cell.id;
            // Simulate hoc: khong can sgd_step, chi promote
            pool.promote_to_replay(cell);
            log_lines.push(format!("{},learn_first,{},,P={} Q={} R={}", 
                step, learned_id,
                pool.count_state(CellState::Pending),
                pool.count_state(CellState::Ready),
                pool.count_state(CellState::Replay)));
        }
    }
    
    // In log
    println!("\n=== EPOCH LOG ===");
    for line in &log_lines {
        println!("{}", line);
    }
    
    // Kiem tra ket qua cuoi
    let (p, q, r) = pool.snapshot_pqr();
    assert_eq!(p, 0, "P phai rong sau khi tra het nhan");
    assert_eq!(q, 0, "Q phai rong sau khi hoc het");
    assert_eq!(r, 10, "R phai co dung 10 mau");
    assert_eq!(pool.learned_first, 10, "Phai hoc dung 10 lan dau");
    assert_eq!(pool.miss_labels, 0, "Khong duoc mat nhan nao");
    
    println!("\n=== FINAL STATE ===");
    println!("P={}, Q={}, R={}", p, q, r);
    println!("learned_first={}, miss_labels={}", pool.learned_first, pool.miss_labels);
    println!("drop_total={}", pool.drop_total());
}
