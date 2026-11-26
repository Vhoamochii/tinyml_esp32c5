//! Phieu do kich thuoc that s, H va K_real(M) theo de cuong muc 3 (proxy PC).
//! Bo phai xac nhan lai bang harness A2. Khong sua engine.
use pc_sim::head_linear::LinearHead;
use pc_sim::mem_pool::{Cell, MemPool};

#[test]
fn mem_metrics() {
    let s = core::mem::size_of::<Cell>();
    let pool = core::mem::size_of::<MemPool<256>>();
    let h = pool - 256 * s;
    let head = core::mem::size_of::<LinearHead<3>>();
    println!("[METRICS] s(Cell)={} byte", s);
    println!("[METRICS] MemPool<256>={} byte, H={} byte", pool, h);
    println!("[METRICS] LinearHead<3>={} byte", head);
    println!("[METRICS] RAM that K=256 = {} byte", pool + head);
    for m in [8192usize, 4096, 2048, 1024, 512] {
        // Ngan sach phai lon hon phan quan ly H thi moi cho duoc o nao.
        if m <= h {
            println!("[METRICS] M={} byte -> K_real=0 (H chiem het)", m);
            continue;
        }
        let k = (m - h) / s;
        println!("[METRICS] M={} byte -> K_real={}", m, k);
    }
    assert!(s >= 31, "Cell phai chua du id+label+z+state+age");
}
