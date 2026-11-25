//! Case 3: Rust goi C qua extern "C".
//! Chi khai bao ham C thuc su dung (sgd_predict, sgd_step);
//! sgd_scores/sgd_softmax van ton tai ben C nhung khong can goi tu Rust.
//! Kieu ABI: i8/u8/f32 tuong ung int8_t/uint8_t/float (std::os::raw khong co c_int8_t).

use pc_sim::head_linear::{LinearHead, Quant, N_FEATURES};

extern "C" {
    fn sgd_predict(w: *const f32, b: *const f32, z: *const f32, C: i32, F: i32) -> u8;
    fn sgd_step(w: *mut f32, b: *mut f32, z: *const f32,
                label: u8, lr: f32, C: i32, F: i32) -> f32;
}

/// Wrapper predict: giai luong tu INT8 -> FP32 (khop LinearHead::predict) roi goi C.
pub fn ffi_predict(head: &LinearHead<3>, z_i8: [i8; N_FEATURES], q: &Quant) -> u8 {
    let mut z = [0.0f32; N_FEATURES];
    for i in 0..N_FEATURES {
        z[i] = (z_i8[i] as f32 - q.zero[i] as f32) * q.scale[i];
    }
    unsafe {
        sgd_predict(head.w.as_ptr() as *const f32 as *const f32 as *const f32, head.b.as_ptr(), z.as_ptr(), 3, N_FEATURES as i32)
    }
}

/// Wrapper sgd_step: giai luong tu INT8 -> FP32 roi goi C, cap nhat head tai cho.
pub fn ffi_sgd_step(head: &mut LinearHead<3>, z_i8: [i8; N_FEATURES], q: &Quant,
                    label: u8, lr: f32) -> f32 {
    if (label as usize) >= 3 {
        return f32::NAN;
    }
    let mut z = [0.0f32; N_FEATURES];
    for i in 0..N_FEATURES {
        z[i] = (z_i8[i] as f32 - q.zero[i] as f32) * q.scale[i];
    }
    unsafe {
        sgd_step(head.w.as_mut_ptr() as *mut f32 as *mut f32 as *mut f32, head.b.as_mut_ptr(), z.as_ptr(),
                 label, lr, 3, N_FEATURES as i32)
    }
}

/// Do overhead FFI thuan: N lan goi predict, tra ve ns/call.
/// black_box chong optimizer xoa vong do.
pub fn ffi_overhead_ns(head: &LinearHead<3>, z_i8: [i8; N_FEATURES], q: &Quant, n: usize) -> f64 {
    use std::time::Instant;
    let mut z = [0.0f32; N_FEATURES];
    for i in 0..N_FEATURES {
        z[i] = (z_i8[i] as f32 - q.zero[i] as f32) * q.scale[i];
    }
    let t0 = Instant::now();
    let mut sink: u8 = 0;
    for _ in 0..n {
        sink = unsafe {
            sgd_predict(head.w.as_ptr() as *const f32 as *const f32 as *const f32, head.b.as_ptr(), z.as_ptr(), 3, N_FEATURES as i32)
        };
    }
    std::hint::black_box(sink);
    t0.elapsed().as_nanos() as f64 / n as f64
}
