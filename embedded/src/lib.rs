//! Thu vien loi cho firmware ESP32-C5.
#![no_std]

// Re-export tu tinyol-core (dung chung voi PC)
pub use tinyol_core::xor_tanh;
pub use tinyol_core::sample_queue;
pub use tinyol_core::mem_pool;

// Module rieng cua embedded (lop cuoi tuyen tinh)
pub mod head_linear;
