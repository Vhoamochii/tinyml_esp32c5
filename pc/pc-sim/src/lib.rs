//! Lop vo PC: engine that (head_linear, mem_pool, stream_runner, schedule)
//! + re-export sample_queue cua tinyol-core cho test T6/T7.
#![allow(dead_code)]

pub mod schedule;
pub mod stream_runner;
pub mod head_linear;
pub mod mem_pool;
pub mod last_layer;

pub use tinyol_core::sample_queue;
pub mod xor_tanh;
