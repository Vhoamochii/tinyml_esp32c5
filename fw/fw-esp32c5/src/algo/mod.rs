//! Re-export engine tu tinyol-core de fw dung chung voi PC.
pub use tinyol_core::mem_pool::{Cell, CellState, DropReason, EvictReason, LabelError, MemPool};
pub use tinyol_core::head_linear::{LinearHead, Quant, N_FEATURES};
pub use tinyol_core::xor_tanh::{XorTanhNet, XOR_DATA};
pub use tinyol_core::epoch_fixture::{EPOCH_FEATURES, EPOCH_LABELS, LABEL_SCHEDULE, PREDICT_STEPS, TOTAL_STEPS};
