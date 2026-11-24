// Giao thức UART giữa Ubuntu (pc-sim, chương trình `stream`) và ESP32-C5 (firmware), bước 5e.
// Hai bên dùng chung module này để khung dữ liệu luôn khớp.
//
// Bắt tay ở BOOT_BAUD: PC gửi CMD_GO, bo trả "[STREAM] BAUD 921600" rồi đổi sang FAST_BAUD;
// PC đổi theo và gửi CMD_PING, bo trả "[STREAM] OK". Sau đó:
//   CMD_CONFIG + CONFIG_LEN byte -> bo tạo lại Learner (bộ nhớ rỗng, trọng số pretrain),
//                                   trả "[STREAM] CONFIG OK K=.. H=.. W=0x.." (W = pretrained_fingerprint)
//   CMD_STEP + bước (u32) + số nhãn (u8) + sum (u32) + mỗi nhãn (ID u32, nhãn u8) + WINDOW_LEN mẫu f32
//            (sum = frame_sum(bước + số nhãn, nhãn + mẫu))
//                                -> bo kiểm tra sum TRƯỚC khi xử lý. Đúng: trả các dòng [EVT] của bước (gửi SAU
//                                   khi đo thời gian), rồi "[STEP] step=.. pred=.. ext_us=.. learn_us=.. sum=.. chk=.."
//                                   (chk = response_sum, để PC biết phản hồi đến nơi đủ và đúng).
//                                   Hỏng (thiếu byte quá FRAME_IDLE_MS, lỗi UART, sai sum): bỏ khung, in lý do,
//                                   chờ đường truyền im rồi trả RESEND_LINE.
//   Mọi sự cố (khung xuống hỏng, phản hồi lên hỏng hoặc không có) PC đều xử lý bằng cách gửi lại nguyên khung.
//   Bo chỉ học mỗi bước một lần: khung đúng mà trùng bước vừa xử lý thì gửi lại kết quả cũ, không học lại.
//   Vì vậy gửi lại không làm lệch kết quả so với Ubuntu
//   CMD_PING                     -> "[STREAM] OK"
// Số nguyên và f32 theo thứ tự byte little-endian. Gửi f32 nguyên bản để đặc trưng trên bo khớp từng bit với PC.
// Không phát theo thời gian thực (UART không đủ băng thông cho 64 KB mỗi 256 ms): PC chờ dòng [STEP] rồi mới
// gửi bước tiếp; bo tự đo thời gian tính, thời gian truyền được PC ghi riêng.

use crate::features::WINDOW_LEN;
use crate::learner::Config;
use crate::policy::Retention;
use crate::pretrained;
use crate::schedule::Schedule;

pub const BOOT_BAUD: u32 = 115_200;
pub const FAST_BAUD: u32 = 921_600;

pub const CMD_GO: u8 = b'G';
pub const CMD_PING: u8 = b'P';
pub const CMD_CONFIG: u8 = b'C';
pub const CMD_STEP: u8 = b'S';

/// Độ dài phần cấu hình sau CMD_CONFIG
pub const CONFIG_LEN: usize = 24;
/// Số nhãn tối đa trong một khung bước
pub const MAX_LABELS_PER_STEP: usize = 32;
/// Số byte mẫu của một cửa sổ
pub const SAMPLE_BYTES: usize = WINDOW_LEN * 4;
/// Độ dài đầu khung sau CMD_STEP: bước (u32) + số nhãn (u8) + sum (u32)
pub const STEP_HEAD_LEN: usize = 9;
/// Trong một khung, quá chừng này ms không có byte mới thì bo coi như mất dữ liệu
pub const FRAME_IDLE_MS: u64 = 500;
/// Dòng bo gửi khi bỏ một khung hỏng và sẵn sàng nhận lại
pub const RESEND_LINE: &str = "[STREAM][GUI LAI]";

/// Giá trị đầu của mã FNV-1a 32 bit
pub const FNV_OFFSET: u32 = 0x811c_9dc5;
const FNV_PRIME: u32 = 0x0100_0193;
/// Phần cuối dòng [STEP] mang mã kiểm tra phản hồi
pub const CHK_KEY: &str = " chk=";

/// Nạp thêm `bytes` vào mã FNV-1a đang tính dở `h`
pub fn fnv_feed(mut h: u32, bytes: &[u8]) -> u32 {
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Mã kiểm tra FNV-1a 32 bit của các phần nối liền nhau (bằng mã của cả dãy ghép lại)
pub fn frame_sum(parts: &[&[u8]]) -> u32 {
    parts.iter().fold(FNV_OFFSET, |h, p| fnv_feed(h, p))
}

/// Mã kiểm tra phản hồi một bước: các dòng [EVT] (mỗi dòng kèm '\n') rồi phần dòng [STEP] đứng trước CHK_KEY
pub fn response_sum<'a>(events: impl IntoIterator<Item = &'a str>, step_prefix: &str) -> u32 {
    let h = events.into_iter().fold(FNV_OFFSET, |h, line| fnv_feed(fnv_feed(h, line.as_bytes()), b"\n"));
    fnv_feed(h, step_prefix.as_bytes())
}

/// Dấu vân tay của trọng số pretrain đang được biên dịch vào (hệ số chuẩn hóa, W, b).
/// PC và bo phải giống nhau; khác nghĩa là firmware được build với pretrained.rs khác.
pub fn pretrained_fingerprint() -> u32 {
    let mut h = FNV_OFFSET;
    let values = pretrained::FEATURE_MEAN
        .iter()
        .chain(pretrained::FEATURE_STD.iter())
        .chain(pretrained::WEIGHTS.iter().flatten())
        .chain(pretrained::BIAS.iter());
    for v in values {
        h = fnv_feed(h, &v.to_bits().to_le_bytes());
    }
    h
}

/// Mã hóa cấu hình bộ học và số lượt học mỗi bước
pub fn encode_config(cfg: &Config, updates_per_step: u32) -> [u8; CONFIG_LEN] {
    let mut out = [0u8; CONFIG_LEN];
    out[0] = match cfg.retention {
        Retention::Fifo => 0,
        Retention::Reservoir => 1,
    };
    out[1] = cfg.replay_enabled as u8;
    out[2] = match cfg.schedule {
        Schedule::Interleaved => 0,
        Schedule::QPriority => 1,
    };
    out[4..8].copy_from_slice(&cfg.r_protected.to_le_bytes());
    out[8..12].copy_from_slice(&cfg.q_max.to_le_bytes());
    out[12..16].copy_from_slice(&cfg.lr.to_le_bytes());
    out[16..20].copy_from_slice(&cfg.seed.to_le_bytes());
    out[20..24].copy_from_slice(&updates_per_step.to_le_bytes());
    out
}

/// Giải mã cấu hình; None nếu mã luật R hoặc lịch học không hợp lệ
pub fn decode_config(raw: &[u8; CONFIG_LEN]) -> Option<(Config, u32)> {
    let u32_at = |i: usize| u32::from_le_bytes([raw[i], raw[i + 1], raw[i + 2], raw[i + 3]]);
    let retention = match raw[0] {
        0 => Retention::Fifo,
        1 => Retention::Reservoir,
        _ => return None,
    };
    let schedule = match raw[2] {
        0 => Schedule::Interleaved,
        1 => Schedule::QPriority,
        _ => return None,
    };
    let cfg = Config {
        r_protected: u32_at(4),
        q_max: u32_at(8),
        schedule,
        replay_enabled: raw[1] != 0,
        lr: f32::from_bits(u32_at(12)),
        seed: u32_at(16),
        retention,
    };
    Some((cfg, u32_at(20)))
}
