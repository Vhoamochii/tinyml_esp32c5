// Hai phép thử bắt buộc (đề cương, mục 4) chạy trên Ubuntu.
// Firmware chạy đúng các hàm này trên ESP32-C5; so từng dòng log giữa hai bên.
//   cargo run --release --bin selftest > ubuntu.log

use tinyml_core::selftest;

fn main() {
    let mut out = |a: std::fmt::Arguments<'_>| println!("{a}");
    let xor_ok = selftest::run_xor(&mut out);
    let epoch_ok = selftest::run_epoch_matching(&mut out);
    println!(
        "[SELFTEST] XOR={} EPOCH={}",
        if xor_ok { "PASS" } else { "FAIL" },
        if epoch_ok { "PASS" } else { "FAIL" }
    );
    if !(xor_ok && epoch_ok) {
        std::process::exit(1);
    }
}
