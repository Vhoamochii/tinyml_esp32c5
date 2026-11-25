//! Compile C scalar -O3 (khong SIMD; SIMD thuoc E4b rieng).
fn main() {
    cc::Build::new()
        .file("csrc/sgd_softmax.c")
        .opt_level(3)
        .warnings(false)
        .compile("sgd_softmax");
    println!("cargo:rerun-if-changed=csrc/sgd_softmax.c");
}
