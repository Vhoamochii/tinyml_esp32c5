fn main() {
    // linkall.x phải là linker script cuối cùng
    println!("cargo:rustc-link-arg=-Tlinkall.x");
}
