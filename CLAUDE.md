# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research project: online bearing-fault diagnosis on an **ESP32-C5** (RISC-V, no FPU) where labels arrive **late**. The device predicts on each vibration window immediately, keeps samples in a shared P/Q/R slot array while waiting for labels, and updates only the FP32 last layer.

The method source of truth is the newest outline **`../src/De_cuong_phuong_phap_Rust_ESP32-C5.docx`**, which is outside this repo; read it with a docx text extractor. Its main points:
- The research question is how many R slots to protect (**r = 0/25/50 % of K**) when the speed changes from N15 (1500 rpm) to N09 (900 rpm), and whether learning in **Q-priority** or **interleaved** order helps when labels arrive in bursts.
- Two checks are mandatory before vibration data: **XOR** and **label matching across an epoch boundary**. Ubuntu and the board must match on every ID, label and learning order.
- Three "cases" are compared. Their exact meaning is still to be defined by the group.

The older LaTeX report in `../report/` (Little's-law policy, 13/6/10 bearing split) predates this outline and conflicts with it. Do not implement from the report without asking. Build it with `cd ../report && latexmk` (XeLaTeX, not pdfLaTeX).

## Branches

There is no `main` branch yet, and the group plans to add one later.
- `feature/pc-sim-memsim` is the **pure-Rust** branch. The layout below describes this branch.
- `feature/rust-python-xor` is **Rust + Python**: a Python 2-8-1 tanh XOR trainer, a `no_std` XOR firmware, and `pc/pc-sim/src/xor_tanh.rs`. Its XOR is not the outline's XOR (see below), and its layout differs.

## Layout: separate Cargo crates, no workspace

There is no top-level `Cargo.toml`. Run `cargo` from inside each crate's own directory.

| Crate | Role |
|---|---|
| `tinyml-core/` | `#![no_std]`, no heap; its only dependency is `libm`. All device-side logic, shared by Ubuntu and the board. |
| `pc/pc-sim/` | `std`. Two binaries: `pc-sim`, the PC-side emitter and scorer with the experiment matrix on **synthetic** two-speed data, and `selftest`, which runs the two mandatory checks. |
| `embedded/firmware/` | `no_std`/`no_main` with `esp-hal ~1.2.2`, `esp-println` and `tinyml-core`. It runs the same two checks and prints the same lines over UART0. It does not receive a data stream yet. |

## Commands

```sh
cd tinyml-core && cargo test                                    # all core tests
cd tinyml-core && cargo test test_reclaim_order -- --nocapture  # single test, with its output
cd pc/pc-sim && cargo run --release --bin selftest              # XOR + epoch checks on Ubuntu
cd pc/pc-sim && cargo run --release --bin pc-sim                # experiment tables (synthetic data)
cd embedded/firmware && cargo run --release                     # flash + monitor; compare its [XOR]/[EVT]/[EPOCH] lines with selftest
```

Firmware config:
- `.cargo/config.toml` sets the target to `riscv32imac-unknown-none-elf`, the linker to `rust-lld`, and the runner to `espflash flash --monitor`.
- `build.rs` passes `-Tlinkall.x`.
- `esp_bootloader_esp_idf::esp_app_desc!()` is required.
- On the user's Ubuntu VM, flashing needs USB passthrough in VMware and membership of the `dialout` group. The board shows up as `/dev/ttyUSB0`.

### Build environment caveat
The team builds and runs on an Ubuntu VM. On this Windows checkout nothing that needs a host linker works:
- MSVC `link.exe` and the Windows SDK are missing, so `cargo build`, `cargo test` and even `cargo check` fail in build scripts. Treat these as environment problems.
- To type-check without linking, call `rustc --emit=metadata` directly. Build `libm` to `.rmeta` from `~/.cargo/registry/src/*/libm-*/src/lib.rs`, then pass `--extern libm=...` for `tinyml-core` and `--extern tinyml_core=...` for the tests and `pc-sim`. `clippy-driver` accepts the same flags.
- The firmware cannot be checked on Windows. Ask the user to build it on Ubuntu.

## Architecture (`tinyml-core`)

In each step the caller invokes three `Learner` methods, in this order:
1. **`on_window`**: predict and log the prediction **before** anything else, then admit the window to P.
2. **`on_label`**: the caller passes due labels in **ascending ID** order.
3. **`train(max_updates)`**.

Modules:
- **`memsim.rs`** holds `MemoryPool<const N>`, a fixed array of `Slot`s.
  - A slot's state changes in place: `Free → P → Q → R → Free`.
  - `order` is a global counter stamped whenever a slot changes state. Therefore `oldest(P)` is the longest-waiting P sample, `oldest(Q)` is the FIFO head, and `oldest(R)` is the sample that **entered R earliest**, which is the group's decision for "oldest R".
  - `SLOT_SIZE_BYTES = size_of::<Slot>()`, currently 76.
  - `slots_for_budget(M, H) = (M − H) / s`.
- **`policy.rs`** (`MemoryPolicy { r_protected, q_max }`) decides which slot to reclaim.
  - When a window arrives, use a free slot first. Otherwise reclaim the oldest R if R holds more than r samples, else the oldest P. Q is never reclaimed. If nothing can be reclaimed, the new sample gets `DROP_NO_SLOT`.
  - When a label arrives, it matches a P slot by ID. If Q is full, the result is `DROP_Q_FULL`. A label for an ID in Q or R is `LABEL_DUP` and never causes a second first-learn. A label for an ID no longer in memory is `LABEL_MISS`.
  - Fields are `u32`, so H is identical on 64-bit and 32-bit targets.
- **`schedule.rs`** offers `Interleaved` (one replay after each new learn) or `QPriority` (replay only when Q is empty).
  - Each first-learn earns one replay credit, capped at K.
  - A replay is drawn uniformly from R, skipping the ID learned in the previous update.
  - `replay_enabled = false` gives the no-replay control.
- **`head.rs`** is `Head<const C>`, the FP32 `Wz + b` with softmax and cross-entropy. `sgd_step` does one update on **one** sample. C is 3 for bearings and 2 for XOR.
- **`event.rs`**: every action emits an `Event`, displayed as `[EVT] step=… id=… KIND`. Ubuntu and board logs are compared line by line.
- **`selftest.rs`** runs the outline's XOR checks and the 10-ID epoch check (Table 3), on z = (x1, x2, x1·x2, 0…).
  - The fixed-score prediction must give 4/4.
  - One SGD step from W = b = 0 with lr 0.1 must give ±0.05 (tolerance 1e-6).
  - Training 200 epochs must give 4/4.
  - These expectations were verified numerically with NumPy.
- **`learner.rs`** wires the pieces together. `MANAGEMENT_BYTES` is H, computed with `size_of`.

PC side (`pc/pc-sim/src/main.rs`):
- The data is synthetic: an old speed, then a new speed (shifted class centres). Records rotate classes, with 32 windows per record, and some old-speed records are held out to measure forgetting.
- Label schedules use groups of 9: `Spread` delays by j and `Burst` delays by 8−j.
- It prints BA at the new speed, forgetting, recovery (a 96-window sliding BA), match %, drops by reason, first-learns, replays, and mean label→learn wait.
- These are not implemented yet: the Paderborn `.mat` reader, the INT8 extractor, the UART streaming protocol, per-bearing BA, and timing/RAM measurement.

## Conventions

- Code comments are in Vietnamese with diacritics. `println!` output uses Vietnamese without diacritics (plain ASCII).
- Logs use bracket tags: `[DEBUG][MAIN]`, `[EVT]`, `[XOR][FIXED|STEP|TRAIN]`, `[EPOCH][...]`, `[SELFTEST]`, `[FW][BOOT|PANIC]`. Test logs use `[TEST][MODULE][CaseName] Bat dau...` / `... PASS`.
- Working-tree files use CRLF line endings (`core.autocrlf`).
