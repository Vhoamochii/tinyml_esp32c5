# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research project: on-device continual learning (TinyML) on an **ESP32-C5** (RISC-V) where labels arrive **late**. The algorithm (a P/Q/R memory policy) is first simulated on a PC in Rust, then ported to the microcontroller. The research outline and reference books are in the parent folder (`../src/`, outside this git repo).

The LaTeX research report, written in Vietnamese, lives **outside this git repo** at `../report/`. It is the source of truth for the method:
- Chapter 2 covers the shared P/Q/R slot model, the policies, the metrics (BA, match rate, forgetting `F_A`) and the experiment design.
- Chapter 3 covers the three implementation directions (H1 pure Rust, H2 Rust + Python, H3 Rust + asm), the requirements YC1–YC6, and the measurement kernels K1–K5.

Build it with `cd ../report && latexmk`, which runs XeLaTeX + BibTeX and writes `build/main.pdf`. Do not use pdfLaTeX, because the Vietnamese text needs fontspec. Spots the group must fill in are marked `\canbosung{...}`. Do not invent results.

## Branches

There is no `main` branch yet, and the group plans to add one later.
- `feature/pc-sim-memsim` is **H1 (pure Rust)**. The layout below describes this branch.
- `feature/rust-python-xor` is **H2**: a Python XOR trainer (`pc/train_xor.py`), a `no_std` XOR firmware, and `pc/pc-sim/src/xor_tanh.rs`. Its layout differs (for example, firmware `src/main.rs` with a different crate name).

The report expects the firmware and the P/Q/R core to be identical across the two branches, with only the PC side differing.

## Layout: separate Cargo crates, no workspace

There is no top-level `Cargo.toml`. Run `cargo` from inside each crate's own directory.

| Crate | Role |
|---|---|
| `tinyml-core/` | `#![no_std]`, no heap; its only dependency is `libm`. This is the device-side logic shared by the PC and the board: the slot pool, the allocation policy, the 16→3 last layer, and the PRNG. |
| `pc/pc-sim/` | `std` host binary, depending on `tinyml-core` by path. It is the PC side only: the data generator, the delayed-label channel, and scoring. `criterion` is a dev-dependency and is not used yet. |
| `embedded/firmware/` | `no_std`/`no_main` with `esp-hal ~1.2.2`, `esp-println` and `tinyml-core`. So far it is only a self-test of one slot lifecycle, and it does not receive a data stream yet. |

## Commands

```sh
cd tinyml-core && cargo test                                   # all core tests
cd tinyml-core && cargo test test_memsim_happy_path -- --nocapture   # single test, with its output
cd pc/pc-sim && cargo run --release                            # 3000-step experiment (params are consts in main.rs)
cd embedded/firmware && cargo build --release                  # cross-compiles for riscv32imac
cd embedded/firmware && cargo run --release                    # flash + monitor via espflash
```

Firmware config:
- `.cargo/config.toml` sets the target to `riscv32imac-unknown-none-elf`, the linker to `rust-lld`, and the runner to `espflash flash --monitor`. It also has `[unstable] build-std`, which only takes effect on nightly, while `rust-toolchain.toml` pins stable.
- `build.rs` passes `-Tlinkall.x`.
- `esp_bootloader_esp_idf::esp_app_desc!()` is required, or espflash refuses to flash.
- `rust-version = "1.95"`.

### Build environment caveat
The specs say development happens on Ubuntu. On this Windows checkout nothing that needs a host linker works:
- MSVC `link.exe` and the Windows SDK are missing, and `/usr/bin/link` shadows the linker in Git Bash. So `cargo build`, `cargo test` and even `cargo check` fail in dependency build scripts. Treat these as environment problems, not code problems.
- To type-check without linking, call `rustc --emit=metadata` directly. Build `libm` to `.rmeta` from `~/.cargo/registry/src/*/libm-*/src/lib.rs`, then pass `--extern libm=...` for `tinyml-core` and `--extern tinyml_core=...` for the tests and `pc-sim`. `clippy-driver` accepts the same flags.
- The firmware cannot be checked here, because `esp-hal` has build scripts. `espflash` is not installed.

## Architecture

Device side (`tinyml-core`), following report Chapter 2:
- **`memsim.rs`** holds `MemoryPool<const N: usize>`, an array of `N` `Slot`s. Each sample occupies **one slot**, and its state changes in place: `Free → P → Q → R → Free`.
  - Sizes: `SLOT_SIZE_BYTES = size_of::<Slot>()`, currently 80, and `slots_for_budget(B) = B / SLOT_SIZE_BYTES`, so 2000 B gives 25 slots. The tests assert both numbers.
  - Operations: `alloc_p`, `assign_label` (P→Q by id; a miss is `MemError::IdNotFound`, a "ghép trượt"), `oldest_q` (FIFO by `t_state`), `move_to_r`, and `free`.
  - The pool is mechanism only. It never decides what to evict.
- **`policy.rs`** is the allocation decision module. YC3 says baseline and proposed policies must differ **only** here. `SimplePolicy` is a **temporary** policy, not one from the report:
  - P slots expire after `p_ttl`;
  - R is capped at `r_max` and filled by reservoir sampling;
  - replay samples are picked uniformly at random.

  The report's baseline (fixed ratios α with slot borrowing) and its proposal (adaptive `n_P` via Little's law) are still to be written as sibling modules.
- **`head.rs`** is the FP32 last layer, 16→3, and the only part trained on-device. It uses softmax with cross-entropy, and `sgd_step` does one averaged update over a batch (new sample + replay, kernel K3). It starts from zeros. The softmax/CE choice is an assumption; confirm it against the group's offline reference (YC1).
- **`rng.rs`** is `XorShift32`, so the PC and the board produce the same random sequence.

PC side (`pc/pc-sim/src/main.rs`): each step does the following.
1. Expire P slots.
2. Generate a sample: a seeded class centre plus uniform noise.
3. **Predict before training** and score the prediction (prequential accuracy).
4. Allocate a P slot, and schedule the label after a random delay in `[MIN_DELAY, MAX_DELAY]`. The label is sent even if the sample was dropped.
5. Match the labels that arrive this step.
6. Drain Q. Each Q sample trains with up to `REPLAY_K` R samples in one SGD step, and then the policy decides whether it goes to R or back to Free.

Known gaps against the report:
- The metric is accuracy, not BA, and the match rate is not computed as a ratio.
- Q is fully drained every step, with no cap on update opportunities.
- The data is stationary, with no A→B→C→A sequence, so forgetting `F_A` cannot be measured.

## Conventions

- Code comments are in Vietnamese with diacritics. `println!` output uses Vietnamese without diacritics (plain ASCII).
- Logs use a bracket-tag format: `[DEBUG][MAIN|STATUS] ...`, `[MEMSIM][ERROR] ...`, `[FW][BOOT|SELFTEST|PANIC] ...`, and in tests `[TEST][MODULE][CaseName] Bat dau...` / `... PASS`.
- Core tests live in `tinyml-core/tests/core_test.rs`. Error strings from `MemError::as_str` are asserted there.
- Working-tree files use CRLF line endings (`core.autocrlf`).
