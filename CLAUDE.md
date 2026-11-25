# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is a research project on online bearing-fault diagnosis on an **ESP32-C5** (RISC-V, no FPU), where labels arrive **late**. For each vibration window, the device predicts right away. It then keeps the window in a shared P/Q/R slot array while it waits for the label, and updates only the FP32 last layer. Everything is written in pure Rust.

The method source of truth is the newest outline **`../src/De_cuong_phuong_phap_Rust_ESP32-C5.docx`**. It lives outside this repo; read it with a docx text extractor. Its main points:
- **Research question:**
  - How many R slots to protect (**r = 0/25/50 % of K**) when the speed changes from N15 (1500 rpm) to N09 (900 rpm).
  - Whether learning in **Q-priority** or **interleaved** order helps when labels arrive in bursts.
- **Mandatory checks before vibration data:** **XOR**, and **label matching across an epoch boundary**. Ubuntu and the board must match on every ID, label and learning order.
- **Work happens in stages:**
  - Step 3: pretrain.
  - Step 4: online learning on PC with real Paderborn data, config selection, then the test phase.
  - Step 5: on the board, covering timing, then UART streaming of real data (5e), then the final table on the board (5f).

The group has defined the three cases as follows:
- **C1:** no replay.
- **C2:** R evicted FIFO, which is the outline's rule.
- **C3:** R as a reservoir sample. **This deviates from the outline.**

The following deviations still need group approval:
- C3;
- the lr-selection rule (revised after seeing data);
- the stage-1 hand-crafted extractor, used instead of the outline's INT8 extractor.

The older LaTeX report in `../report/` (Little's-law policy, 13/6/10 bearing split) predates the outline and conflicts with it. Do not implement from the report without asking. Build it with `cd ../report && latexmk` (XeLaTeX, not pdfLaTeX).

## Branches

There is no `main` branch yet; the group plans to add one later.
- `feature/pc-sim-memsim` is the **pure-Rust** branch. Everything below describes this branch.
- `feature/rust-python-xor` is **Rust + Python**: a Python 2-8-1 tanh XOR trainer and a `no_std` XOR firmware. Its XOR is not the outline's XOR, and its layout differs.

## Layout: separate Cargo crates, no workspace

There is no top-level `Cargo.toml`. Run `cargo` from inside each crate's own directory.

| Crate | Role |
|---|---|
| `tinyml-core/` | `#![no_std]`, no heap; its only dependency is `libm`. Holds all device-side logic: memory, policy, schedule, head, feature extractor, UART protocol. Ubuntu and the board run the same code. |
| `pc/pc-sim/` | `std`. Contains the Paderborn `.mat` reader (`mat.rs`, `paderborn.rs`), the experiment runner, and the tools listed under Commands. |
| `embedded/firmware/` | `no_std`/`no_main`, using `esp-hal ~1.2.2` and `esp-println` (UART0). It runs the selftest, then the benchmarks, then stream mode. |

## Commands

```sh
cd tinyml-core && cargo test                                    # core_test (15) + features_test (6)
cd tinyml-core && cargo test test_reclaim_order -- --nocapture  # tests matching a name, with output
cd pc/pc-sim && cargo run --release --bin selftest              # XOR + epoch checks + [BENCH][EXTRACT] z line
cd pc/pc-sim && cargo run --release --bin pretrain              # REWRITES tinyml-core/src/pretrained.rs
cd pc/pc-sim && cargo run --release --bin pc-sim                # pick lr by rule, then locked table (config bearings)
cd pc/pc-sim && cargo run --release --bin pc-sim -- --sweep     # config sweep: K x r x retention x schedule
cd pc/pc-sim && cargo run --release --bin pc-sim -- --test      # test phase on the 6 test bearings
cd pc/pc-sim && cargo run --release --bin inspect_mat -- <file.mat>   # reader + extractor on one file
python3 pc/pc-sim/tools/features_ref.py [file.mat]              # NumPy reference to compare with inspect_mat
cd embedded/firmware && cargo run --release                     # flash + monitor (espflash); Ctrl+C after "[BENCH] XONG"
cd pc/pc-sim && cargo run --release --bin stream -- [steps=300] [C1|C2|C3] [rai|dot] [seed=11] [lr=0.1] [cauhinh|A|B] [port=/dev/ttyUSB0]
```

Data:
- The Paderborn root is `$PADERBORN_DIR`, defaulting to `$HOME/Work/Dataset`.
- File paths are `<bearing>/<cond>_<bearing>_<n>.mat`, with the channel `Y.vibration_1` at 64 kHz.

Firmware toolchain:
- `.cargo/config.toml` sets the target to `riscv32imac`, the linker to `rust-lld`, the runner to `espflash flash --monitor`, and `build-std = ["core"]`.
- `build.rs` passes `-Tlinkall.x`.
- `esp_bootloader_esp_idf::esp_app_desc!()` is required.

## Working environment

**Where things run.** The user builds and runs everything on an Ubuntu VM in VMware, at `~/Work/Project_ESP32C5`, and pastes output back.
- The board is reached through USB passthrough at `/dev/ttyUSB0`, and the user must be in the `dialout` group.
- Code moves by git: the user pushes from Windows and pulls on Ubuntu, and commits themselves.
- After changing files, list exactly which files must be synced, and say whether a re-flash is needed. A re-flash is needed for any change to the firmware or to `tinyml-core`.

**`tinyml-core/src/pretrained.rs`** is generated by `pretrain`.
- The committed copy is a placeholder with `AVAILABLE = false` and zero weights, so a firmware built from it predicts class 0 every time.
- Real weights exist only where `pretrain` was run, which is the Ubuntu VM. Never suggest `git checkout -- .`, `git reset --hard`, or anything else that would overwrite that file there.
- After every `pretrain`, re-flash the firmware. `stream` compares `protocol::pretrained_fingerprint()` with the `W=0x…` value the board reports, and refuses to run if they differ.

**Windows checkout.** Nothing that needs a host linker works here.
- MSVC `link.exe` is missing, and in Git Bash `/usr/bin/link` shadows it. As a result, `cargo build`, `cargo test` and `cargo check` fail in dependency build scripts.
- Type-check with `rustc`/`clippy-driver --emit=metadata`, building `.rmeta` files in dependency order:
  1. `libm` from `~/.cargo/registry/src/*/libm-*/src/lib.rs`;
  2. `tinyml_core` (`--extern libm=`);
  3. the tests;
  4. the `pc_sim` lib (`--extern tinyml_core= --extern miniz_oxide=`);
  5. the bins, with `CARGO_MANIFEST_DIR` set for `pretrain`.
- Rebuild every downstream `.rmeta` after `tinyml-core` changes. Otherwise you get E0463 "can't find crate".
- `tinyml-core` has pre-existing pedantic clippy warnings, so do not pass `-D warnings` for it.
- `stream` needs a hand-written stub `serialport` crate.
- The firmware can only be approximated, by compiling its functions against a stub of the few `esp-hal` items it uses (`Uart` with `read`/`write`/`read_ready`/`flush`/`apply_config`, `uart::Config`, `time::Instant`). Use only `esp-hal` APIs the firmware already uses, because a wrong guess costs the user a build round-trip.

## Architecture

**Per-step contract** (`Learner`, identical on PC, on the board, and in the PC reference inside `stream`). In each step:
1. **`on_window`**: predict and log the prediction **before** anything else, then admit the window to P.
2. **`on_label`**: pass the due labels in **ascending ID** order.
3. **`train(updates)`**: the locked config does 2 updates per step.

`tinyml-core` modules:
- **`memsim.rs`** holds `MemoryPool<const N>`, a fixed array of `Slot`s.
  - A slot's state changes in place: `Free → P → Q → R → Free`.
  - `order` is a global counter stamped on every state change. `oldest(R)` is therefore the slot that **entered R earliest**, which is the group's decision for "oldest R".
  - `SLOT_SIZE_BYTES` is 76, and `K = slots_for_budget(M, H) = (M − H) / 76`.
- **`policy.rs`** decides reclaiming.
  - **When a window arrives:** use a free slot first. Otherwise reclaim from R if R holds more than r samples, else take the oldest P. Q is never reclaimed. If nothing can be reclaimed, the result is `DROP_NO_SLOT`.
  - **Which R slot is reclaimed** depends on `Retention`. `Fifo` takes the oldest R. `Reservoir` takes the R slot with the smallest `reservoir_key(id, seed)` (fmix32, integer-only, so PC and board agree).
  - **When a label arrives:** it matches a P slot by ID. If Q is full, the result is `DROP_Q_FULL`. A label for an ID already in Q or R is `LABEL_DUP`; one for an ID no longer in memory is `LABEL_MISS`.
  - Fields are `u32`, so H (`MANAGEMENT_BYTES`, currently 44) is the same on 64-bit and 32-bit targets.
- **`schedule.rs`** offers `Interleaved` or `QPriority` (replay only when Q is empty).
  - Each first-learn earns one replay credit, capped at K.
  - A replay is drawn uniformly from R, skipping the ID learned just before.
  - `replay_enabled = false` gives C1.
- **`head.rs`**: `Head<const C>` is the FP32 `Wz + b` with softmax and cross-entropy. `sgd_step` updates on one sample.
- **`features.rs`** is the stage-1 FP32 extractor: 16 hand-crafted, amplitude-invariant features.
  - A window is 16 blocks × 1024 = 16384 samples (256 ms at 64 kHz). `WindowExtractor::push_block` consumes it block by block, so the board never holds a whole window of f32.
  - Features 0–5 are shape features.
  - Features 6–9 are log relative band energies, computed on blocks 0/4/8/12 only.
  - Features 10–15 are the envelope spectrum in **fixed** 10–300 Hz bands, with bins near multiples of 50 Hz excluded (mains noise).
  - The output must be **bit-identical** on Ubuntu and the board. Statistics are accumulated per block in f32, and the FFT uses a twiddle table. Compare the `[BENCH][EXTRACT] z` lines of `selftest` and the firmware.
  - `extract()` (whole window) must equal streaming (a test enforces this).
- **`normalize`** plus the pretrained `FEATURE_MEAN`/`FEATURE_STD` run before `on_window` on every path.
- **`event.rs`**: every action emits an `Event`, displayed as `[EVT] step=… id=… KIND`. Logs are compared line by line.
- **`selftest.rs`** holds the outline's XOR checks (fixed-score 4/4; one SGD step from 0 gives ±0.05; 200 epochs give 4/4) and the 10-ID epoch check.
- **`testsignal.rs`** is the shared fake signal for the extractor benchmark and the cross-check.
- **`protocol.rs`** is the UART protocol for 5e, shared by `stream` and the firmware; see below.

**PC side** (`pc/pc-sim/src/main.rs`):
- **Data split by physical bearing** (`paderborn.rs`):
  - train: 8 bearings;
  - config: K004, KI14, KA16;
  - test streams: A = K005, KI18, KA22 and B = K006, KI21, KA30.
- **A stream** is N15 measurements 1–16 followed by N09 measurements 1–20, alternating records across the 3 classes, with 15 windows per record. N15 measurements 17–20 are held out to measure forgetting.
- **Label schedules** use groups of 9: `rai` (spread) delays by j, and `dot` (burst) delays by 8−j.
- **Locked config:** K=25 (M=1944), r=0, QPriority, Q_MAX=9, 2 updates per step.
- **Choosing lr:**
  - The rule is applied only on the config bearings, and is never tuned on the test set.
  - The rule is: the smallest lr in `LR_CANDIDATES` whose new-speed BA is within `LR_BA_TOLERANCE` of the best.
  - The rule is locked, not the resulting number, so lr is re-chosen if the extractor changes.
- **Reported metrics:** BA by class/bearing, forgetting, recovery, label match, drops by reason, and label→learn wait, as mean ± SD over seeds.

**Board stream (5e, `protocol.rs` + `bin/stream.rs` + `firmware/src/main.rs`):**
- **Opening the port resets the board.** RTS is held for 100 ms while DTR is low, which pulls EN low. The handshake then runs at 115200 (`G`), and both sides switch to 921600 (`P` gets `[STREAM] OK`).
- **Config:** `C` + 24 bytes carries the config. The board answers `CONFIG OK K= H= W=<fingerprint> pretrain=`. `LOCKED_K` in the firmware must equal `K` in `stream.rs`.
- **Step frames:** `S` + step + label count + FNV checksum + labels + 16384 little-endian f32. Raw f32 is sent so the features match bit for bit. The PC waits for each `[STEP]` line before sending the next frame, so this is **not real time**.
- **Board response:**
  - The board verifies the checksum **before** touching the extractor or learner.
  - It times only `push_block` and the learner calls, and sends the `[EVT]` lines afterwards.
  - It ends with `[STEP] … sum= chk=`, where `chk` is `response_sum` over the response text.
- **Recovery:** any failure (a bad frame gives `[STREAM][GUI LAI]`, a bad `chk`, or no response within 3 s) makes the PC resend the whole frame.
  - The board learns each step **at most once**. A verified frame for the step it just processed only re-sends the stored result.
  - Board TX is paced: 16 bytes, flush, then a 2 ms pause (about 8 KB/s). At 921600, board-to-PC text lost byte runs through the USB-UART bridge and the VM, even with the FIFO kept half empty; PC-to-board frames arrive intact.
- **Timing budget:** 256 ms per window. The extractor takes about 176 ms on the board at `CpuClock::max()`, and one learn step about 0.2 ms.

## Conventions

- Code comments are in Vietnamese with diacritics. `println!` output uses Vietnamese without diacritics (plain ASCII). Replies to the user are in Vietnamese.
- Logs use bracket tags: `[EVT]`, `[XOR][FIXED|STEP|TRAIN]`, `[EPOCH][...]`, `[SELFTEST]`, `[BENCH][RAM|EXTRACT|LEARN]`, `[STREAM]`, `[STEP]`, `[FW][BOOT|PANIC]`. Test logs use `[TEST][MODULE][CaseName] Bat dau...` / `... PASS`.
- Working-tree files use CRLF line endings (`core.autocrlf`). Edits made with scripts must keep CRLF.
