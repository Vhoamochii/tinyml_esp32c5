# specs.md — Continual TinyML ESP32-C5, case Rust + Python

## Trang thai (2026-11-24, sau D7)
- Loader PASS: train 80117 / val 30046 / test 60111 / kb 30040; 6 khoa npz.
  Ho so nhan: 26=K0, 27=KA, 28=KI, 29=KB. Cua so/ban ghi dao dong 249-271.
- D4/D5: het khoi o (blocks=54020), lop luan phien; H1 dong.
- D6: log RECALL_BEFORE/AFTER + w_norm + max_abs_z; snapshot truoc N09 dau.
  H2 xac nhan: max_abs_z=1.343e5, w_norm~3.4e4, recall don lop, ba_new~59.
- D7 (da duyet, Python-only): chuan hoa mean/std train N15 (mien nguon,
  --norm-source all de doi), winsorize +/-4 sigma, scale deu 4/127.
  Ghi fixtures/feature_norm.csv (dim,mean,std). Rust KHONG doi.
  Ky vong: max_abs_z<=4, recall khong don lop, BA/forget trung thuc.
- Anti-loop: task "BA/forget bat thuong" da sua 2 lan (D5, D7).
  Neu van benh sau D7: DUNG, 3 gia thuyet, rollback + xac nhan truoc khi va tiep.

## Dinh nghia 3 Case
- Case 1: Rust core + Python orchestration (dang lam).
- Case 2: Pure Rust standalone.
- Case 3: Rust + C/C++ (extern "C").

## Con treo (KHONG tu y sua)
1. Do RAM/P95/qua han tren bo (T=16000 us); so PC chi proxy.
2. Quet r=0/25/50% va 5 hat giong, mean +/- s.d.; Case 2/3 chua co crate.
3. Chot stride/M/H/r truoc khi do bo.
4. match=100/miss=0: pool chua bi ep tai; can quet r de thay thu ho that.

## Cach chay
- Loader: cd pc && python3 data_loader_vibdata.py
- CSV: cd pc/pc-sim && python3 make_feature_csv.py --npz
- Bang: cargo test --test table_harness -- --nocapture
- Regression: cargo test -- --nocapture

## Quy tac
Log [LOADER]/[CSVGEN]/[TEST]... expected/actual PASS/FAIL. Comment tieng Viet, khong emoji.
Khong sua code chua duyet. Loop 3 lan -> rollback. Synthetic khong vao bao cao.
Khong kep forget = max(0,...): giau benh.
