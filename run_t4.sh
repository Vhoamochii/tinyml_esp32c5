#!/usr/bin/env bash
# Chi chay T4: regression + protocol_val. T2/T3 da PASS, khong chay lai loader.
# Chay: bash run_t4.sh 2>&1 | tee t4.log
# Dung main() + return, khong exit -> khong dong terminal.

main() {
    cd ~/Work/Project_ESP32C5/pc/pc-sim || { echo "[STOP] thieu pc/pc-sim"; return 1; }

    # F4: override target-dir ve cho ghi duoc, thang config /mnt/data/fwtarget
    export CARGO_TARGET_DIR="$HOME/Work/Project_ESP32C5/target_pc"
    mkdir -p "$CARGO_TARGET_DIR" || { echo "[STOP] khong tao duoc target dir"; return 1; }
    echo "[T4] CARGO_TARGET_DIR=$CARGO_TARGET_DIR"

    # Tien quyet: val CSV phai ton tai (T3a)
    if [ ! -f fixtures_val/features_val.csv ]; then
        echo "[STOP] thieu fixtures_val/features_val.csv - T3a chua chay"; return 1
    fi

    echo "=== [T4a] Regression table_harness ==="
    cargo test --test table_harness -- --nocapture
    if [ $? -ne 0 ]; then echo "[STOP] table_harness FAIL"; return 1; fi

    echo "=== [T4b] protocol_val - quet r, khoa r tren 3 o chon cau hinh ==="
    cargo test --test protocol_val -- --ignored --nocapture
    if [ $? -ne 0 ]; then echo "[STOP] protocol_val FAIL"; return 1; fi

    # San pham khoa r phai ton tai
    if [ ! -f results/val_lock.csv ]; then echo "[STOP] thieu results/val_lock.csv"; return 1; fi
    echo "[T4] noi dung val_lock.csv:"
    cat results/val_lock.csv
    echo "[DONE] T4 PASS - r da khoa theo tung lich nhan. Tap 6 o test VAN CHUA mo."
}

main "$@"
