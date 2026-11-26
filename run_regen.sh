#!/usr/bin/env bash
# Chay: bash run_regen.sh 2>&1 | tee regen.log
# Dung main() + return de cong fail chi dung script, KHONG dong terminal.
# TUYET DOI khong dan noi dung file nay tung dong vao shell tuong tac.

main() {
    cd ~/Work/Project_ESP32C5 || { echo "[STOP] khong cd duoc"; return 1; }

    echo "=== [T0] Kiem tra moi truong Python ==="
    if ! python3 -c "import numpy, scipy, vibdata" >/dev/null 2>&1; then
        echo "[ENV] thieu deps o moi truong hien tai, thu conda activate tinyml_pc"
        if command -v conda >/dev/null 2>&1; then
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate tinyml_pc
        fi
    fi
    if ! python3 -c "import numpy, scipy, vibdata" >/dev/null 2>&1; then
        echo "[STOP] van thieu numpy/scipy/vibdata. Hay chay: conda activate tinyml_pc"
        python3 -c "import sys; print('[ENV] python=', sys.executable)"
        return 1
    fi
    python3 -c "import sys, numpy, scipy; print('[ENV] python=', sys.executable, 'numpy=', numpy.__version__, 'scipy=', scipy.__version__)"

    echo "=== [T2a] Cach ly npz nhiem ==="
    mkdir -p pc/data/backup_20261126_contaminated
    cp -n pc/data/*.npz pc/data/backup_20261126_contaminated/
    sha256sum pc/data/*.npz > pc/data/backup_20261126_contaminated/SHA256.txt

    echo "=== [T2b] Test doc lap - phai exit 0 ==="
    python3 -u pc/sandbox_loader_test.py
    if [ $? -ne 0 ]; then echo "[STOP] sandbox FAIL - xem regen.log"; return 1; fi

    echo "=== [T2d] Loader that - counts phai PASS ==="
    python3 -u pc/data_loader_vibdata.py
    if [ $? -ne 0 ]; then echo "[STOP] counts lech checkpoint - xem regen.log"; return 1; fi

    echo "=== [T3a] Val CSV -> fixtures_val (khong dung fixtures/) ==="
    cd pc/pc-sim || { echo "[STOP] thieu pc/pc-sim"; return 1; }
    mkdir -p fixtures_val
    python3 make_feature_csv.py --npz ~/Work/Project_ESP32C5/pc/data/val.npz \
        --out fixtures_val/features_val.csv \
        --scales-out fixtures_val/feature_scales.csv \
        --norm-out fixtures_val/feature_norm.csv
    if [ $? -ne 0 ]; then echo "[STOP] val CSV FAIL"; return 1; fi
    echo "[GATE] norm phai 97f6b903..., scales phai 724a2b9d..."
    sha256sum fixtures_val/feature_norm.csv fixtures_val/feature_scales.csv

    echo "=== [T3b] Parity test ra /tmp - khong de fixtures ==="
    python3 make_feature_csv.py --npz ~/Work/Project_ESP32C5/pc/data/test.npz \
        --out /tmp/parity_test.csv --scales-out /tmp/parity_scales.csv --norm-out /tmp/parity_norm.csv
    if [ $? -ne 0 ]; then echo "[STOP] parity CSV FAIL"; return 1; fi
    echo "[GATE] parity_test phai 3c8550f0..."
    sha256sum /tmp/parity_test.csv
    sha256sum -c fixtures/FROZEN.sha256
    if [ $? -ne 0 ]; then echo "[STOP] FROZEN bi vi pham"; return 1; fi

    echo "=== [T4] Regression roi protocol_val ==="
    # F4: override target-dir ve cho ghi duoc
    export CARGO_TARGET_DIR="$HOME/Work/Project_ESP32C5/target_pc"
    mkdir -p "$CARGO_TARGET_DIR" || { echo "[STOP] khong tao duoc target dir"; return 1; }
    # F3: moi lenh cargo phai qua cong exit code
    cargo test --test table_harness -- --nocapture
    if [ $? -ne 0 ]; then echo "[STOP] table_harness FAIL"; return 1; fi
    cargo test --test protocol_val -- --ignored --nocapture
    if [ $? -ne 0 ]; then echo "[STOP] protocol_val FAIL"; return 1; fi
    if [ ! -f results/val_lock.csv ]; then echo "[STOP] thieu results/val_lock.csv"; return 1; fi
    echo "[DONE] toan bo cong PASS - r da khoa. Tap test VAN CHUA mo."
}

main "$@"
