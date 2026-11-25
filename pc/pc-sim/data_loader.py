#!/usr/bin/env python3
"""
Script xử lý dữ liệu Paderborn Bearing Dataset.
- Đọc file .mat, lấy tín hiệu rung (vibration_1).
- Cắt cửa sổ (window) và trích xuất 16 đặc trưng.
- Chia tập theo danh sách ổ vật lý trong đề cương.
- Lưu ra file .npz.
"""
import os
import glob
import numpy as np
import scipy.io as sio
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

# Cấu hình
FS = 64000          # Tần số lấy mẫu (Hz)
WINDOW_SIZE = 1024  # Kích thước cửa sổ (mẫu)
STRIDE = 1024       # Bước trượt (không chồng lấn)
N_FEATURES = 16     # Số đặc trưng

# Danh sách ổ theo đề cương
TRAIN_BEARINGS = ["K001", "K002", "K003", "KI04", "KI16", "KI17", "KA04", "KA15"]
VAL_BEARINGS = ["K004", "KI14", "KA16"]
TEST_BEARINGS = ["K005", "K006", "KI18", "KI21", "KA22", "KA30"]

# Điều kiện vận hành (N15 = 1500 rpm, M07 = 0.7 Nm, F10 = 1000 N)
# Sẽ xử lý cả N15 và N09 (900 rpm) để mô phỏng đổi tốc độ
OPERATING_CONDITIONS = ["N15_M07_F10", "N09_M07_F10"]

def extract_features(signal):
    """Trích xuất 16 đặc trưng từ một cửa sổ tín hiệu."""
    # 1. Thống kê thời gian (8 đặc trưng)
    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    peak_to_peak = np.max(signal) - np.min(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    crest_factor = peak / (rms + 1e-7)  # Tránh chia cho 0

    # 2. Đặc trưng tần số (8 đặc trưng: 4 dải tần)
    # Lấy FFT, tính công suất trong 4 dải tần
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/FS)[:n//2]
    power = np.abs(yf[:n//2])**2

    # Chia 4 dải: 0-500, 500-1000, 1000-2000, 2000-4000 Hz
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000)]
    band_powers = []
    for low, high in bands:
        idx = np.where((xf >= low) & (xf < high))[0]
        if len(idx) > 0:
            band_powers.append(np.sum(power[idx]))
        else:
            band_powers.append(0.0)

    # Gộp thành vector 16 chiều
    features = np.array([
        mean, std, rms, peak, peak_to_peak, skewness, kurt, crest_factor,
        band_powers[0], band_powers[1], band_powers[2], band_powers[3],
        # Thêm 4 đặc trưng khác để đủ 16: form factor, impulse factor, margin factor, entropy
        rms / (mean + 1e-7),
        peak / (np.mean(np.abs(signal)) + 1e-7),
        peak / (rms**2 + 1e-7),
        -np.sum((signal**2) * np.log(signal**2 + 1e-12))  # entropy xấp xỉ
    ], dtype=np.float32)

    return features

def load_bearing_data(bearing_code, oc_list, data_dir):
    """Đọc tất cả file .mat của một ổ, trích xuất đặc trưng."""
    X_list = []
    y_list = []

    # Xác định nhãn lớp dựa vào tiền tố mã ổ
    if bearing_code.startswith("K"):
        label = 0  # Healthy
    elif bearing_code.startswith("KI"):
        label = 1  # Inner Race
    elif bearing_code.startswith("KA"):
        label = 2  # Outer Race
    else:
        raise ValueError(f"Unknown bearing code: {bearing_code}")

    for oc in oc_list:
        # Tìm tất cả file .mat của ổ này trong điều kiện vận hành
        pattern = os.path.join(data_dir, oc, f"{oc}_{bearing_code}_*.mat")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"[WARN] Không tìm thấy file cho {bearing_code} / {oc}")
            continue

        for fpath in files:
            try:
                mat = sio.loadmat(fpath)
                # Lấy tín hiệu rung (vibration_1)
                # Cấu trúc: Y.vibration_1 (có thể là mảng)
                # Tùy phiên bản dataset, có thể là mat['Y'][0,0]['vibration_1']
                if 'Y' in mat:
                    vibration = mat['Y'][0,0]['vibration_1'].flatten()
                elif 'X' in mat:
                    vibration = mat['X'].flatten()
                else:
                    print(f"[WARN] Không tìm thấy trường 'Y' hoặc 'X' trong {fpath}")
                    continue

                # Cắt cửa sổ
                n_windows = (len(vibration) - WINDOW_SIZE) // STRIDE + 1
                for i in range(n_windows):
                    start = i * STRIDE
                    window = vibration[start:start + WINDOW_SIZE]

                    # Bỏ qua cửa sổ có phương sai quá nhỏ (tín hiệu chết)
                    if np.var(window) < 1e-12:
                        continue

                    feats = extract_features(window)
                    X_list.append(feats)
                    y_list.append(label)

            except Exception as e:
                print(f"[ERROR] Lỗi đọc {fpath}: {e}")
                continue

    return np.array(X_list), np.array(y_list)

def main():
    # Đường dẫn đến thư mục chứa dữ liệu Paderborn đã tải
    # Anh/chị cần tải dữ liệu từ trang chủ Paderborn và giải nén vào đây
    DATA_DIR = os.path.expanduser("~/Work/Project_ESP32C5/pc/data/paderborn")

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Không tìm thấy thư mục dữ liệu: {DATA_DIR}")
        print("Vui lòng tải dữ liệu Paderborn từ https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download")
        print("Giải nén các file .mat vào thư mục trên, mỗi điều kiện vận hành là một thư mục con (N15_M07_F10, N09_M07_F10, ...)")
        return

    # Xử lý từng tập
    splits = {
        "train": TRAIN_BEARINGS,
        "val": VAL_BEARINGS,
        "test": TEST_BEARINGS,
    }

    for split_name, bearing_list in splits.items():
        print(f"\n[INFO] Đang xử lý tập {split_name} ({len(bearing_list)} ổ)...")
        X_split = []
        y_split = []

        for bearing in bearing_list:
            print(f"  -> Ổ {bearing}")
            X_b, y_b = load_bearing_data(bearing, OPERATING_CONDITIONS, DATA_DIR)
            if len(X_b) > 0:
                X_split.append(X_b)
                y_split.append(y_b)
                print(f"     + {len(X_b)} cửa sổ")
            else:
                print(f"     - Không có dữ liệu")

        if X_split:
            X = np.vstack(X_split)
            y = np.concatenate(y_split)
            out_path = os.path.join(os.path.dirname(DATA_DIR), f"{split_name}.npz")
            np.savez_compressed(out_path, X=X, y=y)
            print(f"[SAVE] Đã lưu {X.shape[0]} mẫu vào {out_path}")
        else:
            print(f"[WARN] Không có dữ liệu cho tập {split_name}")

if __name__ == "__main__":
    main()
