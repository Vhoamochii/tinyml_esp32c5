#!/usr/bin/env python3
import os
import numpy as np
from vibdata.raw import PU_raw
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings("ignore")

FS = 64000
WINDOW_SIZE = 1024
STRIDE = 1024

TRAIN_BEARINGS = ["K001", "K002", "K003", "KI04", "KI16", "KI17", "KA04", "KA15"]
VAL_BEARINGS = ["K004", "KI14", "KA16"]
TEST_BEARINGS = ["K005", "K006", "KI18", "KI21", "KA22", "KA30"]

def extract_features(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    peak_to_peak = np.max(signal) - np.min(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    crest_factor = peak / (rms + 1e-7)

    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/FS)[:n//2]
    power = np.abs(yf[:n//2])**2

    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000)]
    band_powers = []
    for low, high in bands:
        idx = np.where((xf >= low) & (xf < high))[0]
        band_powers.append(np.sum(power[idx]) if len(idx) > 0 else 0.0)

    features = np.array([
        mean, std, rms, peak, peak_to_peak, skewness, kurt, crest_factor,
        band_powers[0], band_powers[1], band_powers[2], band_powers[3],
        rms / (mean + 1e-7),
        peak / (np.mean(np.abs(signal)) + 1e-7),
        peak / (rms**2 + 1e-7),
        -np.sum((signal**2) * np.log(signal**2 + 1e-12))
    ], dtype=np.float32)
    return features

def process_split(dataset, bearing_list, split_name):
    X_list = []
    y_list = []
    
    print(f"\n[INFO] Đang xử lý tập {split_name}...")
    meta_df = dataset.getMetaInfo()
    
    for i in range(len(dataset)):
        row = meta_df.iloc[i]
        bearing_id = str(row['bearing_code'])
        
        # Kiểm tra xem ổ này có thuộc tập hiện tại không
        if not any(b in bearing_id for b in bearing_list):
            continue
            
        # Xác định nhãn
        if bearing_id.startswith("K"):
            label = 0
        elif bearing_id.startswith("KI"):
            label = 1
        elif bearing_id.startswith("KA"):
            label = 2
        else:
            continue

        sample = dataset[i]
        signal_data = sample['signal']
        
        # Xử lý trường hợp signal là mảng object bọc mảng con
        if isinstance(signal_data, np.ndarray) and signal_data.dtype == object:
            signal = np.array(signal_data[0]).flatten()
        else:
            signal = np.array(signal_data).flatten()

        if len(signal) < WINDOW_SIZE:
            continue

        n_windows = (len(signal) - WINDOW_SIZE) // STRIDE + 1
        for j in range(n_windows):
            start = j * STRIDE
            window = signal[start:start + WINDOW_SIZE]
            if np.var(window) < 1e-12:
                continue
            feats = extract_features(window)
            X_list.append(feats)
            y_list.append(label)

    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        out_dir = os.path.expanduser("~/Work/Project_ESP32C5/pc/data")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez_compressed(out_path, X=X, y=y)
        print(f"[SAVE] Đã lưu {X.shape[0]} mẫu vào {out_path}")
    else:
        print(f"[WARN] Không có mẫu nào được tạo cho tập {split_name}")

def main():
    root_dir = os.path.expanduser("~/Work/Project_ESP32C5/pc/data/paderborn_vibdata")
    os.makedirs(root_dir, exist_ok=True)
    
    print("[INFO] Đang tải dữ liệu Paderborn...")
    dataset = PU_raw(root_dir, download=True)
    print(f"[INFO] Tổng số file trong dataset: {len(dataset)}")

    process_split(dataset, TRAIN_BEARINGS, "train")
    process_split(dataset, VAL_BEARINGS, "val")
    process_split(dataset, TEST_BEARINGS, "test")
    print("\n[SUCCESS] Hoàn tất quá trình T9!")

if __name__ == "__main__":
    main()
