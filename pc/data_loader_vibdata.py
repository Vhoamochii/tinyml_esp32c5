#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loader Paderborn -> npz 6 khoa (X, y, bearing, condition, record, win).

Sua loi ban cu:
  - Nhan startswith("K") dat truoc KI/KA nen moi o bi gan nhan 0.
  - Khong loc dieu kien -> nap ca 4 dieu kien, counts gap doi checkpoint.
  - Chi luu 2 khoa X, y -> make_feature_csv thieu bearing/condition/record/win.

Bat bien giu nguyen de bao toan parity voi checkpoint D7:
  extract_features, FS=64000, WINDOW=1024, STRIDE=1024, nguong var 1e-12,
  thu tu duyet theo meta cua vibdata. Counts ky vong:
  train 80117 / val 30046 / test 60111 / kb 30040.

Chi doc raw, khong cham vao fixtures/. Log prefix [LOADER].
"""
import os
import warnings

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from vibdata.raw import PU_raw

warnings.filterwarnings("ignore")

# Hang so trich xuat - KHONG DOI (parity D7)
FS = 64000
WINDOW_SIZE = 1024
STRIDE = 1024
N_FEATURES = 16

# Hai dieu kien ghep doi toc do: N15=1500rpm, N09=900rpm, cung M07 F10
CONDITIONS = ["N15_M07_F10", "N09_M07_F10"]

# Phan tap theo o vat ly (de cuong Bang 1)
TRAIN_BEARINGS = ["K001", "K002", "K003", "KI04", "KI16", "KI17", "KA04", "KA15"]
VAL_BEARINGS = ["K004", "KI14", "KA16"]
TEST_BEARINGS = ["K005", "K006", "KI18", "KI21", "KA22", "KA30"]
KB_BEARINGS = ["KB23", "KB24", "KB27"]  # hong ca hai vong, thong ke rieng

# Checkpoint specs.md - cong nghiem thu counts
EXPECTED = {"train": 80117, "val": 30046, "test": 60111, "kb": 30040}


def label_for(bearing):
    """Kiem tra tien to dai truoc: 'KI14'.startswith('K') cung True."""
    if bearing.startswith("KI"):
        return 1  # loi vong trong
    if bearing.startswith("KA"):
        return 2  # loi vong ngoai
    if bearing.startswith("KB"):
        return 3  # hong ca hai vong
    if bearing.startswith("K"):
        return 0  # khoe
    return None  # ma la -> bao loi, khong gan nham


def parse_condition_record(file_name):
    """'N15_M07_F10_K004_13.mat' -> ('N15_M07_F10', 13). Hong -> (None, None)."""
    stem = os.path.basename(str(file_name))
    if stem.endswith(".mat"):
        stem = stem[:-4]
    parts = stem.split("_")
    if len(parts) < 5:
        return None, None
    cond = "_".join(parts[:3])
    try:
        rec = int(parts[-1])
    except ValueError:
        return None, None
    return cond, rec


def window_count(n):
    """So cua so nguyen ven trong ban ghi n mau. n < WINDOW -> 0 (bo qua)."""
    if n < WINDOW_SIZE:
        return 0
    return (n - WINDOW_SIZE) // STRIDE + 1


def extract_features(signal):
    """16 dac trung - GIU NGUYEN tung dong so voi ban cu de parity."""
    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    peak_to_peak = np.max(signal) - np.min(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    crest_factor = peak / (rms + 1e-7)

    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / FS)[:n // 2]
    power = np.abs(yf[:n // 2]) ** 2

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
        peak / (rms ** 2 + 1e-7),
        -np.sum((signal ** 2) * np.log(signal ** 2 + 1e-12))
    ], dtype=np.float32)
    return features


def build_split(dataset, meta_records, bearings, conditions):
    """Duyet theo thu tu meta, loc chinh xac ma o + dieu kien, tra 6 mang."""
    X, y, b, c, rec, win = [], [], [], [], [], []
    bset = set(bearings)
    allowed = set(conditions)
    for i in range(len(dataset)):
        row = meta_records[i]
        bearing = str(row.get("bearing_code", ""))
        if bearing not in bset:  # khop chinh xac, khong substring
            continue
        cond, r = parse_condition_record(row.get("file_name", ""))
        if cond is None or cond not in allowed:
            continue
        lab = label_for(bearing)
        if lab is None:
            print(f"[LOADER][WARN] ma o la '{bearing}', bo qua")
            continue
        sig = dataset[i]["signal"]
        # vibdata co the boc mang con trong object array
        if isinstance(sig, np.ndarray) and sig.dtype == object:
            sig = np.array(sig[0]).flatten()
        else:
            sig = np.array(sig).flatten()
        nw = window_count(len(sig))
        for j in range(nw):
            w = sig[j * STRIDE: j * STRIDE + WINDOW_SIZE]
            if np.var(w) < 1e-12:  # cua so chet - giu nguong cu
                continue
            X.append(extract_features(w))
            y.append(lab)
            b.append(bearing)
            c.append(cond)
            rec.append(r)
            win.append(j)
    return {
        "X": np.array(X, dtype=np.float32).reshape(-1, N_FEATURES),
        "y": np.array(y, dtype=np.int64),
        "bearing": np.array(b, dtype=str),
        "condition": np.array(c, dtype=str),
        "record": np.array(rec, dtype=np.int64),
        "win": np.array(win, dtype=np.int64),
    }


def main():
    root = os.path.expanduser("~/Work/Project_ESP32C5/pc/data/paderborn_vibdata")
    out_dir = os.path.expanduser("~/Work/Project_ESP32C5/pc/data")
    os.makedirs(out_dir, exist_ok=True)

    print("[LOADER] nap PU_raw (khong download, raw da co san)...")
    dataset = PU_raw(root_dir=root)
    meta_df = dataset.getMetaInfo()
    if len(meta_df) != len(dataset):
        raise SystemExit("[LOADER][ERROR] meta va dataset lech so dong")
    meta_records = meta_df.to_dict("records")
    print(f"[LOADER] so file={len(dataset)} dieu_kien_loc={CONDITIONS}")

    splits = {
        "train": TRAIN_BEARINGS,
        "val": VAL_BEARINGS,
        "test": TEST_BEARINGS,
        "kb": KB_BEARINGS,
    }
    all_ok = True
    for name, bearings in splits.items():
        d = build_split(dataset, meta_records, bearings, CONDITIONS)
        n = len(d["y"])
        exp = EXPECTED[name]
        ok = (n == exp)
        all_ok = all_ok and ok
        print(f"[LOADER][{name}] rows={n} expected={exp} {'PASS' if ok else 'FAIL'}")
        if n > 0:
            v, cnt = np.unique(d["y"], return_counts=True)
            print(f"[LOADER][{name}] nhan={dict(zip(v.tolist(), cnt.tolist()))} "
                  f"o={sorted(set(d['bearing'].tolist()))}")
            out_path = os.path.join(out_dir, f"{name}.npz")
            np.savez_compressed(out_path, **d)
            print(f"[LOADER][SAVE] {out_path} (6 khoa)")
        else:
            print(f"[LOADER][{name}] TRONG - khong ghi")
    print(f"[LOADER][DONE] counts {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        raise SystemExit("[LOADER][ERROR] counts lech checkpoint - DUNG, khong chay CSV")


if __name__ == "__main__":
    main()
