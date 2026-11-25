#!/usr/bin/env python3
import numpy as np
import os

TRAIN_BEARINGS = {"K001", "K002", "K003", "KI04", "KI16", "KI17", "KA04", "KA15"}
VAL_BEARINGS = {"K004", "KI14", "KA16"}
TEST_BEARINGS = {"K005", "K006", "KI18", "KI21", "KA22", "KA30"}

def check_overlap():
    print("\n=== [B0] KIỂM TRA RÒ RỈ DỮ LIỆU (OVERLAP) ===")
    overlap_train_val = TRAIN_BEARINGS.intersection(VAL_BEARINGS)
    overlap_train_test = TRAIN_BEARINGS.intersection(TEST_BEARINGS)
    overlap_val_test = VAL_BEARINGS.intersection(TEST_BEARINGS)
    
    is_clean = True
    if overlap_train_val:
        print(f"[FAIL] Trùng ổ giữa Train và Val: {overlap_train_val}")
        is_clean = False
    if overlap_train_test:
        print(f"[FAIL] Trùng ổ giữa Train và Test: {overlap_train_test}")
        is_clean = False
    if overlap_val_test:
        print(f"[FAIL] Trùng ổ giữa Val và Test: {overlap_val_test}")
        is_clean = False
        
    if is_clean:
        print("[PASS] 3 tập ổ hoàn toàn rời nhau (không rò rỉ dữ liệu vật lý).")
    return is_clean

def check_npz_file(filepath, split_name):
    if not os.path.exists(filepath):
        print(f"[WARN] Không tìm thấy file {filepath}. Bỏ qua kiểm tra.")
        return False
        
    print(f"\n--- Kiểm tra tập {split_name.upper()} ({filepath}) ---")
    data = np.load(filepath)
    X, y = data['X'], data['y']
    
    print(f"Kích thước X: {X.shape} | Kích thước y: {y.shape}")
    
    if np.isnan(X).any() or np.isinf(X).any():
        print("[FAIL] Phát hiện giá trị NaN hoặc Inf trong đặc trưng X.")
        return False
    print("[PASS] Không có NaN/Inf trong X.")
    
    unique, counts = np.unique(y, return_counts=True)
    label_map = {0: "Healthy", 1: "Inner", 2: "Outer"}
    dist = {label_map.get(int(k), str(k)): int(v) for k, v in zip(unique, counts)}
    print(f"Phân bố nhãn: {dist}")
    
    max_val, min_val = np.max(X), np.min(X)
    print(f"Giá trị đặc trưng: Min={min_val:.4f}, Max={max_val:.4f}")
    
    return True

if __name__ == "__main__":
    print("=== BẮT ĐẦU KIỂM TRA DỮ LIỆU (B0) ===")
    
    if not check_overlap():
        print("\n[ACTION REQUIRED] Cần retrain lại dữ liệu vì có rò rỉ!")
        exit(1)
        
    data_dir = os.path.expanduser("~/Work/Project_ESP32C5/pc/data")
    check_npz_file(os.path.join(data_dir, "train.npz"), "train")
    check_npz_file(os.path.join(data_dir, "val.npz"), "val")
    check_npz_file(os.path.join(data_dir, "test.npz"), "test")
    
    print("\n=== HOÀN TẤT B0 ===")
