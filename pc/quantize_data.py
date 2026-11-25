#!/usr/bin/env python3
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def main():
    data_dir = os.path.expanduser("~/Work/Project_ESP32C5/pc/data")
    
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    test_path = os.path.join(data_dir, "test.npz")
    
    print("=== [B6] BẮT ĐẦU CHUẨN HÓA VÀ LƯỢNG TỬ HÓA (CÓ CLIPPING) ===")
    
    # 1. Load tập Train
    train_data = np.load(train_path)
    X_train_fp32 = train_data['X']
    y_train = train_data['y']
    
    # 2. Chuẩn hóa dữ liệu (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fp32)
    
    # 3. CẮT NGƯỠNG (CLIPPING) để loại bỏ ảnh hưởng của outliers
    # Chọn ngưỡng [-6, 6] để bao phủ 99.99% dữ liệu phân bố chuẩn
    CLIP_MIN = -6.0
    CLIP_MAX = 6.0
    
    X_train_clipped = np.clip(X_train_scaled, CLIP_MIN, CLIP_MAX)
    
    print(f"[INFO] Dải giá trị sau khi cắt ngưỡng: Min={np.min(X_train_clipped):.4f}, Max={np.max(X_train_clipped):.4f}")
    
    # 4. Tính tham số lượng tử hóa (Dựa trên dữ liệu đã cắt ngưỡng)
    min_val = CLIP_MIN
    max_val = CLIP_MAX
    
    scale = (max_val - min_val) / 255.0
    zero_point = -128 - round(min_val / scale)
    zero_point = int(np.clip(zero_point, -128, 127))
    
    print(f"[INFO] Tham số lượng tử hóa INT8:")
    print(f"  - Scale: {scale:.6f}, ZeroPoint: {zero_point}")
    
    # Lưu tham số
    np.savez_compressed(
        os.path.join(data_dir, "quant_params.npz"),
        scale=scale, zero_point=zero_point,
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
        clip_min=CLIP_MIN, clip_max=CLIP_MAX
    )

    # Hàm phụ trợ lượng tử hóa
    def process_and_save(X_fp32, y, name, is_scaled=False):
        if not is_scaled:
            X_scaled = scaler.transform(X_fp32)
        else:
            X_scaled = X_fp32
            
        # Cắt ngưỡng
        X_clipped = np.clip(X_scaled, CLIP_MIN, CLIP_MAX)
        
        # Lượng tử hóa
        X_int8 = np.round(X_clipped / scale) + zero_point
        X_int8 = np.clip(X_int8, -128, 127).astype(np.int8)
        
        # Giải lượng tử hóa (tính sai lệch dựa trên dữ liệu đã cắt)
        X_dequantized = (X_int8.astype(np.float32) - zero_point) * scale
        
        mae = np.mean(np.abs(X_clipped - X_dequantized))
        max_err = np.max(np.abs(X_clipped - X_dequantized))
        
        print(f"\n[TEST][B6][{name.upper()}] Kích thước: {X_int8.shape}")
        print(f"  - Mean Absolute Error (MAE): {mae:.6f}")
        print(f"  - Max Absolute Error: {max_err:.6f}")
        
        out_path = os.path.join(data_dir, f"{name}_int8.npz")
        np.savez_compressed(out_path, X=X_int8, y=y)

    # Xử lý Train
    process_and_save(X_train_scaled, y_train, "train", is_scaled=True)
    
    # Xử lý Val
    val_data = np.load(val_path)
    process_and_save(val_data['X'], val_data['y'], "val")
    
    # Xử lý Test
    test_data = np.load(test_path)
    process_and_save(test_data['X'], test_data['y'], "test")
    
    print("\n=== HOÀN TẤT B6 ===")

if __name__ == "__main__":
    main()
