import json
import hashlib
import os
import sys

CONFIG_FILE = "config_lock.json"

def generate_manifest():
    """Tạo file cấu hình khóa và tính mã SHA256."""
    # Các thông số đã chốt từ C1, C2, C4
    manifest = {
        "version": "v1.0.0",
        "dataset": "Paderborn",
        "operating_conditions": ["N15_M07_F10", "N09_M07_F10"],
        "features": 16,
        "classes": 3,
        "architecture": "LastLayer_Softmax",
        "learning_rate": 0.05,
        "K_capacity": 100,           # Tổng số ô nhớ
        "Q_capacity": 20,            # Kích thước hàng đợi Q
        "r_selected_pct": 25,        # Đã chọn ở C2
        "strategy": "interleaved",   # Đã chọn ở C4
        "store_type": "INT8",
        "stride": 1024,
        "window_size": 1024,
        "deadline_T_ms": 16,         # 1024 mẫu / 64kHz = 16ms
        "seeds": [42, 43, 44, 45, 46], # 5 hạt giống theo đề cương
        "test_bearings": ["K005", "K006", "KI18", "KI21", "KA22", "KA30"]
    }

    # Ghi ra file
    with open(CONFIG_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

    # Tính mã băm SHA256
    with open(CONFIG_FILE, "rb") as f:
        file_bytes = f.read()
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    print(f"[C5][MANIFEST] Đã tạo file {CONFIG_FILE}")
    print(f"[C5][SHA256]   Hash: {sha256_hash}")
    
    # Lưu hash ra một file text riêng để dễ so sánh
    with open("config_lock.sha256", "w") as f:
        f.write(sha256_hash)
    
    return sha256_hash

def verify_and_unlock():
    """Giả lập quy trình mở tập Test: Chỉ mở khi hash khớp."""
    print("\n[C5][VERIFY] Đang kiểm tra tính toàn vẹn của cấu hình...")
    
    if not os.path.exists(CONFIG_FILE) or not os.path.exists("config_lock.sha256"):
        print("[C5][ERROR] Không tìm thấy file cấu hình hoặc file hash.")
        sys.exit(1)
        
    with open("config_lock.sha256", "r") as f:
        expected_hash = f.read().strip()
        
    with open(CONFIG_FILE, "rb") as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()
        
    if current_hash == expected_hash:
        print("[C5][SUCCESS] Hash khớp. Tập kiểm thử (Test Set) được phép MỞ.")
        print("-> Bắt đầu chạy 6 ổ kiểm thử với 5 seeds...")
        # Code chạy test thật sẽ nằm ở đây
    else:
        print("[C5][FAIL] CẢNH BÁO: File cấu hình đã bị sửa đổi trái phép!")
        print(f"  Expected: {expected_hash}")
        print(f"  Current : {current_hash}")
        print("-> TỪ CHỐI mở tập kiểm thử để tránh thiên lệch kết quả.")
        sys.exit(1)

def tamper_test():
    """Hàm test giả lập việc ai đó cố tình sửa r từ 25% lên 50%."""
    print("\n[C5][TAMPER] Giả lập hacker sửa r_selected_pct từ 25 lên 50...")
    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)
    
    data["r_selected_pct"] = 50
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)
        
    # Chạy verify lại
    verify_and_unlock()

if __name__ == "__main__":
    # Bước 1: Tạo manifest và hash
    generate_manifest()
    
    # Bước 2: Verify lần 1 (hợp lệ)
    verify_and_unlock()
    
    # Bước 3: Thử sửa file và verify lần 2 (báo lỗi)
    tamper_test()
