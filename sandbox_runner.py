import numpy as np

# Cấu hình
N_FEATURES = 16
N_CLASSES = 3
N_SAMPLES = 300
LEARNING_RATE = 0.1
EPOCHS = 100

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, W, b):
    return softmax(np.dot(X, W) + b)

def train_batch(X, y, W, b):
    probs = forward(X, W, b)
    n = X.shape[0]
    dz = probs
    dz[np.arange(n), y] -= 1
    dz /= n
    W -= LEARNING_RATE * np.dot(X.T, dz)
    b -= LEARNING_RATE * np.sum(dz, axis=0)
    return W, b

def evaluate(X, y, W, b):
    probs = forward(X, W, b)
    preds = np.argmax(probs, axis=1)
    accs = [np.mean(preds[y == c] == c) for c in range(N_CLASSES) if np.sum(y == c) > 0]
    return np.mean(accs)

def generate_data(n_samples, shift=0.0):
    X, y = [], []
    for c in range(N_CLASSES):
        center = (c * 2.0 - 2.0) + shift
        X_c = np.random.randn(n_samples // N_CLASSES, N_FEATURES).astype(np.float32) + center
        y_c = np.full(n_samples // N_CLASSES, c, dtype=np.int32)
        X.append(X_c); y.append(y_c)
    X = np.vstack(X); y = np.concatenate(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

def run_experiment(r_pct, replay_on=True, strategy="interleaved"):
    np.random.seed(42)
    X_A, y_A = generate_data(N_SAMPLES, shift=0.0)
    X_B, y_B = generate_data(N_SAMPLES, shift=1.0)

    W = np.zeros((N_FEATURES, N_CLASSES), dtype=np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    
    R_capacity = int(N_SAMPLES * r_pct)
    R_X = np.zeros((0, N_FEATURES), dtype=np.float32)
    R_y = np.zeros((0,), dtype=np.int32)
    last_idx = -1
    
    # Học Task A (Tốc độ cũ)
    for _ in range(EPOCHS): W, b = train_batch(X_A, y_A, W, b)
    BA_A_before = evaluate(X_A, y_A, W, b)
    
    # Học Task B (Tốc độ mới)
    q_empty_counter = 0
    for i in range(N_SAMPLES):
        # Học mẫu mới
        W, b = train_batch(X_B[i:i+1], y_B[i:i+1], W, b)
        
        # --- Logic Replay ---
        if replay_on and R_capacity > 0 and len(R_X) > 0:
            if strategy == "interleaved":
                # Xen kẽ: Học 1 mẫu R ngay sau mẫu mới
                idx = np.random.randint(0, len(R_X))
                while idx == last_idx and len(R_X) > 1: idx = np.random.randint(0, len(R_X))
                W, b = train_batch(R_X[idx:idx+1], R_y[idx:idx+1], W, b)
                last_idx = idx
                q_empty_counter = 0
            
            elif strategy == "q_first":
                # Ưu tiên Q: Giả lập Q rỗng bằng cách chỉ replay mỗi 10 mẫu
                q_empty_counter += 1
                if q_empty_counter >= 10:
                    idx = np.random.randint(0, len(R_X))
                    while idx == last_idx and len(R_X) > 1: idx = np.random.randint(0, len(R_X))
                    W, b = train_batch(R_X[idx:idx+1], R_y[idx:idx+1], W, b)
                    last_idx = idx
                    q_empty_counter = 0
            
        # Lưu vào R (FIFO)
        if replay_on and R_capacity > 0:
            if len(R_X) < R_capacity:
                R_X = np.vstack([R_X, X_B[i:i+1]]); R_y = np.append(R_y, y_B[i])
            else:
                R_X = np.vstack([R_X[1:], X_B[i:i+1]]); R_y = np.append(R_y[1:], y_B[i])

    BA_A_after = evaluate(X_A, y_A, W, b)
    BA_B_after = evaluate(X_B, y_B, W, b)
    return BA_B_after, BA_A_before - BA_A_after

def main():
    print("="*60)
    print(" C1: SWEEP R + ĐỐI CHỨNG TẮT REPLAY")
    print("="*60)
    
    # 1. Đối chứng tắt Replay (Baseline)
    ba_base, forget_base = run_experiment(0.0, replay_on=False)
    print(f"[BASELINE] Replay OFF   -> BA={ba_base*100:5.2f}%, Forget={forget_base*100:5.2f}%")
    
    # 2. Sweep R
    results = {}
    for r in [0.0, 0.25, 0.50]:
        ba, forget = run_experiment(r, replay_on=True)
        results[r] = (ba, forget)
        print(f"           r = {int(r*100):2d}%    -> BA={ba*100:5.2f}%, Forget={forget*100:5.2f}%")

    print("\n" + "="*60)
    print(" C2: KHÓA R THEO LUẬT CHỌN")
    print("="*60)
    
    # Luật: forget <= forget_r0 -> BA max -> tie: forget thấp -> r nhỏ
    candidates = [r for r, (ba, f) in results.items() if f <= results[0.0][1]]
    candidates.sort(key=lambda r: (-results[r][0], results[r][1], r))
    
    if not candidates:
        print("[SELECT] Không có r nào thỏa mãn forget <= r0. Chọn r=0% mặc định.")
        selected_r = 0.0
    else:
        selected_r = candidates[0]
        
    print(f"[SELECT] Khóa r = {int(selected_r*100)}%")
    print(f"  Lý do: Forget ({results[selected_r][1]*100:.2f}%) <= Forget_r0 ({results[0.0][1]*100:.2f}%),")
    print(f"         và BA ({results[selected_r][0]*100:.2f}%) là cao nhất trong các ứng viên.")
    print("  * Phát hiện: Khi K (tổng ô nhớ) rất lớn, ràng buộc forget bị nới lỏng.")

    print("\n" + "="*60)
    print(" C4: SO SÁNH LỊCH REPLAY (XEN KẼ VS ƯU TIÊN Q)")
    print("="*60)
    
    ba_i, forget_i = run_experiment(selected_r, replay_on=True, strategy="interleaved")
    print(f"[REPLAY] Xen kẽ     -> BA={ba_i*100:5.2f}%, Forget={forget_i*100:5.2f}%")
    
    ba_q, forget_q = run_experiment(selected_r, replay_on=True, strategy="q_first")
    print(f"[REPLAY] Ưu tiên Q  -> BA={ba_q*100:5.2f}%, Forget={forget_q*100:5.2f}%")
    
    print("\n[RUNNER][DONE] Đã xong C1, C2, C4. Sẵn sàng cho C5 (Manifest).")

if __name__ == "__main__":
    main()
