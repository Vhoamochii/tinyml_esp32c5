import numpy as np

# Cấu hình
N_FEATURES = 16
N_CLASSES = 3
N_SAMPLES_PER_TASK = 300
LEARNING_RATE = 0.1
EPOCHS_PER_TASK = 100

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
    dW = np.dot(X.T, dz)
    db = np.sum(dz, axis=0)
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    return W, b

def evaluate(X, y, W, b):
    probs = forward(X, W, b)
    preds = np.argmax(probs, axis=1)
    accs = []
    for c in range(N_CLASSES):
        idx = (y == c)
        if np.sum(idx) > 0:
            accs.append(np.mean(preds[idx] == c))
    return np.mean(accs)

def generate_data(n_samples, shift=0.0):
    """Tạo dữ liệu có quy luật: Mỗi lớp có tâm phân phối khác nhau."""
    X = []
    y = []
    for c in range(N_CLASSES):
        # Tâm của mỗi lớp: -2, 0, 2 (cộng thêm độ dịch chuyển shift)
        center = (c * 2.0 - 2.0) + shift
        # Tạo ngẫu nhiên quanh tâm đó
        X_c = np.random.randn(n_samples // N_CLASSES, N_FEATURES).astype(np.float32) + center
        y_c = np.full(n_samples // N_CLASSES, c, dtype=np.int32)
        X.append(X_c)
        y.append(y_c)
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Xáo trộn dữ liệu
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

def run_experiment(r_pct, replay_on=True, strategy="interleaved"):
    np.random.seed(42)
    
    # Task A (Tốc độ cũ)
    X_A, y_A = generate_data(N_SAMPLES_PER_TASK, shift=0.0)
    # Task B (Tốc độ mới - bị dịch phân phối nhẹ để giả lập drift)
    X_B, y_B = generate_data(N_SAMPLES_PER_TASK, shift=1.0)

    W = np.zeros((N_FEATURES, N_CLASSES), dtype=np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    
    R_capacity = int(N_SAMPLES_PER_TASK * r_pct)
    R_X = np.zeros((0, N_FEATURES), dtype=np.float32)
    R_y = np.zeros((0,), dtype=np.int32)
    last_learned_R_idx = -1
    
    # 1. Học Task A
    for _ in range(EPOCHS_PER_TASK):
        W, b = train_batch(X_A, y_A, W, b)
    BA_A_before = evaluate(X_A, y_A, W, b)
    
    # 2. Học Task B (trực tuyến - từng mẫu một)
    for i in range(N_SAMPLES_PER_TASK):
        # Gom batch: 1 mẫu mới + 1 mẫu replay (nếu có)
        batch_X = [X_B[i:i+1]]
        batch_y = [y_B[i:i+1]]
        
        if replay_on and R_capacity > 0 and len(R_X) > 0:
            if strategy == "interleaved":
                # Xen kẽ: luôn lấy 1 mẫu replay
                if len(R_X) > 1:
                    idx = np.random.randint(0, len(R_X))
                    while idx == last_learned_R_idx:
                        idx = np.random.randint(0, len(R_X))
                else:
                    idx = 0
                batch_X.append(R_X[idx:idx+1])
                batch_y.append(R_y[idx:idx+1])
                last_learned_R_idx = idx
                
            elif strategy == "q_first":
                # Ưu tiên Q: Chỉ replay định kỳ (ví dụ mỗi 5 mẫu) để giả lập việc Q được ưu tiên
                if i % 5 == 0 and len(R_X) > 1:
                    idx = np.random.randint(0, len(R_X))
                    while idx == last_learned_R_idx:
                        idx = np.random.randint(0, len(R_X))
                    batch_X.append(R_X[idx:idx+1])
                    batch_y.append(R_y[idx:idx+1])
                    last_learned_R_idx = idx
            
        # Cập nhật mạng với batch (gộp mẫu mới + mẫu cũ)
        X_batch = np.vstack(batch_X)
        y_batch = np.concatenate(batch_y)
        W, b = train_batch(X_batch, y_batch, W, b)
        
        # Lưu mẫu mới vào R
        if replay_on and R_capacity > 0:
            if len(R_X) < R_capacity:
                R_X = np.vstack([R_X, X_B[i:i+1]])
                R_y = np.append(R_y, y_B[i])
            else:
                R_X = np.vstack([R_X[1:], X_B[i:i+1]])
                R_y = np.append(R_y[1:], y_B[i])

    BA_A_after = evaluate(X_A, y_A, W, b)
    BA_B_after = evaluate(X_B, y_B, W, b)
    forget = BA_A_before - BA_A_after
    return BA_B_after, forget

def main():
    print("--- BƯỚC 1: SWEEP R (C1) ---")
    results = {}
    for r in [0.0, 0.25, 0.50]:
        ba, forget = run_experiment(r, replay_on=True, strategy="interleaved")
        results[r] = (ba, forget)
        print(f"r={int(r*100):2d}% -> BA={ba*100:.2f}%, Forget={forget*100:.2f}%")
    
    forget_r0 = results[0.0][1]
    
    print("\n--- BƯỚC 2: KHÓA R (C2) ---")
    # Luật chọn: forget <= forget_r0 -> BA max -> tie: forget thấp -> r nhỏ
    candidates = [r for r, (ba, f) in results.items() if f <= forget_r0]
    candidates.sort(key=lambda r: (-results[r][0], results[r][1], r))
    
    selected_r = candidates[0]
    print(f"[SELECT] Chọn r = {int(selected_r*100)}% vì:")
    print(f"  - Forget ({results[selected_r][1]*100:.2f}%) <= Forget_r0 ({forget_r0*100:.2f}%)")
    print(f"  - BA đạt cao nhất ({results[selected_r][0]*100:.2f}%) trong các ứng viên.")
    
    print("\n--- BƯỚC 3: SO SÁNH LỊCH REPLAY (C4) ---")
    print(f"Cố định r = {int(selected_r*100)}%")
    
    ba_inter, forget_inter = run_experiment(selected_r, replay_on=True, strategy="interleaved")
    print(f"[REPLAY] Xen kẽ   -> BA={ba_inter*100:.2f}%, Forget={forget_inter*100:.2f}%")
    
    ba_q, forget_q = run_experiment(selected_r, replay_on=True, strategy="q_first")
    print(f"[REPLAY] Ưu tiên Q -> BA={ba_q*100:.2f}%, Forget={forget_q*100:.2f}%")

if __name__ == "__main__":
    main()
