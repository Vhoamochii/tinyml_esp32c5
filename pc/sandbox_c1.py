import numpy as np

# Cấu hình bài toán mô phỏng
N_FEATURES = 16
N_CLASSES = 3
N_SAMPLES_PER_TASK = 100
LEARNING_RATE = 0.05
EPOCHS_PER_TASK = 50

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, W, b):
    return softmax(np.dot(X, W) + b)

def train_step(X, y, W, b):
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

def run_experiment(replay_pct, replay_on=True):
    np.random.seed(42)
    
    # Tạo dữ liệu giả lập: Task A và Task B (dịch chuyển phân phối)
    X_A = np.random.randn(N_SAMPLES_PER_TASK, N_FEATURES).astype(np.float32)
    y_A = np.random.randint(0, N_CLASSES, N_SAMPLES_PER_TASK)
    
    X_B = np.random.randn(N_SAMPLES_PER_TASK, N_FEATURES).astype(np.float32) + 2.0 # Dịch chuyển mạnh hơn
    y_B = np.random.randint(0, N_CLASSES, N_SAMPLES_PER_TASK)

    # Khởi tạo mô hình
    W = np.zeros((N_FEATURES, N_CLASSES), dtype=np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    
    # 1. Huấn luyện Task A
    for _ in range(EPOCHS_PER_TASK):
        W, b = train_step(X_A, y_A, W, b)
        
    BA_A_before = evaluate(X_A, y_A, W, b)
    
    # --- SỬA LỖI TẠI ĐÂY ---
    # Khởi tạo bộ đệm Replay (R) chứa dữ liệu của Task A (Dữ liệu cũ)
    R_capacity = int(N_SAMPLES_PER_TASK * replay_pct)
    if replay_on and R_capacity > 0:
        # Lấy ngẫu nhiên R_capacity mẫu từ Task A
        indices = np.random.choice(N_SAMPLES_PER_TASK, R_capacity, replace=False)
        R_X = X_A[indices]
        R_y = y_A[indices]
    else:
        R_X = np.zeros((0, N_FEATURES), dtype=np.float32)
        R_y = np.zeros((0,), dtype=np.int32)

    # 2. Huấn luyện Task B (Học trực tuyến)
    for i in range(N_SAMPLES_PER_TASK):
        # Học mẫu mới (Task B)
        W, b = train_step(X_B[i:i+1], y_B[i:i+1], W, b)
        
        # Trộn mẫu cũ (Replay từ Task A)
        if replay_on and R_capacity > 0 and len(R_X) > 0:
            idx = np.random.randint(0, len(R_X))
            W, b = train_step(R_X[idx:idx+1], R_y[idx:idx+1], W, b)

    # Đánh giá
    BA_A_after = evaluate(X_A, y_A, W, b)
    BA_B_after = evaluate(X_B, y_B, W, b)
    
    forget = BA_A_before - BA_A_after
    
    return BA_A_before, BA_A_after, BA_B_after, forget

def main():
    configs = [
        ("Baseline (Replay OFF)", 0.0, False),
        ("r = 0% (ON nhưng rỗng)", 0.0, True),
        ("r = 25%", 0.25, True),
        ("r = 50%", 0.50, True)
    ]
    
    print(f"{'Cấu hình':<25} | {'BA A (Trước)':<15} | {'BA A (Sau)':<15} | {'BA B':<10} | {'Mức quên':<10}")
    print("-" * 85)
    
    for name, pct, on in configs:
        ba_before, ba_after, ba_b, forget = run_experiment(pct, on)
        tag = "[BASELINE]" if not on else "         "
        print(f"{tag} {name:<17} | {ba_before*100:>12.2f}% | {ba_after*100:>12.2f}% | {ba_b*100:>8.2f}% | {forget*100:>8.2f}%")

if __name__ == "__main__":
    main()
