import numpy as np
import sys

# Cấu hình mạng
INPUT_SIZE = 2
HIDDEN_SIZE = 8
OUTPUT_SIZE = 1
EPOCHS = 10000
LEARNING_RATE = 0.1

# Dữ liệu XOR: Input (0, 1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

# Mục tiêu: Map 0 thành -1, 1 thành 1 để đồng bộ với hàm Tanh
Y = np.array([[-1], [1], [1], [-1]], dtype=np.float32)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def train():
    np.random.seed(42)
    W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.5
    b1 = np.zeros((1, HIDDEN_SIZE), dtype=np.float32)
    
    W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE).astype(np.float32) * 0.5
    b2 = np.zeros((1, OUTPUT_SIZE), dtype=np.float32)

    for epoch in range(EPOCHS):
        Z1 = np.dot(X, W1) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = tanh(Z2)
        
        loss = np.mean((A2 - Y)**2)
        
        dZ2 = (A2 - Y) * tanh_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, W2.T) * tanh_derivative(Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        W2 -= LEARNING_RATE * dW2
        b2 -= LEARNING_RATE * db2
        W1 -= LEARNING_RATE * dW1
        b1 -= LEARNING_RATE * db1
        
        if epoch % 1000 == 0:
            print(f"[TRAIN] Epoch {epoch:5d} | Loss: {loss:.6f}")
            
    print("\n[EVAL] Kết quả dự đoán sau khi huấn luyện:")
    correct = 0
    for i in range(4):
        # Ép kiểu float cẩn thận cho từng phần tử
        in1 = float(X[i][0])
        in2 = float(X[i][1])
        expected = float(Y[i][0])
        
        z1 = np.dot(X[i], W1) + b1
        a1 = tanh(z1)
        z2 = np.dot(a1, W2) + b2
        out = float(tanh(z2)[0][0])
        
        pred_class = 1.0 if out >= 0.0 else -1.0
        is_correct = (pred_class == expected)
        if is_correct: correct += 1
        
        print(f"  Input: [{in1:.0f}, {in2:.0f}] | Expected: {expected:2.0f} | Output: {out:6.3f} | Pred: {pred_class:2.0f} | {'PASS' if is_correct else 'FAIL'}")
        
    print(f"\n[TRAIN][DONE] Độ chính xác: {correct}/4")
    
    print("\n" + "="*50)
    print("COPY CÁC MẢNG SAU VÀO FILE src/main.rs CỦA RUST:")
    print("="*50 + "\n")
    
    def format_array(name, arr):
        flat = arr.flatten()
        s = f"const {name}: [f32; {len(flat)}] = [\n    "
        s += ", ".join([f"{x:.6f}" for x in flat])
        s += "\n];"
        return s

    print(format_array("W1", W1))
    print(format_array("B1", b1))
    print(format_array("W2", W2))
    print(format_array("B2", b2))
    print()

if __name__ == "__main__":
    train()
