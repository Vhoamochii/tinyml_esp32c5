# Đối chiếu bộ trích xuất Rust (tinyml-core/src/features.rs) với bản NumPy float64.
#   python3 pc/pc-sim/tools/features_ref.py                      # tín hiệu giả: sine, tiếng gõ
#   python3 pc/pc-sim/tools/features_ref.py <file .mat>          # cửa sổ đầu của file thật (cần scipy)
# So với: cargo run --release --bin inspect_mat -- <file .mat>
import sys
import numpy as np

FS = 64000.0
BLOCK = 1024
BLOCKS = 16
WIN = BLOCK * BLOCKS
DECIM = 16
REL_EDGES = [1, 32, 128, 256, 513]
ENV_EDGES = [3, 6, 9, 16, 26, 44, 77]
HP_FC = 2000.0
EPS = 1e-12


def biquad_hp():
    w0 = 2 * np.pi * HP_FC / FS
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * np.sqrt(0.5))
    a0 = 1 + alpha
    b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]) / a0
    a = np.array([1.0, -2 * cw / a0, (1 - alpha) / a0])
    return b, a


def hp_filter(x):
    b, a = biquad_hp()
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n, v in enumerate(x):
        out = b[0] * v + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1 = x1, v
        y2, y1 = y1, out
        y[n] = out
    return y


def extract(x):
    x = np.asarray(x, dtype=np.float64)
    assert len(x) == WIN
    mean = x.mean()
    d = x - mean
    m2, m3, m4 = (d**2).mean(), (d**3).mean(), (d**4).mean()
    std = np.sqrt(m2)
    rms = np.sqrt((x**2).mean())
    peak = np.abs(x).max()
    mean_abs = np.abs(x).mean()
    clr = np.sqrt(np.abs(x)).mean()
    shape = [peak / (rms + EPS), m4 / (m2 * m2 + EPS), m3 / (m2 * std + EPS),
             peak / (mean_abs + EPS), rms / (mean_abs + EPS), peak / (clr * clr + EPS)]

    rel = np.zeros(4)
    for blk in x.reshape(BLOCKS, BLOCK):
        p = np.abs(np.fft.fft(blk - blk.mean()))**2
        for i in range(4):
            rel[i] += p[REL_EDGES[i]:REL_EDGES[i + 1]].sum()
    rel_f = np.log10(rel / (rel.sum() + EPS) + EPS)

    env = np.abs(hp_filter(x)).reshape(-1, DECIM).mean(axis=1)
    env = env - env.mean()
    n = len(env)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    pe = np.abs(np.fft.fft(env * hann))**2
    total = pe[ENV_EDGES[0]:ENV_EDGES[-1]].mean()
    env_f = [np.log10(pe[ENV_EDGES[i]:ENV_EDGES[i + 1]].mean() / (total + EPS) + EPS) for i in range(6)]
    return np.array(shape + list(rel_f) + env_f)


def impacts(rate_hz, resonance_hz=5000.0, tau=0.0005, amp=1.0):
    n = np.arange(WIN)
    x = np.zeros(WIN)
    k = 0
    while True:
        start = int(round(k * FS / rate_hz))
        if start >= WIN:
            break
        t = (n[start:] - start) / FS
        x[start:] += amp * np.exp(-t / tau) * np.sin(2 * np.pi * resonance_hz * t)
        k += 1
    return x


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    if len(sys.argv) > 1:
        import scipy.io
        m = scipy.io.loadmat(sys.argv[1])
        name = [k for k in m if not k.startswith("__")][0]
        y = m[name]["Y"][0, 0][0]
        sig = [c for c in y if c["Name"][0] == "vibration_1"][0]["Data"].ravel().astype(np.float32)
        print("16 dac trung cua cua so dau tien (Python):")
        print(extract(sig[:WIN].astype(np.float64)))
    else:
        t = np.arange(WIN) / FS
        print("sine 5k :", extract(np.sin(2 * np.pi * 5000 * t)))
        print("sine x3 :", extract(3 * np.sin(2 * np.pi * 5000 * t)))
        for r in [76.4, 123.6, 45.8, 74.2]:
            print(f"go {r:6.1f}:", extract(impacts(r))[10:])
