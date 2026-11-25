#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1+D5+D7: npz that 6 khoa -> CSV INT8 + scale + bang chuan hoa.
D7 (da duyet): chuan hoa mean/std truoc luong tu, winsorize +/-4 sigma,
scale deu = 4/127. Muc tieu: |z| giai luong tu <= 4, SGD lr=0.1 on dinh,
het co che "may nho nhan cuoi" do bao hoa softmax (H2).
Thong ke chuan hoa mac dinh tinh tu train N15 (mien nguon) de tranh
ro ri tri thuc mien tuong lai; --norm-source all neu muon toan train.
D5 giu nguyen: xen ke o, khoa sap (condkey, record, win, bearing).
Giu reserve N15 17-20 trong CSV; loc y > 2.
--synthetic: chi thong pipeline, KHONG dua vao bao cao.
"""
import argparse
import csv
import os
import numpy as np

N_FEAT = 16
WINSOR_SIGMA = 4.0
QMAX = 127.0
UNIFORM_SCALE = WINSOR_SIGMA / QMAX  # 4/127 ~ 0.0314961
HEADER = ["bearing", "class", "condition", "record", "win",
          *[f"z{i}" for i in range(N_FEAT)]]
DEFAULT_NPZ = os.path.expanduser("~/Work/Project_ESP32C5/pc/data/test.npz")
DEFAULT_TRAIN = os.path.expanduser("~/Work/Project_ESP32C5/pc/data/train.npz")


def log(tag, msg):
    print(f"[CSVGEN][{tag}] {msg}")


def compute_norm(train_path, source):
    """D7a: mean/std tung chieu tu train. source='n15' chi lay mien nguon."""
    d = np.load(train_path)
    if "X" not in d.keys():
        raise SystemExit("[CSVGEN][ERROR] train.npz thieu khoa X")
    X = np.asarray(d["X"], dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != N_FEAT:
        raise SystemExit(f"[CSVGEN][ERROR] X train sai shape: {X.shape}")
    if source == "n15":
        if "condition" not in d.keys():
            raise SystemExit("[CSVGEN][ERROR] train.npz thieu condition cho loc n15")
        mask = np.asarray(d["condition"]).astype(str) == "N15_M07_F10"
        X = X[mask]
    if X.shape[0] == 0:
        raise SystemExit(f"[CSVGEN][ERROR] khong con dong train nao voi source={source}")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Guard: chieu hang so (std ~ 0) khong duoc chia 0
    std = np.maximum(std, 1e-12)
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
        raise SystemExit("[CSVGEN][ERROR] mean/std khong huu han")
    log("NORM", f"source={source} rows={X.shape[0]} "
                f"mean_range=[{mean.min():.6g},{mean.max():.6g}] "
                f"std_range=[{std.min():.6g},{std.max():.6g}]")
    return mean, std


def write_scales(path, scale):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "scale"])
        for j, s in enumerate(scale):
            w.writerow([j, repr(float(s))])
    log("SCALE", f"ghi {path}: min={scale.min():.6g} max={scale.max():.6g}")


def write_norm(path, mean, std):
    """D7c: luu mean/std de tai lap va xuat firmware sau nay."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "mean", "std"])
        for j in range(N_FEAT):
            w.writerow([j, repr(float(mean[j])), repr(float(std[j]))])
    log("NORMFILE", f"ghi {path} (dim,mean,std)")


def from_npz(npz_path, train_path, out, scales_out, norm_out, norm_source):
    mean, std = compute_norm(train_path, norm_source)
    scale = np.full(N_FEAT, UNIFORM_SCALE, dtype=np.float64)
    write_scales(scales_out, scale)
    write_norm(norm_out, mean, std)

    d = np.load(npz_path)
    need = ["X", "y", "bearing", "condition", "record", "win"]
    missing = [k for k in need if k not in d.keys()]
    if missing:
        raise SystemExit(f"[CSVGEN][ERROR] thieu khoa {missing} trong {npz_path}")
    X = np.asarray(d["X"], dtype=np.float64)
    y = np.asarray(d["y"]).astype(int)
    b = np.asarray(d["bearing"]).astype(str)
    c = np.asarray(d["condition"]).astype(str)
    rec = np.asarray(d["record"]).astype(int)
    win = np.asarray(d["win"]).astype(int)
    n_in = len(y)
    if not (len(X) == n_in == len(b) == len(c) == len(rec) == len(win)):
        raise SystemExit("[CSVGEN][ERROR] cac mang khong cung so dong")

    keep = (y >= 0) & (y <= 2)
    n_kb = int(np.sum(~keep))
    X, y, b, c, rec, win = X[keep], y[keep], b[keep], c[keep], rec[keep], win[keep]

    # D7b: chuan hoa -> winsorize +/-4 sigma -> luong tu scale deu 4/127
    Xstd = (X - mean) / std
    over = np.abs(Xstd) > WINSOR_SIGMA
    n_winsor = int(np.sum(over))
    for cname in ("N15_M07_F10", "N09_M07_F10"):
        m = c == cname
        tot = int(np.sum(m)) * N_FEAT
        hit = int(np.sum(over[m]))
        pct = 100.0 * hit / tot if tot > 0 else 0.0
        log("WINSOR", f"{cname}: {hit}/{tot} gia tri bi cat ({pct:.3f}%)")
    Xw = np.clip(Xstd, -WINSOR_SIGMA, WINSOR_SIGMA)
    Zf = np.rint(Xw / UNIFORM_SCALE)
    n_round_clip = int(np.sum(np.abs(Zf) > QMAX))  # ky vong 0 sau winsor
    Z = np.clip(Zf, -QMAX, QMAX).astype(np.int8)

    # D5: xen ke o, khoa sap (condkey, record, win, bearing)
    condkey = np.where(c == "N15_M07_F10", 0, 1)
    order = np.lexsort((b, win, rec, condkey))
    n_reserve = int(np.sum((c == "N15_M07_F10") & (rec >= 17)))
    n_n09 = int(np.sum(condkey == 1))
    n_n15_stream = int(np.sum((condkey == 0) & (rec < 17)))

    ob, oc, orec, oy = b[order], c[order], rec[order], y[order]
    stream_mask = ~((oc == "N15_M07_F10") & (orec >= 17))
    blocks, prev, first12 = 0, None, []
    for i in range(len(order)):
        if not stream_mask[i]:
            continue
        key = (oc[i], ob[i])
        if key != prev:
            blocks += 1
            prev = key
        if len(first12) < 12 and oc[i] == "N15_M07_F10":
            first12.append((str(ob[i]), int(oy[i])))
    log("ORDER", f"blocks={blocks} (ky vong >> 12 neu xen ke o) first12={first12}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i in order:
            w.writerow([b[i], int(y[i]), c[i], int(rec[i]), int(win[i]),
                        *Z[i].tolist()])

    ok = (len(order) > 0) and (n_reserve > 0) and (n_n09 > 0) and (blocks > 12)
    log("CLIP", f"round_clip expected=0 actual={n_round_clip} "
                f"(winsor={n_winsor} gia tri, hanh vi dung cua D7)")
    log("COUNT", f"in={n_in} kb_loai={n_kb} out={len(order)} "
                 f"reserve_N15_17_20={n_reserve} N15_stream={n_n15_stream} N09={n_n09}")
    log("DONE", f"{out}: {len(order)} dong {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("[CSVGEN][ERROR] thieu reserve/N09 hoac van con khoi o")


def synthetic(out, scales_out, norm_out):
    import random
    random.seed(42)
    write_scales(scales_out, np.ones(N_FEAT, dtype=np.float64))
    write_norm(norm_out, np.zeros(N_FEAT), np.ones(N_FEAT))
    bearings = ["K004", "KI14", "KA16"]
    cls = {"K004": 0, "KI14": 1, "KA16": 2}
    n = 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for b in bearings:
            for cond in ["N15_M07_F10", "N09_M07_F10"]:
                for rec in range(1, 21):
                    for win in range(8):
                        base = 40 + 30 * cls[b]
                        if cond == "N09_M07_F10":
                            base -= 25
                        z = [max(-127, min(127, int(random.gauss(base, 8))))
                             for _ in range(N_FEAT)]
                        w.writerow([b, cls[b], cond, rec, win, *z])
                        n += 1
    log("SYNTHETIC", f"{n} cua so. CANH BAO: du lieu phong, khong dung cho bao cao")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz", nargs="?", const=DEFAULT_NPZ, default=None)
    g.add_argument("--synthetic", action="store_true")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--norm-source", choices=["n15", "all"], default="n15")
    p.add_argument("--out", default="fixtures/features_paderborn.csv")
    p.add_argument("--scales-out", default="fixtures/feature_scales.csv")
    p.add_argument("--norm-out", default="fixtures/feature_norm.csv")
    a = p.parse_args()
    if a.synthetic:
        synthetic(a.out, a.scales_out, a.norm_out)
    else:
        from_npz(a.npz or DEFAULT_NPZ, a.train, a.out,
                 a.scales_out, a.norm_out, a.norm_source)
