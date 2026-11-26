#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test doc lap cho loader sua: happy / edge / error.
Khong can raw that - dung dataset gia. Chi chay sau khi file loader da ghi.
Log [TEST][LoaderFix][Case] expected=... actual=... PASS/FAIL. Exit 1 neu co FAIL.
Luu y: sorted() xep chu hoa 'X' truoc chu thuong, nen 'y' dung cuoi.
"""
import importlib.util
import os
import sys

import numpy as np

# Nap module loader theo duong dan de test dung file vua sua
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "loader_under_test", os.path.join(HERE, "data_loader_vibdata.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

FAILS = []
# Thu tu dung cua sorted(keys): 'X' hoa dung truoc, 'y' thuong dung cuoi
KEYS_SORTED = ["X", "bearing", "condition", "record", "win", "y"]


def check(name, expected, actual):
    ok = (expected == actual)
    print(f"[TEST][LoaderFix][{name}] expected={expected} actual={actual} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(name)


class StubDS:
    """Dataset gia: len + [i] -> {'signal': arr}. Khong dung vibdata/raw."""
    def __init__(self, signals):
        self.signals = signals

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        return {"signal": self.signals[i]}


def meta(bearing, file_name):
    return {"bearing_code": bearing, "file_name": file_name}


rng = np.random.default_rng(0)
sig_ok = rng.normal(0.0, 1.0, 256000)          # var > 1e-12, features huu han
sig_dead = np.zeros(256000)                     # var = 0 -> cua so bi loai

# --- HAPPY ---
check("label_K001", 0, L.label_for("K001"))
check("label_KI14", 1, L.label_for("KI14"))
check("label_KA16", 2, L.label_for("KA16"))
check("label_KB23", 3, L.label_for("KB23"))
check("parse_normal", ("N15_M07_F10", 13),
      L.parse_condition_record("N15_M07_F10_K004_13.mat"))
check("wincount_256000", 250, L.window_count(256000))

ds = StubDS([sig_ok, sig_ok])
mm = [meta("K001", "N15_M07_F10_K001_1.mat"),
      meta("KI14", "N15_M07_F10_KI14_1.mat")]
d = L.build_split(ds, mm, ["K001", "KI14"], L.CONDITIONS)
check("happy_keys", KEYS_SORTED, sorted(d.keys()))
check("happy_rows", 500, len(d["y"]))
check("happy_labels", {0: 250, 1: 250},
      dict(zip(*[a.tolist() for a in np.unique(d["y"], return_counts=True)])))
check("happy_Xshape", (500, 16), d["X"].shape)

# --- EDGE ---
check("wincount_1024", 1, L.window_count(1024))
check("wincount_1023", 0, L.window_count(1023))
check("wincount_0", 0, L.window_count(0))
check("parse_rec1", ("N09_M07_F10", 1),
      L.parse_condition_record("N09_M07_F10_K004_1.mat"))
check("parse_rec20", ("N15_M07_F10", 20),
      L.parse_condition_record("N15_M07_F10_K004_20.mat"))

# Dieu kien thu 3 (N15_M01_F10) phai bi loc bo
ds3 = StubDS([sig_ok, sig_ok])
mm3 = [meta("K001", "N15_M07_F10_K001_1.mat"),
       meta("K001", "N15_M01_F10_K001_1.mat")]
d3 = L.build_split(ds3, mm3, ["K001"], L.CONDITIONS)
check("edge_cond_filter_rows", 250, len(d3["y"]))
check("edge_cond_only", ["N15_M07_F10"], sorted(set(d3["condition"].tolist())))

# Tin hieu chet: var=0 -> khong cua so nao
dsd = StubDS([sig_dead])
mmd = [meta("K001", "N15_M07_F10_K001_1.mat")]
dd = L.build_split(dsd, mmd, ["K001"], L.CONDITIONS)
check("edge_dead_signal_rows", 0, len(dd["y"]))

# --- ERROR ---
check("err_label_unknown", None, L.label_for("X999"))
check("err_parse_bad", (None, None), L.parse_condition_record("bad.mat"))
dse = StubDS([])
de = L.build_split(dse, [], ["K001"], L.CONDITIONS)
check("err_empty_keys", KEYS_SORTED, sorted(de.keys()))
check("err_empty_Xshape", (0, 16), de["X"].shape)
check("err_empty_rows", 0, len(de["y"]))

print(f"[TEST][LoaderFix][SUMMARY] fails={len(FAILS)} {FAILS}")
sys.exit(1 if FAILS else 0)
