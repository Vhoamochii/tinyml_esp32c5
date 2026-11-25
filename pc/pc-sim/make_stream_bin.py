#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2: nen CSV thanh blob nhi phan de firmware nhung bang include_bytes.
Dinh dang: u32 LE so dong; moi dong 20 byte:
  z[16] i8 | cls u8 | cond u8 (0=N15,1=N09) | record u8 | bearing_idx u8
Bang ten o cung thu tu xuat hien (firmware giu cung bang de parity log).
Khong chua reserve loai bo: reserve van nam trong blob, firmware tu tach.
"""
import csv, hashlib, os, struct, sys

IN = "fixtures/features_paderborn.csv"
OUT = os.path.expanduser("~/Work/Project_ESP32C5/fw/fw-esp32c5/data/stream.bin")

def main():
    rows = list(csv.DictReader(open(IN)))
    names = []
    for r in rows:
        if r["bearing"] not in names:
            names.append(r["bearing"])
    if len(names) > 255:
        sys.exit("[BINGEN][ERROR] qua 255 ten o")
    buf = bytearray(struct.pack("<I", len(rows)))
    buf += struct.pack("<I", len(names))
    for n in names:
        b = n.encode()
        buf += struct.pack("<B", len(b)) + b
    for r in rows:
        z = [int(r[f"z{i}"]) for i in range(16)]
        cond = 0 if r["condition"] == "N15_M07_F10" else 1
        buf += struct.pack("<16bBBBB", *z, int(r["class"]), cond,
                           int(r["record"]), names.index(r["bearing"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "wb").write(bytes(buf))
    sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    print(f"[BINGEN][DONE] rows={len(rows)} bearings={names} bytes={len(buf)} sha256={sha[:16]}")
    print(f"[BINGEN][OUT] {OUT}")

if __name__ == "__main__":
    main()
