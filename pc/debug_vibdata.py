import os
from vibdata.raw import PU_raw

root_dir = os.path.expanduser("~/Work/Project_ESP32C5/pc/data/paderborn_vibdata")
print("[DEBUG] Đang tải dataset...")
dataset = PU_raw(root_dir, download=True)

print("\n=== [1] KIỂM TRA METADATA ===")
meta_df = dataset.getMetaInfo()
print(f"Các cột hiện có: {meta_df.columns.tolist()}")
print("5 dòng đầu tiên:")
print(meta_df.head())

print("\n=== [2] KIỂM TRA DỮ LIỆU MẪU (dataset[0]) ===")
sample = dataset[0]
print(f"Kiểu dữ liệu: {type(sample)}")
if isinstance(sample, dict):
    print(f"Các key có sẵn: {list(sample.keys())}")
    for k, v in sample.items():
        if hasattr(v, 'shape'):
            print(f"  - Key '{k}': shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  - Key '{k}': type={type(v)}")
else:
    print(f"Shape: {sample.shape}")

print("\n=== [3] KIỂM TRA 3 MẪU TIẾP THEO ===")
for i in range(1, 4):
    s = dataset[i]
    if isinstance(s, dict):
        print(f"Mẫu {i}: keys={list(s.keys())}")
    else:
        print(f"Mẫu {i}: shape={s.shape}")
