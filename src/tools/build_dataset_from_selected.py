#!/usr/bin/env python3
"""
腳本: src/05_build_dataset_from_selected.py
=============================================================================
功能:
當使用者在 curation_pool_767/selected/ 中挑選好圖片後：
1. 自動辨識使用者挑選的照片檔名（支援直接複製 images 或從 previews 複製）
2. 自動配對對應的標準 2 類別 YOLO labels (.txt)
3. 自動以 8:2 劃分 Train / Val
4. 建立精煉後的標準資料集 dataset_curated/ 並產生 data.yaml
=============================================================================
"""

import os
import shutil
import random
from pathlib import Path

POOL_DIR = Path("archive/02_curation_pool_767").resolve()
SELECTED_DIR = POOL_DIR / "selected"
TARGET_DATASET = Path("dataset_curated").resolve()

def main():
    print("=" * 68)
    print(" [BUILD] Building curated dataset from selected images...")
    print("=" * 68)

    selected_files = list(SELECTED_DIR.glob("*.jpg")) + list(SELECTED_DIR.glob("*.png"))
    if not selected_files:
        print(f"[Notice] No images found in {SELECTED_DIR}")
        print("Please copy your chosen images into curation_pool_767/selected/ first!")
        return

    print(f"Detected {len(selected_files)} selected images.")

    # 建立目標資料夾
    for split in ["train", "val"]:
        (TARGET_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    items = []
    for f in selected_files:
        # 處理若使用者從 previews 複製帶有 view_ 前綴的檔名
        base_name = f.stem
        if base_name.startswith("view_"):
            base_name = base_name[len("view_"):]

        clean_img = POOL_DIR / "images" / f"{base_name}.jpg"
        clean_lbl = POOL_DIR / "labels" / f"{base_name}.txt"

        if not clean_img.exists():
            clean_img = f  # 若本身就是乾淨圖

        if clean_img.exists() and clean_lbl.exists():
            items.append((clean_img, clean_lbl, base_name))
        else:
            print(f"[Warning] Missing label or image for: {base_name}")

    if not items:
        print("[Error] No valid image-label pairs found.")
        return

    # 隨機打亂並按 8:2 切分
    random.seed(42)
    random.shuffle(items)
    n_train = int(len(items) * 0.8)
    train_items = items[:n_train]
    val_items = items[n_train:]

    for img_p, lbl_p, bname in train_items:
        shutil.copy2(img_p, TARGET_DATASET / "images" / "train" / f"{bname}.jpg")
        shutil.copy2(lbl_p, TARGET_DATASET / "labels" / "train" / f"{bname}.txt")

    for img_p, lbl_p, bname in val_items:
        shutil.copy2(img_p, TARGET_DATASET / "images" / "val" / f"{bname}.jpg")
        shutil.copy2(lbl_p, TARGET_DATASET / "labels" / "val" / f"{bname}.txt")

    yaml_content = f"""path: {TARGET_DATASET.as_posix()}
train: images/train
val: images/val

nc: 2
names: ['covered', 'uncovered']
"""
    with open(TARGET_DATASET / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("=" * 68)
    print(" [DONE] Curated dataset successfully built!")
    print(f" Total Selected: {len(items)} images")
    print(f" Train: {len(train_items)} images")
    print(f" Val: {len(val_items)} images")
    print(f" Dataset Path: {TARGET_DATASET}")
    print(f" YAML Config: {TARGET_DATASET / 'data.yaml'}")
    print("=" * 68)

if __name__ == "__main__":
    main()
