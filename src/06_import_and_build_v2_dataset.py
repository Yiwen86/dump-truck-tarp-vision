#!/usr/bin/env python3
"""
腳本: src/06_import_and_build_v2_dataset.py
=============================================================================
功能:
1. 安全匯入 Downloads 中的全新 57 張標註資料集 (Roboflow 格式)。
2. 完整保留至 dataset_v2/raw_downloaded_57/ 作為持久備份。
3. 採用分層抽樣 (Stratified Split)：
   - 總計: 57 張 (covered: 7, uncovered: 50)
   - Train: 46 張 (covered: 6, uncovered: 40)
   - Val:   11 張 (covered: 1, uncovered: 10)
4. 與 dataset_tarp (1,868 Train / 534 Val / 268 Test) 進行特徵融合，
   建構出兼具通用泛化與路口俯視特徵的完整 dataset_v2 資料集與 data.yaml。
=============================================================================
"""

import os
import sys
import shutil
import random
from pathlib import Path

# 確保 Windows 命令列 UTF-8 輸出安全
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 路徑定義
SOURCE_DIR = Path(r"C:\Users\USER\Downloads\truck-tarp-detection.v1i.yolov8 (1)").resolve()
BASE_DATASET = Path("dataset_tarp").resolve()
TARGET_DATASET = Path("dataset_v2").resolve()
BACKUP_RAW_DIR = TARGET_DATASET / "raw_downloaded_57"

def get_class_for_label_file(label_path: Path):
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return None
    # 取得第一筆目標的類別 ID
    return int(lines[0].split()[0])

def main():
    print("=" * 70)
    print(" [IMPORT & BUILD] Creating dataset_v2 with Data Fusion & Stratified Split")
    print("=" * 70)

    src_images_dir = SOURCE_DIR / "train" / "images"
    src_labels_dir = SOURCE_DIR / "train" / "labels"

    if not src_images_dir.exists() or not src_labels_dir.exists():
        print(f"[Error] Source directory not found: {SOURCE_DIR}")
        return

    # 1. 建立目標結構
    for split in ["train", "valid", "test"]:
        (TARGET_DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (TARGET_DATASET / split / "labels").mkdir(parents=True, exist_ok=True)
    BACKUP_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 備份 57 張原始資料
    print(" [1/4] 備份 57 張原圖與標註至 dataset_v2/raw_downloaded_57/...")
    src_images = sorted(list(src_images_dir.glob("*.jpg")) + list(src_images_dir.glob("*.png")))
    for img_p in src_images:
        shutil.copy2(img_p, BACKUP_RAW_DIR / img_p.name)
        lbl_p = src_labels_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, BACKUP_RAW_DIR / lbl_p.name)
    print(f"       已完成 {len(src_images)} 張原始圖檔與標註備份。")

    # 3. 複製 base dataset (dataset_tarp)
    print(" [2/4] 複製基礎資料集 (dataset_tarp) 至 dataset_v2/...")
    for split in ["train", "valid", "test"]:
        old_img_dir = BASE_DATASET / split / "images"
        old_lbl_dir = BASE_DATASET / split / "labels"
        
        target_img_dir = TARGET_DATASET / split / "images"
        target_lbl_dir = TARGET_DATASET / split / "labels"

        count = 0
        if old_img_dir.exists():
            for f in old_img_dir.glob("*.*"):
                shutil.copy2(f, target_img_dir / f.name)
                lbl_f = old_lbl_dir / f"{f.stem}.txt"
                if lbl_f.exists():
                    shutil.copy2(lbl_f, target_lbl_dir / lbl_f.name)
                count += 1
        print(f"       {split.capitalize()}: 已複製 {count} 張基礎圖檔與標註。")

    # 4. 分層抽樣 (Stratified Split) 57 張新圖片
    print(" [3/4] 對 57 張新圖片執行分層抽樣 (Stratified Split)...")
    covered_items = []
    uncovered_items = []

    for img_p in src_images:
        lbl_p = src_labels_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            print(f"[Warning] No label for {img_p.name}")
            continue
        cls_id = get_class_for_label_file(lbl_p)
        if cls_id == 0:
            covered_items.append((img_p, lbl_p))
        elif cls_id == 1:
            uncovered_items.append((img_p, lbl_p))

    print(f"       檢測到 Covered (類別0): {len(covered_items)} 張")
    print(f"       檢測到 Uncovered (類別1): {len(uncovered_items)} 張")

    # 固定隨機種子確保可重現性
    random.seed(42)
    random.shuffle(covered_items)
    random.shuffle(uncovered_items)

    # 切分：covered 7 張 -> 6 train / 1 val
    #       uncovered 50 張 -> 40 train / 10 val
    new_train = covered_items[:6] + uncovered_items[:40]
    new_val = covered_items[6:] + uncovered_items[40:]

    print(f"       抽樣結果 -> Train: {len(new_train)} 張 (6 covered, 40 uncovered)")
    print(f"       抽樣結果 -> Val:   {len(new_val)} 張 (1 covered, 10 uncovered)")

    # 寫入 Train
    for img_p, lbl_p in new_train:
        shutil.copy2(img_p, TARGET_DATASET / "train" / "images" / img_p.name)
        shutil.copy2(lbl_p, TARGET_DATASET / "train" / "labels" / lbl_p.name)

    # 寫入 Val
    for img_p, lbl_p in new_val:
        shutil.copy2(img_p, TARGET_DATASET / "valid" / "images" / img_p.name)
        shutil.copy2(lbl_p, TARGET_DATASET / "valid" / "labels" / lbl_p.name)

    # 5. 產出 data.yaml
    print(" [4/4] 產出 dataset_v2/data.yaml 設定檔...")
    yaml_content = f"""path: {TARGET_DATASET.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['covered', 'uncovered']
"""
    with open(TARGET_DATASET / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # 統計最終數量
    total_train = len(list((TARGET_DATASET / "train" / "images").glob("*.*")))
    total_val = len(list((TARGET_DATASET / "valid" / "images").glob("*.*")))
    total_test = len(list((TARGET_DATASET / "test" / "images").glob("*.*")))

    print("=" * 70)
    print(" [SUCCESS] dataset_v2 構建完成！")
    print(f" 目標路徑: {TARGET_DATASET}")
    print(f" 最終資料量統計:")
    print(f"    - Train (訓練集): {total_train} 張 (原 1,868 + 新 46)")
    print(f"    - Valid (驗證集): {total_val} 張 (原 534 + 新 11)")
    print(f"    - Test  (測試集): {total_test} 張 (保留原 268)")
    print(f"    - 總計: {total_train + total_val + total_test} 張")
    print("=" * 70)

if __name__ == "__main__":
    main()
