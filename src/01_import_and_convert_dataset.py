#!/usr/bin/env python3
"""
腳本: src/01_import_and_convert_dataset.py
=============================================================================
功能:
1. 從 Downloads 中的 "merged dataset.v3i.yolov8" 複製並建立 "dataset_tarp"。
2. 簡化類別由 4 類合併為 2 類：
   - 0: covered   (包含 loaded_covered 與 overloaded_covered)
   - 1: uncovered (包含 loaded_uncovered 與 overloaded_uncovered)
3. 產生專屬 data.yaml 配置檔。
=============================================================================
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

SOURCE_DIR = Path(r"C:\Users\USER\Downloads\merged dataset.v3i.yolov8")
TARGET_DIR = Path(r"c:\Users\USER\Desktop\dump-truck-tarp\dataset_tarp")

# 原類別對應:
# 0 (loaded_covered) -> 0 (covered)
# 1 (loaded_uncovered) -> 1 (uncovered)
# 2 (overloaded_covered) -> 0 (covered)
# 3 (overloaded_uncovered) -> 1 (uncovered)
CLASS_MAPPING = {
    0: 0,
    1: 1,
    2: 0,
    3: 1,
}

def convert_labels(src_label_file: Path, dst_label_file: Path):
    if not src_label_file.exists():
        return
    with open(src_label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        orig_cls = int(parts[0])
        new_cls = CLASS_MAPPING.get(orig_cls, 1)
        new_line = f"{new_cls} " + " ".join(parts[1:]) + "\n"
        new_lines.append(new_line)
    
    with open(dst_label_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def process_split(split_name: str):
    src_split = SOURCE_DIR / split_name
    dst_split = TARGET_DIR / split_name
    
    src_img_dir = src_split / "images"
    src_lbl_dir = src_split / "labels"
    
    dst_img_dir = dst_split / "images"
    dst_lbl_dir = dst_split / "labels"
    
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
    print(f"處理 {split_name} 分割集: 共 {len(img_files)} 張圖片...")
    
    for img_path in tqdm(img_files, desc=f"Importing {split_name}"):
        dst_img_path = dst_img_dir / img_path.name
        if not dst_img_path.exists() or dst_img_path.stat().st_size != img_path.stat().st_size:
            shutil.copy2(img_path, dst_img_path)
            
        lbl_name = img_path.stem + ".txt"
        src_lbl_path = src_lbl_dir / lbl_name
        dst_lbl_path = dst_lbl_dir / lbl_name
        if src_lbl_path.exists():
            convert_labels(src_lbl_path, dst_lbl_path)

def create_yaml():
    yaml_content = f"""path: {TARGET_DIR.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['covered', 'uncovered']
"""
    yaml_path = TARGET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"已建立 data.yaml: {yaml_path}")

def main():
    if not SOURCE_DIR.exists():
        print(f"Source dir not found: {SOURCE_DIR}")
        return
    
    print("=" * 68)
    print(" [IMPORT] Importing Roboflow dataset and converting to 2 classes (covered / uncovered)")
    print(f" Source: {SOURCE_DIR}")
    print(f" Target: {TARGET_DIR}")
    print("=" * 68)
    
    for split in ["train", "valid", "test"]:
        process_split(split)
        
    create_yaml()
    print("=" * 68)
    print(" [DONE] Dataset import and class mapping completed successfully!")
    print("=" * 68)

if __name__ == "__main__":
    main()
