#!/usr/bin/env python3
"""
腳本: src/04_extract_unique_767.py
=============================================================================
功能:
從 2,674 張 Roboflow 增強資料集中，去重抽取出 767 張「不重複原始基底影像」：
1. 輸出至 curation_pool_767/images/ (原始乾淨影像)
2. 輸出至 curation_pool_767/labels/ (已轉換為 2 類別的標準 YOLO txt)
3. 輸出至 curation_pool_767/previews/ (已畫好 🟢/🔴 框線的預覽圖，方便縮圖直接挑選)
=============================================================================
"""

import os
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm

SOURCE_DIR = Path(r"C:\Users\USER\Downloads\merged dataset.v3i.yolov8")
TARGET_DIR = Path("archive/02_curation_pool_767").resolve()

# 4 類別轉 2 類別
# 0 (loaded_covered) -> 0 (covered)
# 1 (loaded_uncovered) -> 1 (uncovered)
# 2 (overloaded_covered) -> 0 (covered)
# 3 (overloaded_uncovered) -> 1 (uncovered)
CLASS_MAPPING = {0: 0, 1: 1, 2: 0, 3: 1}

def convert_and_draw(img_path: Path, src_lbl_path: Path, dst_lbl_path: Path, preview_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]

    new_lines = []
    boxes_to_draw = []

    if src_lbl_path.exists():
        with open(src_lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                orig_cls = int(parts[0])
                new_cls = CLASS_MAPPING.get(orig_cls, 1)
                new_lines.append(f"{new_cls} " + " ".join(parts[1:]) + "\n")

                # 解析座標供畫預覽
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2.0) * w)
                y1 = int((cy - bh / 2.0) * h)
                x2 = int((cx + bw / 2.0) * w)
                y2 = int((cy + bh / 2.0) * h)
                boxes_to_draw.append((x1, y1, x2, y2, new_cls))

    with open(dst_lbl_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # 繪製預覽圖
    annotated = img.copy()
    for x1, y1, x2, y2, cls_id in boxes_to_draw:
        if cls_id == 0:
            color = (0, 200, 0)
            tag = "Covered"
        else:
            color = (0, 0, 255)
            tag = "UNCOVERED"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, tag, (x1 + 3, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(str(preview_path), annotated)

def main():
    print("=" * 68)
    print(" [EXTRACT] Starting extraction of 767 unique base images and labels...")
    print(f" Source: {SOURCE_DIR}")
    print(f" Target: {TARGET_DIR}")
    print("=" * 68)

    dst_img_dir = TARGET_DIR / "images"
    dst_lbl_dir = TARGET_DIR / "labels"
    dst_prv_dir = TARGET_DIR / "previews"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    dst_prv_dir.mkdir(parents=True, exist_ok=True)

    all_jpgs = list(SOURCE_DIR.rglob("*.jpg"))
    seen_bases = set()
    unique_items = []

    for img_p in all_jpgs:
        # 去掉 Roboflow 的 .rf.hash 尾綴
        base_key = img_p.name.split(".rf.")[0]
        if base_key not in seen_bases:
            seen_bases.add(base_key)
            # 找到對應標註 txt
            split_dir = img_p.parent.parent
            lbl_p = split_dir / "labels" / (img_p.stem + ".txt")
            unique_items.append((img_p, lbl_p, base_key))

    print(f"Unique base scenes found: {len(unique_items)}")

    for img_p, lbl_p, base_key in tqdm(unique_items, desc="Exporting unique 767"):
        # 統一正規化乾淨檔名
        dst_img_name = f"{base_key}.jpg"
        dst_lbl_name = f"{base_key}.txt"

        dst_img_p = dst_img_dir / dst_img_name
        dst_lbl_p = dst_lbl_dir / dst_lbl_name
        dst_prv_p = dst_prv_dir / f"view_{base_key}.jpg"

        # 複製乾淨圖
        shutil.copy2(img_p, dst_img_p)

        # 轉換標註並生成帶框預覽圖
        convert_and_draw(img_p, lbl_p, dst_lbl_p, dst_prv_p)

    guide_txt = TARGET_DIR / "README_PICKING_GUIDE.txt"
    with open(guide_txt, "w", encoding="utf-8") as f:
        guide_content = (
            "【767 張不重複卡車圖片挑選指南】\n"
            "=============================================================================\n"
            "1. 快速檢視方式：\n"
            "   打開「previews/」資料夾，在 Windows 檔案總管中切換為「大圖示」或「特大圖示」。\n"
            "   每張圖上面已經用 綠框 (Covered 帆布有蓋) 與 紅框 (UNCOVERED 未蓋) 標記好。\n\n"
            "2. 挑選原則：\n"
            "   - 看到真正是「大型砂石車車斗」、且框選位置合理的 -> 保留 / 複製出來。\n"
            "   - 看到「紙箱、廂型車、非砂石車、標記錯誤」的 -> 略過。\n\n"
            "3. 建議數量：\n"
            "   - 綠框 (Covered)：挑選約 40~50 張（務必多挑幾張灰色/水泥灰帆布）。\n"
            "   - 紅框 (UNCOVERED)：挑選約 40~50 張（露砂石、泥土或碎石）。\n\n"
            "4. 挑好後：\n"
            "   建立一個「selected/」資料夾，把挑選好的圖片丟進去，隨時跟我說，\n"
            "   我會自動將其轉為新訓練集並以 GPU 重新訓練！\n"
            "=============================================================================\n"
        )
        f.write(guide_content)

    print("=" * 68)
    print(" [DONE] 767 unique base images, labels, and previews exported successfully!")
    print(f" Target directory: {TARGET_DIR}")
    print("=" * 68)

if __name__ == "__main__":
    main()
