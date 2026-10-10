#!/usr/bin/env python3
"""
腳本: src/05_prepare_dataset.py
=============================================================================
功能:
為方案一（YOLOv8 模型微調）準備標準 YOLO 格式標註資料集。
1. 彙整 421 張道路影像，涵蓋四大真實場景：
   - 類別 A: 遠/中景正常行駛卡車 (YOLO 高信心檢出)
   - 類別 B: 手前近景俯視大車斗/帆布 (COCO 常誤判為 skis/refrigerator/suitcase 者)
   - 類別 C: 左側邊界嚴重裁切車輛 (如使用者提供之夜間車頭截斷、大燈直射者)
   - 類別 D: 空車道 / 純泥土路面背景 (負樣本 Negative Background，無框)
2. 自動劃分訓練集 (train: 80%) 與驗證集 (val: 20%)。
3. 輸出 YOLO 標準資料夾結構 (dataset/images, dataset/labels, dataset/data.yaml)。
4. 輸出視覺化標註預覽至 runs/dataset_preview/ 供確認品質。
=============================================================================
"""

import os
import shutil
import glob
import re
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def to_yolo_format(box_xyxy, img_w=512, img_h=512):
    """將 [x1, y1, x2, y2] 轉為 YOLO 歸一化 [cls, cx, cy, w, h]"""
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0.0, min(float(x1), float(img_w)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    y2 = max(0.0, min(float(y2), float(img_h)))
    
    w = x2 - x1
    h = y2 - y1
    if w < 10 or h < 10:
        return None
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return f"0 {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n"

def main():
    print("=" * 68)
    print(" 🛠️ 方案一: 準備專屬卡車偵測資料集 (Dataset Preparation)")
    print("=" * 68)

    raw_dir = Path("data/raw_images")
    dataset_dir = Path("dataset")
    preview_dir = Path("runs/dataset_preview")

    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # 載入基礎 YOLO 輔助特徵提取
    base_model = YOLO("yolov8n.pt")
    all_images = sorted(list(raw_dir.glob("*.jpg")))
    print(f"原始影像庫總數: {len(all_images)} 張")

    # 收集有卡車與無卡車標註
    labels_map = {} # filename -> list of [x1, y1, x2, y2]

    for img_path in all_images:
        fn = img_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        m = re.match(r"frame_(\d+)_jpg", fn)
        fid = int(m.group(1)) if m else -1

        boxes = []

        # 1. 執行基礎 YOLO 預測 (低門檻捕獲各種車輛與大型物體)
        preds = base_model.predict(img, conf=0.08, verbose=False)[0].boxes
        
        # 檢查左側裁切車輛 (例如 frame 0 ~ 13 常見)
        # 邊界: x1 <= 10, x2 <= 100, y1 >= 10, y2 <= 260
        left_edge_detected = False
        for b in preds:
            cls_id = int(b.cls[0])
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf[0])
            # 左側車門/車頭 (COCO 常見誤分類為 bus, truck, car, knife, tv)
            if xyxy[0] <= 15 and xyxy[2] <= 110 and 20 <= xyxy[1] <= 100:
                boxes.append([0, int(xyxy[1]), int(max(xyxy[2], 50)), int(min(h, xyxy[3]+30))])
                left_edge_detected = True
                break

        # 如果左側車頭有強光但沒被框出 (Frame 0~12)
        if not left_edge_detected and 0 <= fid <= 12:
            # 檢查左上方是否有車身白車頂/大燈光暈
            roi_left = cv2.cvtColor(img[30:220, 0:65], cv2.COLOR_BGR2GRAY)
            if np.mean(roi_left) > 100:
                boxes.append([0, 30, 65, 220])

        # 2. 捕獲手前俯視大車斗 (Frame 20 ~ 85)
        # COCO 誤判為 skis, refrigerator, suitcase 或直接被截斷之大車
        for b in preds:
            cls_id = int(b.cls[0])
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            # COCO: 7 truck, 5 bus, 28 skis, 72 refrigerator, 28 suitcase
            if cls_id in [7, 5, 28, 72, 28]:
                # 排除右側固定藍色廠房建築 (x1 > 280, x2 > 450, y1 > 100)
                if xyxy[0] > 280 and xyxy[2] > 480 and xyxy[1] > 100:
                    continue
                boxes.append(xyxy)

        # 3. 捕獲中遠景正常車輛 (Class 7 truck, Class 5 bus)
        for b in preds:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            if cls_id in [7, 5] and conf >= 0.20:
                # 排除右側廠房
                if not (xyxy[0] > 280 and xyxy[2] > 480 and xyxy[1] > 100):
                    boxes.append(xyxy)

        # 4. 去除重疊方框 (簡單 NMS)
        clean_boxes = []
        for box in boxes:
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area < 200:
                continue
            duplicate = False
            for cb in clean_boxes:
                # 計算 IoU
                ix1 = max(box[0], cb[0])
                iy1 = max(box[1], cb[1])
                ix2 = min(box[2], cb[2])
                iy2 = min(box[3], cb[3])
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                union = box_area + (cb[2]-cb[0])*(cb[3]-cb[1]) - inter
                if union > 0 and inter / union > 0.6:
                    duplicate = True
                    break
            if not duplicate:
                clean_boxes.append(box)

        # 特別保障：Frame 96 以後是純背景空路面 (包含被大燈照亮、有灰塵泥土的空路)，設定 boxes 為空 (負樣本)
        if fid >= 96 and fid not in [112, 113]: # 部分遠景車輛外
            # 清空非車輛的誤檢 (例如藍色牆壁)
            clean_boxes = [b for b in clean_boxes if not (b[0] > 270 and b[2] > 400)]

        labels_map[fn] = clean_boxes

    # 分割資料集 (80% train, 20% val)
    np.random.seed(42)
    shuffled_names = list(labels_map.keys())
    np.random.shuffle(shuffled_names)
    
    split_idx = int(0.8 * len(shuffled_names))
    train_names = shuffled_names[:split_idx]
    val_names = shuffled_names[split_idx:]

    print(f"訓練集 (Train): {len(train_names)} 張, 驗證集 (Val): {len(val_names)} 張")

    for split, names in [("train", train_names), ("val", val_names)]:
        for fn in names:
            src_img = raw_dir / fn
            dst_img = dataset_dir / "images" / split / fn
            shutil.copy2(src_img, dst_img)

            # 寫入 YOLO 標註
            boxes = labels_map[fn]
            dst_txt = dataset_dir / "labels" / split / f"{Path(fn).stem}.txt"
            with open(dst_txt, "w", encoding="utf-8") as f:
                for b in boxes:
                    yolo_line = to_yolo_format(b)
                    if yolo_line:
                        f.write(yolo_line)

    # 建立 data.yaml
    yaml_content = f"""path: {dataset_dir.resolve().as_posix()}
train: images/train
val: images/val
names:
  0: truck
"""
    yaml_file = dataset_dir / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"✅ 資料集配置檔已生成: {yaml_file}")

    # 輸出 10 張預覽標註供抽驗確認
    for fn in train_names[:10]:
        img = cv2.imread(str(raw_dir / fn))
        for b in labels_map[fn]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(img, "truck", (int(b[0]), max(15, int(b[1])-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(preview_dir / f"labeled_{fn}"), img)

    print(f"✅ 標註預覽圖片已輸出至: {preview_dir}")
    print("=" * 68)

if __name__ == "__main__":
    main()
