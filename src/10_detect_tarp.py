#!/usr/bin/env python3
"""
腳本: src/10_detect_tarp.py
=============================================================================
功能:
卡車車斗帆布覆蓋合規性推論判定腳本 (Tarp Compliance Inference)
1. 載入微調訓練後的專屬權重 (runs/tarp_training/yolov8n_tarp/weights/best.pt)。
2. 輸入圖片或資料夾，進行端到端車斗帆布檢測。
3. 視覺化疊加標註：
   - 🟢 綠色框: covered (合規有蓋好)
   - 🔴 紅色框: uncovered (違規未覆蓋/裸露)
   - 橫幅狀態列: 即時判定整車是否合規。
4. 輸出結構化 CSV 報表。
=============================================================================
"""

import sys
import argparse
import csv
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="卡車車斗帆布覆蓋合規性推論系統"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset_tarp/test/images",
        help="輸入圖片路徑或資料夾 (預設: dataset_tarp/test/images)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/tarp_training/yolov8n_tarp/weights/best.pt",
        help="專屬權重檔路徑 (預設: runs/tarp_training/yolov8n_tarp/weights/best.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/tarp_evaluation",
        help="結果輸出目錄 (預設: runs/tarp_evaluation)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="判定信心度門檻 (預設: 0.30)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IOU 門檻 (預設: 0.45)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推論設備 (預設: 0 為 GPU, 或 cpu)",
    )
    return parser.parse_args()

def draw_tarp_annotations(image, boxes, clses, confs):
    annotated = image.copy()
    h, w = annotated.shape[:2]

    has_uncovered = False
    has_covered = False

    for box, cls_id, conf in zip(boxes, clses, confs):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)
        
        if cls_id == 0:
            # covered: 綠色
            color = (0, 200, 0)
            label = f"Covered {conf * 100:.1f}%"
            has_covered = True
        else:
            # uncovered: 紅色
            color = (0, 0, 255)
            label = f"UNCOVERED {conf * 100:.1f}%"
            has_uncovered = True

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # 頂部狀態橫幅
    if has_uncovered:
        banner_text = "[VIOLATION] Uncovered Bed Detected (Non-compliant)"
        banner_color = (0, 0, 220)
    elif has_covered:
        banner_text = "[COMPLIANT] Tarp Properly Covered"
        banner_color = (0, 160, 0)
    else:
        banner_text = "[IDLE / NO TARGET] No Dump Bed Detected"
        banner_color = (80, 80, 80)

    cv2.rectangle(annotated, (0, 0), (w, 36), banner_color, -1)
    cv2.putText(annotated, banner_text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return annotated, "NON_COMPLIANT" if has_uncovered else ("COMPLIANT" if has_covered else "NONE")

def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[Error] Weights file not found: {weights_path}")
        sys.exit(1)

    print("=" * 68)
    print(" 🚀 Starting Tarp Compliance Evaluation")
    print(f" Weights: {weights_path}")
    print(f" Input: {input_path}")
    print(f" Output: {output_dir}")
    print("=" * 68)

    model = YOLO(str(weights_path))

    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))

    if not image_files:
        print("[Error] No images found to process.")
        sys.exit(1)

    csv_path = output_dir / "tarp_compliance_summary.csv"
    csv_rows = []

    for img_path in tqdm(image_files, desc="Evaluating"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        clses = res.boxes.cls.cpu().numpy() if res.boxes is not None else []
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []

        annotated_img, status = draw_tarp_annotations(img, boxes, clses, confs)

        out_img_path = output_dir / f"eval_{img_path.name}"
        cv2.imwrite(str(out_img_path), annotated_img)

        csv_rows.append({
            "filename": img_path.name,
            "compliance_status": status,
            "detected_targets": len(boxes),
            "max_confidence": f"{max(confs):.4f}" if len(confs) > 0 else "0.0",
        })

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "compliance_status", "detected_targets", "max_confidence"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("=" * 68)
    print(" [DONE] Inference finished!")
    print(f" Results saved to: {output_dir}")
    print(f" Summary CSV: {csv_path}")
    print("=" * 68)

if __name__ == "__main__":
    main()
