#!/usr/bin/env python3
"""
腳本: src/07_detect_custom_yolo.py
=============================================================================
方案一: 專屬微調 YOLOv8 卡車偵測端到端推論腳本 (Custom YOLO Detection)
-----------------------------------------------------------------------------
功能：
1. 載入微調訓練後的專屬權重 (runs/train/truck_custom_yolo/weights/best.pt)。
2. 完全捨棄傳統脆弱的虛擬線圈/亮度門檻，回歸純深度學習神經網絡端到端推論：
   - 精準捕捉手前俯視大車斗與帆布
   - 精準捕捉邊界裁切車身（哪怕只露 20% 車頭門把與輪胎）
   - 完全免疫車燈照射泥土地面引起的亮度漂移
3. 視覺化疊加標註與信心度顯示。
4. 輸出結構化 CSV 報表至 runs/custom_truck_detection/custom_truck_detection_summary.csv。
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
        description="方案一: 專屬微調 YOLO 卡車判定系統"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_images",
        help="輸入圖片路徑或資料夾 (預設: data/raw_images)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="微調後的專屬權重檔 (預設自動搜尋最佳權重)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/custom_truck_detection",
        help="結果輸出目錄 (預設: runs/custom_truck_detection)",
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
    return parser.parse_args()


def draw_bounding_box(img, box, label_text, color=(0, 215, 80)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)

    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    banner_y1 = max(0, y1 - text_h - 10)
    banner_y2 = y1

    cv2.rectangle(img, (x1, banner_y1), (x1 + text_w + 12, banner_y2), color, thickness=-1)
    cv2.putText(
        img,
        label_text,
        (x1 + 6, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_status_header(img, has_truck, truck_count=0):
    if has_truck:
        text = f"[TRUCK DETECTED] Count: {truck_count}"
        bg_color = (0, 160, 40)
    else:
        text = "[NO TRUCK DETECTED] Road Idle"
        bg_color = (80, 80, 80)

    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(img, (14, 14), (14 + text_w + 20, 14 + text_h + 16), bg_color, -1)
    cv2.putText(
        img,
        text,
        (24, 14 + text_h + 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[錯誤] 輸入路徑不存在: {args.input}")
        sys.exit(1)

    weights_candidates = [
        Path(args.weights) if args.weights else None,
        Path("runs/detect/runs/truck_custom_yolo/weights/best.pt"),
        Path("runs/truck_custom_yolo/weights/best.pt"),
        Path("runs/detect/truck_custom_yolo/weights/best.pt"),
        Path("runs/detect/runs/train/truck_custom_yolo/weights/best.pt"),
        Path("runs/train/truck_custom_yolo/weights/best.pt"),
    ]
    weights_path = None
    for cand in weights_candidates:
        if cand and cand.exists():
            weights_path = cand
            break

    if weights_path is None:
        print("[錯誤] 找不到微調權重 best.pt，請先確認微調訓練是否完成！")
        sys.exit(1)

    print("=" * 68)
    print(" 🚛 方案一: 專屬微調 YOLO 卡車判定系統 (Custom YOLO)")
    print(f" 影像輸入: {input_path.resolve()}")
    print(f" 專屬權重: {weights_path.resolve()} (conf >= {args.conf})")
    print(f" 輸出目錄: {output_dir.resolve()}")
    print("=" * 68)

    model = YOLO(str(weights_path))

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = [p for p in input_path.iterdir() if p.suffix.lower() in valid_exts]

    total_images = len(image_paths)
    if total_images == 0:
        print("[警告] 目錄內無有效圖片！")
        return

    results_record = []
    truck_images = 0
    idle_images = 0

    for img_path in tqdm(image_paths, desc="Custom YOLO Inference"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        annotated_img = img.copy()

        # 執行專屬微調模型預測
        res = model.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]

        boxes = res.boxes
        has_truck = len(boxes) > 0
        truck_count = len(boxes)

        detections_detail = []
        if has_truck:
            truck_images += 1
            max_conf = 0.0
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                max_conf = max(max_conf, conf)
                detections_detail.append({"box": [round(x, 1) for x in xyxy], "conf": round(conf, 4)})
                draw_bounding_box(
                    annotated_img,
                    xyxy,
                    f"Truck {conf*100:.1f}%",
                    color=(0, 215, 80),
                )
            draw_status_header(annotated_img, has_truck=True, truck_count=truck_count)
        else:
            idle_images += 1
            max_conf = 0.0
            draw_status_header(annotated_img, has_truck=False)

        # 儲存圖片
        out_file = output_dir / f"det_{img_path.name}"
        cv2.imwrite(str(out_file), annotated_img)

        results_record.append(
            {
                "filename": img_path.name,
                "has_truck": "YES" if has_truck else "NO",
                "truck_count": truck_count,
                "max_confidence": f"{max_conf:.4f}",
                "detections_detail": str(detections_detail),
            }
        )

    # 輸出 CSV
    csv_path = output_dir / "custom_truck_detection_summary.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["filename", "has_truck", "truck_count", "max_confidence", "detections_detail"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_record)

    truck_rate = (truck_images / total_images * 100) if total_images > 0 else 0
    idle_rate = (idle_images / total_images * 100) if total_images > 0 else 0

    print("\n" + "=" * 68)
    print(" 🏁 專屬微調 YOLO 檢測完成報告")
    print("=" * 68)
    print(f" 總分析圖片數: {total_images} 張")
    print(f" 🚛 【檢出卡車】: {truck_images} 張 ({truck_rate:.1f}%)")
    print(f" 🛣️ 【道路無車 / Idle】: {idle_images} 張 ({idle_rate:.1f}%)")
    print(f" 🖼️ 標註輸出: {output_dir.resolve()}")
    print(f" 📊 CSV 摘要: {csv_path.resolve()}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
