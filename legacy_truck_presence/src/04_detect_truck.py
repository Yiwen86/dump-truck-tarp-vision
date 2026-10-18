#!/usr/bin/env python3
"""
腳本: src/04_detect_truck.py
=============================================================================
方案二: 工業界混合雙重檢測管線 (YOLO + 車道虛擬檢測線圈 Virtual Loop ROI)
-----------------------------------------------------------------------------
功能：
1. 解決俯視監視器（Bird's-eye View）下的三大痛點：
   - 遠景/中景：由 YOLO 目標偵測模型抓取完整車輛。
   - 手前近景：車頭被鏡頭裁切、僅露出車頂與車斗帆布時，由「車道虛擬檢測線圈」
     透過車道佔用分析精準捕捉，徹底消滅漏檢！
2. 視覺化疊加標註：
   - YOLO 檢出：以方框標註車輛位置與信心度。
   - 虛擬線圈觸發：繪製車道佔用檢測框，標註 [TRUCK] Foreground Bed/Cabin。
   - 頂部狀態橫幅：即時顯示 [TRUCK DETECTED] 或 [NO TRUCK DETECTED] Road Idle。
3. 輸出結構化 CSV 報表，詳細記錄觸發來源 (YOLO、Virtual Loop 或兩者同時觸發)。
=============================================================================
"""

import os
import sys
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# COCO 類別對照
# 7: truck (卡車 / 砂石車)
# 5: bus (大客車 / 俯視長條重車)
DEFAULT_CLASSES = [7, 5]


def parse_args():
    parser = argparse.ArgumentParser(
        description="工業級混合卡車判定 (YOLO + 車道虛擬線圈 Virtual Loop)"
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
        default="yolov8n.pt",
        help="YOLO 目標偵測權重檔 (預設: yolov8n.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/truck_detection",
        help="結果圖片輸出目錄 (預設: runs/truck_detection)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO 卡車偵測信心度門檻值 (預設: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IOU 門檻值 (預設: 0.45)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=DEFAULT_CLASSES,
        help="欲偵測的 COCO 類別 ID (預設: 7 5 [truck, bus])",
    )
    parser.add_argument(
        "--disable_virtual_loop",
        action="store_true",
        help="停用車道虛擬線圈檢測 (僅使用 YOLO)",
    )
    parser.add_argument(
        "--loop_median_thresh",
        type=float,
        default=65.0,
        help="虛擬線圈車道中位數亮度門檻值 (預設: 65.0)",
    )
    parser.add_argument(
        "--loop_bright_thresh",
        type=float,
        default=0.50,
        help="虛擬線圈明亮像素 (>70) 佔比門檻值 (預設: 0.50)",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        default=True,
        help="是否儲存偵測結果 CSV 清單 (預設: True)",
    )
    parser.add_argument(
        "--no_csv",
        action="store_false",
        dest="save_csv",
        help="不儲存 CSV 清單",
    )
    return parser.parse_args()


def check_virtual_loop(img, median_thresh=65.0, bright_thresh=0.50):
    """
    方案二: 車道虛擬檢測線圈 (Virtual Loop ROI)
    監控畫面中央核心車道，透過柏油路面被大面積車頂/車斗覆蓋的物理特徵進行判定
    """
    h, w = img.shape[:2]
    # 車道核心虛擬線圈幾何座標 (x1, y1, x2, y2)
    x1 = int(0.25 * w)
    y1 = int(0.40 * h)
    x2 = int(0.65 * w)
    y2 = int(0.88 * h)

    roi = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    median_val = float(np.median(roi))
    ratio_bright = float(np.count_nonzero(roi > 70) / roi.size)

    # 判定標準：路面中位數顯著高於柏油黑度 (46) 且明亮像素超過 50%
    is_occupied = (median_val > median_thresh) and (ratio_bright > bright_thresh)

    return is_occupied, (x1, y1, x2, y2), median_val, ratio_bright


def draw_bounding_box(img, box, label_text, color=(0, 210, 80)):
    """
    繪製目標框與上方文字 Banner
    """
    x1, y1, x2, y2 = map(int, box)

    # 1. 外框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)

    # 2. 標籤底色塊
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    banner_y1 = max(0, y1 - text_h - 10)
    banner_y2 = y1

    cv2.rectangle(img, (x1, banner_y1), (x1 + text_w + 12, banner_y2), color, thickness=-1)
    # 文字
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


def draw_status_header(img, has_truck, source_desc=""):
    """
    畫面左上方繪製全景狀態標籤
    """
    if has_truck:
        text = f"[TRUCK DETECTED] {source_desc}"
        bg_color = (0, 160, 40)  # 翠綠色
    else:
        text = "[NO TRUCK DETECTED] Road Idle"
        bg_color = (80, 80, 80)  # 灰色

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

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[錯誤] 權重模型檔案不存在: {args.weights}")
        sys.exit(1)

    use_virtual_loop = not args.disable_virtual_loop

    print("=" * 68)
    print(" 🚛 工業級混合卡車判定系統 (方案二: YOLO + 車道虛擬線圈 Virtual Loop)")
    print(f" 輸入影像來源: {input_path.resolve()}")
    print(f" YOLO 偵測權重: {weights_path.resolve()} (conf >= {args.conf})")
    print(f" 支援類別 ID: {args.classes} (7: truck, 5: bus/大型載貨車)")
    print(f" 車道虛擬線圈: {'已啟用 (有效補足俯視手前大車漏檢)' if use_virtual_loop else '已停用'}")
    print(f" 結果輸出目錄: {output_dir.resolve()}")
    print("=" * 68)

    # 載入 YOLO 模型
    detector = YOLO(str(weights_path))

    # 收集待處理圖片
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = [p for p in input_path.iterdir() if p.suffix.lower() in valid_exts]

    total_images = len(image_paths)
    if total_images == 0:
        print("[警告] 目錄內無有效圖片！")
        return

    print(f"待分析圖片數量: {total_images}\n開始進行混合判定...")

    results_record = []
    truck_present_images = 0
    no_truck_images = 0
    yolo_only_count = 0
    loop_only_count = 0
    both_count = 0

    for img_path in tqdm(image_paths, desc="Hybrid Detection"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        annotated_img = img.copy()

        # 1. 執行 YOLO 目標偵測 (中/遠景車輛)
        det_results = detector.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            verbose=False,
        )

        boxes = det_results[0].boxes
        yolo_detected = len(boxes) > 0

        # 2. 執行車道虛擬線圈佔用分析 (手前近景俯視大車)
        loop_occupied = False
        loop_box = None
        loop_median = 0.0
        loop_bright = 0.0

        if use_virtual_loop:
            loop_occupied, loop_box, loop_median, loop_bright = check_virtual_loop(
                img,
                median_thresh=args.loop_median_thresh,
                bright_thresh=args.loop_bright_thresh,
            )

        # 3. 綜合判定
        has_truck = yolo_detected or loop_occupied

        detection_source = "None"
        truck_count = len(boxes)

        if has_truck:
            truck_present_images += 1

            if yolo_detected and loop_occupied:
                both_count += 1
                detection_source = "YOLO + Virtual Loop"
                header_desc = f"{truck_count} truck(s) [Dual Confirmed]"
            elif loop_occupied:
                loop_only_count += 1
                detection_source = "Virtual Loop Only"
                header_desc = "Foreground Truck Present"
                truck_count = max(1, truck_count)
            else:
                yolo_only_count += 1
                detection_source = "YOLO Only"
                header_desc = f"{truck_count} truck(s) [YOLO]"

            # 繪製 YOLO 偵測框
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_label = "Truck" if cls_id == 7 else "Truck/Heavy"
                draw_bounding_box(
                    annotated_img,
                    xyxy,
                    f"{class_label} {conf*100:.1f}%",
                    color=(0, 210, 80),
                )

            # 繪製車道虛擬線圈框 (如果被佔用，且畫面下方未被 YOLO 框完整覆蓋)
            if loop_occupied:
                draw_bounding_box(
                    annotated_img,
                    loop_box,
                    f"[TRUCK] Close-up Bed/Cab ({loop_median:.0f})",
                    color=(0, 180, 255),  # 鮮明橙黃色
                )

            draw_status_header(annotated_img, has_truck=True, source_desc=header_desc)

        else:
            no_truck_images += 1
            draw_status_header(annotated_img, has_truck=False)

        # 儲存標註後影像
        out_file = output_dir / f"det_{img_path.name}"
        cv2.imwrite(str(out_file), annotated_img)

        # 紀錄結構化資訊
        results_record.append(
            {
                "filename": img_path.name,
                "has_truck": "YES" if has_truck else "NO",
                "detection_source": detection_source,
                "truck_count": truck_count,
                "virtual_loop_occupied": "YES" if loop_occupied else "NO",
                "virtual_loop_median": f"{loop_median:.1f}",
                "virtual_loop_bright_ratio": f"{loop_bright*100:.1f}%",
            }
        )

    # 輸出 CSV 報表
    if args.save_csv and results_record:
        csv_path = output_dir / "truck_detection_summary.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
            fieldnames = [
                "filename",
                "has_truck",
                "detection_source",
                "truck_count",
                "virtual_loop_occupied",
                "virtual_loop_median",
                "virtual_loop_bright_ratio",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_record)
        csv_msg = f" 📊 CSV 結構化摘要清單已輸出: {csv_path.resolve()}"
    else:
        csv_msg = ""

    # 統計結果列印
    truck_rate = (truck_present_images / total_images * 100) if total_images > 0 else 0
    idle_rate = (no_truck_images / total_images * 100) if total_images > 0 else 0

    print("\n" + "=" * 68)
    print(" 🏁 工業級混合卡車判定完成報告 (Hybrid Truck Presence Summary)")
    print("=" * 68)
    print(f" 總分析圖片數: {total_images} 張")
    print(f" 🚛 【有卡車】圖片數: {truck_present_images} 張 ({truck_rate:.1f}%)")
    print(f"    - 雙重同時確認 (YOLO + 線圈): {both_count} 張")
    print(f"    - 🌟 虛擬線圈成功救回漏檢 (手前俯視大車): {loop_only_count} 張")
    print(f"    - YOLO 遠/中景檢出: {yolo_only_count} 張")
    print(f" 🛣️ 【無卡車 / 空道路】: {no_truck_images} 張 ({idle_rate:.1f}%)")
    print(f" 🖼️ 視覺化標註圖片輸出於: {output_dir.resolve()}")
    if csv_msg:
        print(csv_msg)
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
