#!/usr/bin/env python3
"""
腳本: src/detect.py
=============================================================================
功能:
砂石車/卡車車斗帆布覆蓋合規性推論判定系統 (Tarp Compliance Inference)
1. 載入最優模型權重 (預設: weights/best.pt)。
2. 支援單張圖片或整個資料夾批次端到端檢測。
3. 視覺化標註成果：
   - 🟢 綠色邊框: covered (合規有蓋好)
   - 🔴 紅色邊框: uncovered (違規未覆蓋/土石裸露)
   - 頂部狀態橫幅: [COMPLIANT] / [VIOLATION] / [IDLE]
4. 自動導出結構化 CSV 合規統計報表。
=============================================================================
"""

import sys
import argparse
import csv
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# 確保 Windows UTF-8 輸出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def get_default_weights():
    candidates = [
        Path("weights/best.pt"),
        Path("weights/archive/best_tarp_v2.pt"),
        Path("runs/train/weights/best.pt"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "weights/best.pt"

def parse_args():
    parser = argparse.ArgumentParser(
        description="砂石車車斗帆布覆蓋合規性推論系統"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/test/images",
        help="輸入圖片路徑或資料夾 (預設: dataset/test/images)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=get_default_weights(),
        help="模型權重檔路徑 (預設: weights/best.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/detect",
        help="結果輸出目錄 (預設: runs/detect)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="信心度門檻 (預設: 0.30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推論硬體裝置 (例如 0 或 cpu)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(" 🚀 啟動帆布覆蓋合規性推論系統 (Tarp Compliance Inference)")
    print(f" 模型權重 : {weights_path}")
    print(f" 輸入來源 : {input_path}")
    print(f" 輸出目錄 : {output_dir}")
    print(f" 信心門檻 : {args.conf}")
    print("=" * 68)

    if not weights_path.exists():
        print(f"[Error] 權重檔案不存在: {weights_path}")
        sys.exit(1)

    model = YOLO(str(weights_path))

    # 收集圖片清單
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = [
            p for p in input_path.iterdir()
            if p.suffix.lower() in valid_exts
        ]
    else:
        print(f"[Error] 輸入路徑不存在: {input_path}")
        sys.exit(1)

    if not image_files:
        print(f"[Warning] 於 {input_path} 未找到任何有效影像。")
        sys.exit(0)

    # 準備 CSV 報表
    csv_path = output_dir / "tarp_compliance_summary.csv"
    csv_rows = []

    for img_p in tqdm(image_files, desc="執行推論進度"):
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        results = model.predict(
            source=img,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        clses = res.boxes.cls.cpu().numpy() if res.boxes is not None else []
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []

        annotated = img.copy()
        h, w = annotated.shape[:2]

        has_uncovered = False
        has_covered = False
        max_conf = float(max(confs)) if len(confs) > 0 else 0.0

        for box, cls_id, conf_val in zip(boxes, clses, confs):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls_id)

            if cls_id == 0:
                color = (0, 200, 0)
                label = f"Covered {conf_val * 100:.1f}%"
                has_covered = True
            else:
                color = (0, 0, 255)
                label = f"UNCOVERED {conf_val * 100:.1f}%"
                has_uncovered = True

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        if has_uncovered:
            status = "NON_COMPLIANT"
            banner_text = "[VIOLATION] Uncovered Bed Detected (Non-compliant)"
            banner_color = (0, 0, 220)
        elif has_covered:
            status = "COMPLIANT"
            banner_text = "[COMPLIANT] Tarp Properly Covered"
            banner_color = (0, 160, 0)
        else:
            status = "NONE"
            banner_text = "[IDLE] No Dump Bed Detected"
            banner_color = (80, 80, 80)

        cv2.rectangle(annotated, (0, 0), (w, 36), banner_color, -1)
        cv2.putText(
            annotated,
            banner_text,
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        out_img_path = output_dir / f"pred_{img_p.name}"
        cv2.imwrite(str(out_img_path), annotated)

        csv_rows.append({
            "filename": img_p.name,
            "compliance_status": status,
            "detected_targets": len(boxes),
            "max_confidence": round(max_conf, 4),
        })

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["filename", "compliance_status", "detected_targets", "max_confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("=" * 68)
    print(" [DONE] 推論完成！")
    print(f" 成果標註圖存放於: {output_dir}")
    print(f" 合規統計報表清單: {csv_path}")
    print("=" * 68)

if __name__ == "__main__":
    main()
