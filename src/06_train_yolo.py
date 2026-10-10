#!/usr/bin/env python3
"""
腳本: src/06_train_yolo.py
=============================================================================
方案一: 微調 (Fine-Tuning) 專屬 YOLOv8 模型
-----------------------------------------------------------------------------
功能：
1. 載入 yolov8n.pt 預訓練權重。
2. 針對本專案俯視監控路口資料集 (dataset/data.yaml) 進行遷移學習微調。
3. 訓練 25 個 epoch，專注優化「俯視車斗」、「嚴重裁切車身」與「夜間眩光」等特徵。
4. 輸出最佳權重至 runs/train/truck_custom_yolo/weights/best.pt。
=============================================================================
"""

import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    yaml_path = Path("dataset/data.yaml").resolve()
    if not yaml_path.exists():
        print(f"[錯誤] 資料集設定檔不存在: {yaml_path}")
        sys.exit(1)

    weights_path = Path("yolov8n.pt").resolve()
    if not weights_path.exists():
        print(f"[錯誤] 基礎權重不存在: {weights_path}")
        sys.exit(1)

    print("=" * 68)
    print(" 🚀 開始執行方案一: 微調 (Fine-Tuning) 專屬 YOLO 卡車偵測模型")
    print(f" 基礎模型: {weights_path}")
    print(f" 資料集配置: {yaml_path}")
    print(" 訓練 Epoch: 25")
    print(" 影像尺寸: 512")
    print(" 設備: CPU")
    print("=" * 68)

    model = YOLO(str(weights_path))

    # 執行微調 (CPU 優化配置)
    results = model.train(
        data=str(yaml_path),
        epochs=10,
        imgsz=384,
        batch=16,
        workers=0,
        device="cpu",
        project="runs",
        name="truck_custom_yolo",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("=" * 68)
    print(" 🏁 專屬模型微調完成！")
    print(" 權重存放目錄: runs/truck_custom_yolo/weights/best.pt")
    print("=" * 68)

if __name__ == "__main__":
    main()
