#!/usr/bin/env python3
"""
腳本: src/02_train_tarp_yolo.py
=============================================================================
功能:
一鍵啟動 YOLOv8 GPU 訓練 —— 專屬卡車車斗帆布覆蓋判定模型 (Tarp Detection)
1. 載入 weights/yolov8n.pt 預訓練權重。
2. 資料集: dataset_tarp/data.yaml (2 類別: covered / uncovered)。
3. 使用 GPU (NVIDIA RTX 4060) 進行端到端加速微調。
4. 輸出最佳權重至 runs/tarp_training/weights/best.pt 並同步備份至 weights/。
=============================================================================
"""

import sys
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    print("=" * 68)
    print(" [TRAIN] Starting YOLOv8 Tarp Coverage Model Training on GPU")
    print("=" * 68)

    # 檢查 CUDA GPU 狀態
    cuda_available = torch.cuda.is_available()
    print(f" CUDA Available: {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f" GPU Device: {gpu_name} (Total {gpu_count} device)")
        device = 0
    else:
        print(" [WARNING] CUDA not detected by PyTorch, falling back to CPU")
        device = "cpu"

    yaml_path = Path("dataset_tarp/data.yaml").resolve()
    if not yaml_path.exists():
        print(f"[Error] data.yaml not found: {yaml_path}")
        sys.exit(1)

    weights_candidates = [
        Path("weights/yolov8n.pt").resolve(),
        Path("yolov8n.pt").resolve(),
    ]
    weights_path = None
    for cand in weights_candidates:
        if cand.exists():
            weights_path = cand
            break
    if weights_path is None:
        print(f"[Warning] Local yolov8n.pt not found, downloading default...")
        weights_path = "yolov8n.pt"

    print(f" Dataset config: {yaml_path}")
    print(f" Base weights: {weights_path}")
    print(f" Target device: {device}")
    print(" Epochs: 25 | Image size: 640 | Batch: 16")
    print("=" * 68)

    model = YOLO(str(weights_path))

    runs_dir = Path("runs").resolve()
    results = model.train(
        data=str(yaml_path),
        epochs=25,
        imgsz=640,
        batch=16,
        workers=2,
        device=device,
        project=str(runs_dir),
        name="tarp_training",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    best_out = Path("runs/tarp_training/weights/best.pt").resolve()
    if best_out.exists():
        weights_dest = Path("weights/best_tarp_v1.pt").resolve()
        weights_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_out, weights_dest)
        print(f" [BACKUP] Copied best weights to: {weights_dest}")

    print("=" * 68)
    print(" [DONE] Model training completed successfully!")
    print(f" Best weights: runs/tarp_training/weights/best.pt (and weights/best_tarp_v1.pt)")
    print("=" * 68)

if __name__ == "__main__":
    main()
