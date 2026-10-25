#!/usr/bin/env python3
"""
腳本: src/07_train_tarp_v2.py
=============================================================================
功能:
一鍵啟動 YOLOv8 V2 升級版模型訓練 (Domain Fine-Tuning)
1. 載入第一版最佳權重 weights/best_tarp_v1.pt。
2. 資料集: dataset_v2/data.yaml (含原 2,674 張基礎資料 + 新增 57 張真實路況俯視圖)。
3. 使用 GPU (NVIDIA GeForce RTX 4060) 進行端到端微調訓練。
4. 輸出最佳權重至 runs/tarp_v2_training/weights/best.pt 並自動保存為 weights/best_tarp_v2.pt。
=============================================================================
"""

import sys
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

# 確保 Windows 命令列 UTF-8 輸出安全
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def main():
    print("=" * 70)
    print(" [TRAIN V2] Starting YOLOv8 Tarp V2 Domain Fine-Tuning on RTX 4060")
    print("=" * 70)

    # 1. 檢查 CUDA GPU 狀態
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

    # 2. 資料集路徑檢查
    yaml_path = Path("dataset_v2/data.yaml").resolve()
    if not yaml_path.exists():
        print(f"[Error] data.yaml not found: {yaml_path}")
        print("Please run src/06_import_and_build_v2_dataset.py first!")
        sys.exit(1)

    # 3. 載入 V1 最佳模型權重
    v1_weights_candidates = [
        Path("weights/best_tarp_v1.pt").resolve(),
        Path("runs/tarp_training/weights/best.pt").resolve(),
    ]
    base_weights = None
    for cand in v1_weights_candidates:
        if cand.exists():
            base_weights = cand
            break

    if base_weights is None:
        print("[Error] weights/best_tarp_v1.pt not found! Please check weights/ directory.")
        sys.exit(1)

    print(f" Dataset config : {yaml_path}")
    print(f" Base V1 weights: {base_weights}")
    print(f" Target device  : {device}")
    print(" Epochs         : 25 | Image size: 640 | Batch: 16 | lr0: 0.002")
    print("=" * 70)

    # 4. 載入模型
    model = YOLO(str(base_weights))

    runs_dir = Path("runs").resolve()
    # 5. 啟動微調
    results = model.train(
        data=str(yaml_path),
        epochs=25,
        imgsz=640,
        batch=16,
        workers=2,
        device=device,
        project=str(runs_dir),
        name="tarp_v2_training",
        exist_ok=True,
        lr0=0.002,
        lrf=0.01,
        plots=True,
        verbose=True,
    )

    # 6. 自動備份產出 weights/best_tarp_v2.pt
    best_out = Path("runs/tarp_v2_training/weights/best.pt").resolve()
    weights_dest = Path("weights/best_tarp_v2.pt").resolve()

    if best_out.exists():
        weights_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_out, weights_dest)
        print("=" * 70)
        print(" [SUCCESS] V2 Model training completed successfully!")
        print(f" Best weights in runs : {best_out}")
        print(f" Exported V2 weights  : {weights_dest}")
        print("=" * 70)
    else:
        print("[Warning] best.pt not found in runs/tarp_v2_training/weights/")

if __name__ == "__main__":
    main()
