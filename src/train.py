#!/usr/bin/env python3
"""
腳本: src/train.py
=============================================================================
功能:
一鍵啟動 YOLOv8 砂石車車斗帆布覆蓋合規性模型 GPU 訓練/微調主程式
1. 載入基準模型 weights/best.pt (或 weights/archive/yolov8n.pt)。
2. 資料集: dataset/data.yaml (含 2,731 張影像之融合資料集)。
3. 使用 GPU (NVIDIA RTX 4060) 端到端加速訓練。
4. 輸出最佳權重至 runs/train/weights/best.pt 並自動更新同步至 weights/best.pt。
=============================================================================
"""

import sys
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

# 確保 Windows UTF-8 輸出安全
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def main():
    print("=" * 70)
    print(" [TRAIN] 啟動 YOLOv8 砂石車帆布合規判定模型訓練 (RTX 4060 加速)")
    print("=" * 70)

    # 1. 檢查 CUDA GPU 狀態
    cuda_available = torch.cuda.is_available()
    print(f" CUDA 可用狀態: {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f" 運算裝置 : {gpu_name} (共 {gpu_count} 裝置)")
        device = 0
    else:
        print(" [警告] 未檢測到 CUDA GPU，將自動退回 CPU 運算")
        device = "cpu"

    # 2. 資料集配置路徑檢查
    yaml_path = Path("dataset/data.yaml").resolve()
    if not yaml_path.exists():
        print(f"[Error] 未找到 data.yaml: {yaml_path}")
        sys.exit(1)

    # 3. 權重載入優先級
    candidates = [
        Path("weights/best.pt").resolve(),
        Path("weights/archive/best_tarp_v2.pt").resolve(),
        Path("weights/archive/yolov8n.pt").resolve(),
        Path("yolov8n.pt").resolve(),
    ]
    base_weights = next((c for c in candidates if c.exists()), "yolov8n.pt")

    print(f" 資料集配置 : {yaml_path}")
    print(f" 初始基底權重: {base_weights}")
    print(f" 硬體運算裝置: {device}")
    print(" 訓練輪次 : 25 Epochs | 影像尺寸: 640 | Batch Size: 16 | 自適應微調")
    print("=" * 70)

    model = YOLO(str(base_weights))

    runs_dir = Path("runs").resolve()
    results = model.train(
        data=str(yaml_path),
        epochs=25,
        imgsz=640,
        batch=16,
        workers=2,
        device=device,
        project=str(runs_dir),
        name="train",
        exist_ok=True,
        lr0=0.002,
        lrf=0.01,
        plots=True,
        verbose=True,
    )

    # 自動保存至 weights/best.pt
    best_out = Path("runs/train/weights/best.pt").resolve()
    target_weights = Path("weights/best.pt").resolve()

    if best_out.exists():
        target_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_out, target_weights)
        print("=" * 70)
        print(" [SUCCESS] 模型訓練與微調圓滿完成！")
        print(f" 訓練產出最佳權重 : {best_out}")
        print(f" 正式同步部署權重 : {target_weights}")
        print("=" * 70)
    else:
        print("[Warning] 未能在 runs/train/weights/ 找到 best.pt")

if __name__ == "__main__":
    main()
