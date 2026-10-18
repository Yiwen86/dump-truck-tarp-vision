# 卡車車斗帆布覆蓋合規性判定專案 (Dump Truck Tarp Coverage Detection)

本專案專注於利用 **YOLOv8 深度學習神經網絡**，針對道路與路口監視器影像，即時精準判定**砂石車/卡車車斗上方「帆布是否有蓋好」**。

---

## 🎯 核心辨識類別 (Class Definitions)

模型直接針對車斗區域框選並分類為兩大合規狀態：

| 類別 ID | 類別名稱 | 意義說明 | 畫面標註邊框 | 狀態判決 |
| :---: | :--- | :--- | :---: | :---: |
| **`0`** | **`covered`** | 帆布完整覆蓋車斗（含正常裝載與超載有蓋） | 🟢 綠色框 | **[COMPLIANT] 合格** |
| **`1`** | **`uncovered`** | 車斗裸露、未蓋帆布、砂石土石外露（違規） | 🔴 紅色框 | **[VIOLATION] 違規** |

---

## 📊 模型訓練成果與指標 (NVIDIA RTX 4060 GPU 加速)

* **訓練資料量**：2,674 張高畫質標註影像（Train: 1,868 張 / Val: 534 張 / Test: 268 張）
* **微調 Epoch**：25 個 Epoch（總耗時約 5.7 分鐘）
* **單張推論延遲**：約 **1.5 毫秒**（極高幀率 ~660 FPS）

| 評估類別 | 實例數 (Instances) | 精確率 (Precision) | 召回率 (Recall) | **mAP@50 (核心指標)** | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **總體平均 (All)** | **676** | **85.0%** | **86.4%** | **🔥 92.2%** | **68.9%** |
| 🟢 **`covered` (帆布蓋好)** | 322 | 85.8% | 89.8% | **🌟 93.9%** | 73.2% |
| 🔴 **`uncovered` (未蓋好)** | 354 | 84.1% | 83.1% | **🌟 90.5%** | 64.6% |

---

## 📁 專案目錄結構 (已整理優化)

```text
dump-truck-tarp/
├── dataset_tarp/                     # 帆布覆蓋專屬 YOLO 資料集 (2,674 張)
│   ├── train/                        # 訓練集 (1,868 張)
│   ├── valid/                        # 驗證集 (534 張)
│   ├── test/                         # 測試集 (268 張)
│   └── data.yaml                     # 資料集類別設定檔
├── src/
│   ├── 08_import_and_convert_dataset.py # 資料集匯入與 4 轉 2 類別轉換腳本
│   ├── 09_train_tarp_yolo.py        # GPU 一鍵式 YOLOv8 模型訓練腳本
│   └── 10_detect_tarp.py            # 帆布合規性端到端推論腳本 (支援圖檔/目錄/CSV)
├── runs/
│   ├── tarp_training/weights/best.pt # 訓練好的最佳模型權重 (mAP50=92.2%)
│   ├── tarp_test_eval/              # 268 張測試集推論成果與 CSV 報表
│   └── detect/runs/tarp_training/   # 訓練圖表 (Loss、PR曲線、混淆矩陣)
├── legacy_truck_presence/            # 📦 歷史封存：先前舊的「卡車存在性 (Presence)」資料與程式
│   ├── data/                         # 舊 421 張路況原始影像
│   ├── dataset/                      # 舊卡車存在標註資料集
│   ├── runs/                         # 舊檢測圖、線圈分析與微調權重
│   └── src/                          # 舊腳本 (04, 05, 06, 07)
├── yolov8n.pt                        # 基礎預訓練權重
├── pyproject.toml
└── README.md
```

---

## 🚀 執行方式

### 1. 執行帆布合規性推論判定 (`src/10_detect_tarp.py`)
```bash
# 批次判定指定資料夾內的圖片 (使用 GPU 加速)
.venv\Scripts\python.exe src/10_detect_tarp.py --input <圖片資料夾路徑> --output_dir runs/my_results --device 0

# 測試預設測試集
.venv\Scripts\python.exe src/10_detect_tarp.py --input dataset_tarp/test/images --output_dir runs/tarp_test_eval --device 0

# 判定單張圖片
.venv\Scripts\python.exe src/10_detect_tarp.py --input <圖片檔名.jpg> --output_dir runs/my_results --device 0
```

### 2. 重新啟動 GPU 訓練 (`src/09_train_tarp_yolo.py`)
```bash
.venv\Scripts\python.exe src/09_train_tarp_yolo.py
```
* 自動調用 NVIDIA RTX 4060 GPU。
* 輸出最佳權重至 `runs/tarp_training/weights/best.pt`。
