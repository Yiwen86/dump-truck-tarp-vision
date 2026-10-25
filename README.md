# 卡車車斗帆布覆蓋合規性判定專案 (Dump Truck Tarp Coverage Detection)

本專案專注於利用 **YOLOv8 深度學習神經網絡**，針對道路與路口監視器影像，即時精準判定**砂石車/卡車車斗上方「帆布是否有蓋好」**。

> 📄 **專案完整進度報告**：請參閱 [專案現況與進度完整說明報告 (PROJECT_STATUS.md)](PROJECT_STATUS.md) 查閱完整 V1/V2 Benchmark、真實俯視樣本評估數據與架構全覽。

---

## 🎯 核心辨識類別 (Class Definitions)

模型直接針對車斗區域框選並分類為兩大合規狀態：

| 類別 ID | 類別名稱 | 意義說明 | 畫面標註邊框 | 狀態判決 |
| :---: | :--- | :--- | :---: | :---: |
| **`0`** | **`covered`** | 帆布完整覆蓋車斗（含正常裝載與超載有蓋） | 🟢 綠色框 | **[COMPLIANT] 合格** |
| **`1`** | **`uncovered`** | 車斗裸露、未蓋帆布、砂石土石外露（違規） | 🔴 紅色框 | **[VIOLATION] 違規** |

---

## 📊 模型訓練成果與指標 (NVIDIA RTX 4060 GPU 加速)

### 🌟 V2 最新模型 (Domain Fine-Tuning 領域融合微調)
融合原資料集與 57 張真實台灣路況俯視圖進行分層抽樣微調，**mAP@50 飆升至 95.3%**：

* **訓練資料量**：2,731 張影像（Train: 1,917 張 / Val: 546 張 / Test: 268 張）
* **單張推論延遲**：約 **1.5 毫秒**（~660 FPS 極高即時性）
* **核心權重**：[`weights/best_tarp_v2.pt`](weights/best_tarp_v2.pt)

| 評估類別 | 實例數 (Instances) | 精確率 (Precision) | 召回率 (Recall) | **mAP@50 (核心指標)** | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **總體平均 (All V2)** | **687** | **87.9%** (+2.9%) | **91.0%** (+4.6%) | **🔥 95.3% (+3.1%)** | **73.4% (+4.5%)** |
| 🟢 **`covered` (帆布蓋好)** | 323 | 88.9% | 94.1% | **🌟 97.2% (+3.3%)** | 77.1% |
| 🔴 **`uncovered` (未蓋好)** | 364 | 86.9% | 87.9% | **🌟 93.5% (+3.0%)** | 69.7% |

<details>
<summary>點擊展開：查看 V1 基準模型表現 (mAP@50: 92.2%)</summary>

| 評估類別 | 實例數 (Instances) | 精確率 (Precision) | 召回率 (Recall) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **總體平均 (All V1)** | 676 | 85.0% | 86.4% | 92.2% | 68.9% |
| 🟢 `covered` | 322 | 85.8% | 89.8% | 93.9% | 73.2% |
| 🔴 `uncovered` | 354 | 84.1% | 83.1% | 90.5% | 64.6% |

</details>

---

## 📁 專案目錄結構 (已全面整合優化)

```text
dump-truck-tarp/
├── dataset_v2/                       # 🔥 V2 融合資料集 (2,731 張，含真實俯視路況)
│   ├── train/ & valid/ & test/       # 嚴格分層抽樣 (Stratified Split)
│   ├── raw_downloaded_57/            # 57 張原始下載真實圖檔持久備份
│   └── data.yaml                     # V2 類別與路徑設定檔
├── dataset_tarp/                     # V1 帆布覆蓋專屬 YOLO 資料集 (2,674 張)
├── weights/                          # 核心模型權重庫
│   ├── best_tarp_v2.pt               # 🔥 最新 V2 模型 (mAP50=95.3%，優先載入)
│   ├── best_tarp_v1.pt               # V1 基準模型 (mAP50=92.2%)
│   ├── yolov8n.pt                    # YOLOv8 官方基底預訓練權重
│   └── yolo26n.pt                    # 輔助/備用權重
├── src/                              # 模組化工作流腳本
│   ├── 01_import_and_convert_dataset.py # 資料集匯入與 4 轉 2 類別轉換腳本
│   ├── 02_train_tarp_yolo.py        # V1 模型訓練腳本
│   ├── 03_detect_tarp.py            # 帆布合規性端到端推論腳本 (自動優先載入 V2 權重)
│   ├── 04_extract_unique_767.py     # 767 張不重複原始基底影像去重提煉腳本
│   ├── 05_build_dataset_from_selected.py # 精華挑選圖片資料集構建腳本
│   ├── 06_import_and_build_v2_dataset.py # 匯入外部真實路況並進行分層資料融合
│   ├── 07_train_tarp_v2.py          # V2 模型 GPU 微調訓練腳本 (從 V1 接續微調)
│   └── app_demo.py                  # 即時視覺化 Web 互動展示系統 (Flask)
├── runs/
│   ├── tarp_v2_training/            # V2 訓練日誌、曲線圖、混淆矩陣與權重
│   ├── tarp_v2_test_eval/           # V2 測試集推論成果圖與 CSV 統計報表
│   ├── v2_eval_real_overhead/       # V2 在真實俯視圖上的推論結果
│   └── tarp_training/               # V1 歷史訓練日誌與權重
├── curation_pool_767/                # 人工高品質圖片篩選池 (767 張)
├── legacy_truck_presence/            # 📦 歷史封存：先前舊的「卡車存在性 (Presence)」資料與程式
├── pyproject.toml / uv.lock
└── README.md
```

---

## 🚀 執行方式

### 1. 執行帆布合規性推論判定 (`src/03_detect_tarp.py`)
```bash
# 批次判定指定資料夾內的圖片 (自動優先載入 weights/best_tarp_v2.pt)
.venv\Scripts\python.exe src/03_detect_tarp.py --input <圖片資料夾路徑> --output_dir runs/my_results --device 0

# 測試預設測試集
.venv\Scripts\python.exe src/03_detect_tarp.py --input dataset_tarp/test/images --output_dir runs/tarp_v2_test_eval --device 0

# 測試真實路況 57 張俯視圖
.venv\Scripts\python.exe src/03_detect_tarp.py --input dataset_v2/raw_downloaded_57 --output_dir runs/v2_eval_real_overhead --device 0
```

### 2. 啟動 Web 視覺化互動 DEMO (`src/app_demo.py`)
```bash
.venv\Scripts\python.exe src/app_demo.py
```
* 開啟瀏覽器訪問 `http://127.0.0.1:5000` 即可拖曳圖片或使用範例進行即時合規判定（系統自動優先調用 V2 權重）。

### 3. 一鍵重新微調 V2 模型 (`src/07_train_tarp_v2.py`)
```bash
.venv\Scripts\python.exe src/07_train_tarp_v2.py
```
* 自動調用 NVIDIA RTX 4060 GPU。
* 自動以 `weights/best_tarp_v1.pt` 為基底進行自適應領域微調，並產出 `weights/best_tarp_v2.pt`。

---

## 📝 專案說明與重構歷程
完整指標對照、目錄整理與腳本序號細節，請參閱：
* [專案現況與進度完整說明報告 (PROJECT_STATUS.md)](PROJECT_STATUS.md)
* [專案重構記錄 (REFACTORING_LOG.md)](REFACTORING_LOG.md)
