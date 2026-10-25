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

## 📁 專案目錄結構 (已全面整合優化：單一主線 + 完整封存)

```text
dump-truck-tarp/
├── dataset/                          # 統一為單一標準資料集 (2,731 張完整資料)
│   ├── train/ & valid/ & test/       # 嚴格分層抽樣 (Stratified Split)
│   ├── raw_overhead_57/              # 57 張真實路況原始檔案持久備份
│   └── data.yaml                     # 資料集標準設定檔
├── weights/                          # 核心模型權重
│   ├── best.pt                       # 🔥 唯一主力模型 (現役 V2, mAP50=95.3%)
│   └── archive/                      # 過程權重歸檔 (best_tarp_v1.pt, yolo26n.pt 等)
├── src/                              # 精簡清晰的主線工作流
│   ├── train.py                      # 一鍵訓練 / 微調最新模型
│   ├── detect.py                     # 一鍵推論判定與 CSV 報表導出
│   ├── app_demo.py                   # Web 視覺化即時展示系統 (Flask)
│   └── tools/                        # 輔助與挑圖工具 (extract_unique_767, build_dataset)
├── runs/                             # 實驗與檢測結果 (層級化收納)
│   ├── train/                        # 現役最優模型訓練日誌、曲線圖與混淆矩陣
│   ├── eval_test/                    # 測試集推論成果圖與 CSV 報表
│   ├── eval_overhead/                # 57 張真實俯視路況成果圖與 CSV 報表
│   └── archive/                      # 舊版歷程結果封存
├── archive/                          # 📦 完整保留所有歷史研發歷程 (中繼過程全數保留)
│   ├── 01_legacy_truck_presence/     # 第一階段：卡車存在性判定 (線圈與舊模型)
│   ├── 02_curation_pool_767/         # 767 張基底去重篩選池
│   ├── 03_v1_dataset_backup/         # V1 早期獨立資料集備份
│   └── history_scripts/              # 歷史演進腳本 (01, 02, 03, 06, 07 階段腳本)
├── pyproject.toml / uv.lock
└── README.md
```

---

## 🚀 執行方式

### 1. 執行帆布合規性推論判定 (`src/detect.py`)
```bash
# 批次判定指定資料夾內的圖片 (預設自動調用 weights/best.pt)
.venv\Scripts\python.exe src/detect.py --input <圖片資料夾路徑> --output_dir runs/detect --device 0

# 測試預設測試集
.venv\Scripts\python.exe src/detect.py --input dataset/test/images --output_dir runs/eval_test --device 0

# 測試真實路況 57 張俯視圖
.venv\Scripts\python.exe src/detect.py --input dataset/raw_overhead_57 --output_dir runs/eval_overhead --device 0
```

### 2. 啟動 Web 視覺化互動 DEMO (`src/app_demo.py`)
```bash
.venv\Scripts\python.exe src/app_demo.py
```
* 開啟瀏覽器訪問 `http://127.0.0.1:5000` 即可拖曳圖片或使用範例進行即時合規判定（系統自動優先調用 `weights/best.pt`）。

### 3. 一鍵重新微調/訓練模型 (`src/train.py`)
```bash
.venv\Scripts\python.exe src/train.py
```
* 自動調用 NVIDIA RTX 4060 GPU，基於 `dataset/data.yaml` 進行訓練，並自動更新產出 `weights/best.pt`。

---

## 📝 專案說明與重構歷程
完整指標對照、目錄整理與腳本序號細節，請參閱：
* [專案現況與進度完整說明報告 (PROJECT_STATUS.md)](PROJECT_STATUS.md)
* [專案重構記錄 (REFACTORING_LOG.md)](REFACTORING_LOG.md)
