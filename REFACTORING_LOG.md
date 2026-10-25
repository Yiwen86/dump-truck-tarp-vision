# 專案重構與目錄整理記錄 (Refactoring Log)

**記錄時間**：2026-10-21  
**關聯 Git Commit**：`4b3ca38` (`refactor: consolidate weights, clean runs residue, update gitignore, and renumber active scripts to 01-05`)

---

## 📌 重構緣由與目標
本專案經歷了「卡車存在判定（Presence）」至「帆布覆蓋合規性判定（Tarp Coverage）」的演進，歷史檔案封存後導致現有主線腳本編號從 `08` 開始，且訓練暫存與權重分散於各層資料夾。本步驟旨在全面**集中權重、清理 YOLO 暫存巢狀目錄、建立防肥大規則，並重整主線工作流腳本**。

---

## 🔄 改變對照與清單

### 1. 模型權重統一收納 (Weights Consolidation)
將散落在根目錄與暫存中的權重統一收整至 [`weights/`](file:///c:/Users/USER/Desktop/dump-truck-tarp/weights)：

| 原位置 / 狀態 | 現位置 | 說明 |
| :--- | :--- | :--- |
| `yolov8n.pt` (根目錄散落) | [`weights/yolov8n.pt`](file:///c:/Users/USER/Desktop/dump-truck-tarp/weights/yolov8n.pt) | 官方 YOLOv8 Nano 基底預訓練權重 |
| `runs/tarp_training/weights/best.pt` | [`weights/best_tarp_v1.pt`](file:///c:/Users/USER/Desktop/dump-truck-tarp/weights/best_tarp_v1.pt) | 現行訓練最佳模型 (mAP50=92.2%)，具名化方便辨識與調用 |
| `weights/yolo26n.pt` | [`weights/yolo26n.pt`](file:///c:/Users/USER/Desktop/dump-truck-tarp/weights/yolo26n.pt) | 保留現有輔助/備用權重 |

---

### 2. YOLO Runs 輸出與殘留清理 (Runs Clean-up)
先前 Ultralytics YOLO 在微調訓練時自動於 `runs/detect/runs/tarp_training/yolov8n_tarp/` 產生巢狀目錄，造成圖表與權重路徑混亂：

* **整合圖表至正式目錄**：將損失曲線 (`results.png`)、PR曲線 (`BoxPR_curve.png`)、混淆矩陣 (`confusion_matrix.png`) 與權重 (`last.pt`) 全部整併移至 [`runs/tarp_training/`](file:///c:/Users/USER/Desktop/dump-truck-tarp/runs/tarp_training)。
* **徹底刪除殘留空目錄**：移除多餘且巢狀的 `runs/detect/` 資料夾。
* **現行 `runs/` 清爽架構**：
  - `runs/tarp_training/`：訓練過程全部圖表與最終權重
  - `runs/tarp_test_eval/`：268 張 test 集推論驗證圖與 CSV 統計
  - `runs/eval_desktop_overhead/`：51 張桌面真實砂石車俯視圖推論驗證圖與 CSV 統計

---

### 3. 主線腳本序號與邏輯重編 (Script Renumbering)
將原本接續歷史封存的 `08 ~ 12` 編號，重新梳理為現行帆布任務的標準生命週期序號 `01 ~ 05`：

| 原腳本檔名 | 新腳本檔名 | 程式內部改動與職責 |
| :--- | :--- | :--- |
| `src/08_import_and_convert_dataset.py` | [`src/01_import_and_convert_dataset.py`](file:///c:/Users/USER/Desktop/dump-truck-tarp/src/01_import_and_convert_dataset.py) | 標註 4 轉 2 類別轉換並構建 `dataset_tarp/`。 |
| `src/09_train_tarp_yolo.py` | [`src/02_train_tarp_yolo.py`](file:///c:/Users/USER/Desktop/dump-truck-tarp/src/02_train_tarp_yolo.py) | 預設改由 `weights/yolov8n.pt` 載入；訓練完畢自動將 `best.pt` 備份至 `weights/best_tarp_v1.pt`；輸出目錄修正為 `runs/tarp_training/`。 |
| `src/10_detect_tarp.py` | [`src/03_detect_tarp.py`](file:///c:/Users/USER/Desktop/dump-truck-tarp/src/03_detect_tarp.py) | 預設權重查詢路徑優先指向 `weights/best_tarp_v1.pt`。 |
| `src/11_extract_unique_767.py` | [`src/04_extract_unique_767.py`](file:///c:/Users/USER/Desktop/dump-truck-tarp/src/04_extract_unique_767.py) | 提煉 767 張去重原始圖檔並建立綠框/紅框預覽。 |
| `src/12_build_dataset_from_selected.py` | [`src/05_build_dataset_from_selected.py`](file:///c:/Users/USER/Desktop/dump-truck-tarp/src/05_build_dataset_from_selected.py) | 將人工挑選的精華圖轉為 8:2 切分新資料集。 |
| `src/app_demo.py` | [`src/app_demo.py`](file:///c:/Users/USER/Desktop/dump-truck-tarp/src/app_demo.py) | 權重載入優先級調整為 `weights/best_tarp_v1.pt` -> `runs/tarp_training/weights/best.pt`。 |

---

### 4. Git 忽略清單擴充 ([.gitignore](file:///c:/Users/USER/Desktop/dump-truck-tarp/.gitignore))
為防止未來大體積影像池導致 Git 倉庫膨脹，新增了以下規則：
```gitignore
# Temporary & Curation Workspaces (大體積人工挑圖與測試池)
curation_pool_767/
dataset_curated/
runs/eval_desktop_overhead/
```

---

### 5. 說明文件更新 ([README.md](file:///c:/Users/USER/Desktop/dump-truck-tarp/README.md))
- 更新了整體目錄樹示意圖。
- 更新了所有示範指令（推論、訓練、Web DEMO 啟動指令）。

---

## 💻 常用指令對照表 (舊 ➔ 新)

| 動作 | 舊指令 | 新指令 |
| :--- | :--- | :--- |
| **批次/單張推論** | `python src/10_detect_tarp.py ...` | `.venv\Scripts\python.exe src/03_detect_tarp.py --input <路徑>` |
| **GPU 重訓** | `python src/09_train_tarp_yolo.py` | `.venv\Scripts\python.exe src/02_train_tarp_yolo.py` |
| **啟動 Web DEMO** | `python src/app_demo.py` | `.venv\Scripts\python.exe src/app_demo.py` |
| **去重提取 767 原圖** | `python src/11_extract_unique_767.py` | `.venv\Scripts\python.exe src/04_extract_unique_767.py` |
| **構建精煉資料集** | `python src/12_build_dataset_from_selected.py` | `.venv\Scripts\python.exe src/05_build_dataset_from_selected.py` |
| **匯入並構建 V2 資料集** | *(新增加功能)* | `.venv\Scripts\python.exe src/06_import_and_build_v2_dataset.py` |
| **訓練 V2 模型 (領域融合)** | *(新增加功能)* | `.venv\Scripts\python.exe src/07_train_tarp_v2.py` |

---

## 🌟 2026-10-25: V2 模型領域微調升級 (Domain Fine-Tuning)

### 1. 目標背景
使用者提供了 57 張真實台灣路況俯視角之精準標註資料（Covered 7 張、Uncovered 50 張）。為改善俯視監視器視角下對未蓋砂石車的辨識率，採用**方案 A（V1 基底 + 領域融合微調）**。

### 2. 核心變更
- **資料分層抽樣與融合**：
  - 建立 [`src/06_import_and_build_v2_dataset.py`](src/06_import_and_build_v2_dataset.py)。
  - 將 57 張圖片以分層抽樣（Stratified Split）精準劃分為 Train (46 張: 6 covered / 40 uncovered) 與 Val (11 張: 1 covered / 10 uncovered)。
  - 與原 2,674 張基礎資料融合，建立包含 2,731 張影像的 `dataset_v2/`。
- **RTX 4060 GPU 加速微調**：
  - 建立 [`src/07_train_tarp_v2.py`](src/07_train_tarp_v2.py)，以 `weights/best_tarp_v1.pt` 為基底進行自適應學習率微調。
  - 產出最佳權重 [`weights/best_tarp_v2.pt`](weights/best_tarp_v2.pt)。
- **推論與 Web 展示系統升級**：
  - 更新 [`src/03_detect_tarp.py`](src/03_detect_tarp.py) 與 [`src/app_demo.py`](src/app_demo.py)，優先自動調用 `best_tarp_v2.pt`。
- **成效對比**：
  - mAP@50 由 **92.2% 提升至 95.3% (+3.1%)**
  - Precision 由 85.0% 提升至 **87.9% (+2.9%)**
  - Recall 由 86.4% 提升至 **91.0% (+4.6%)**
  - 在 57 張真實俯視路況中，檢出率達 98.2%（56/57 正確檢出）。

---

## 🌟 2026-10-26: 架構全面收斂重構 (單一主線 + 歷程完整封存)

### 1. 重構目標
專案歷經三階段演進後，根目錄出現 `dataset_tarp` 與 `dataset_v2` 雙資料集、腳本 01~07 與多個 runs 混雜。為達到**「單純展現最後結果、中間過程全部保留」**之原則，將專案重構為**單一主線專案**，並將所有演進歷程歸檔至 `archive/`。

### 2. 結構變更對照
- **資料集統一**：`dataset_v2` 升格為唯一標準資料集 [`dataset/`](dataset/)（2,731 張影像），包含 `raw_overhead_57/` 持久備份；舊 `dataset_tarp/` 安全移入 [`archive/03_v1_dataset_backup/`](archive/03_v1_dataset_backup/)。
- **權重庫聚焦**：最新 V2 模型設為主權重 [`weights/best.pt`](weights/best.pt)；歷史版本移至 [`weights/archive/`](weights/archive/)。
- **主線腳本收斂**：
  - 訓練主程式：[`src/train.py`](src/train.py)（取代 02, 07）
  - 推論主程式：[`src/detect.py`](src/detect.py)（取代 03）
  - 互動展示：[`src/app_demo.py`](src/app_demo.py)（更新對應路徑）
  - 篩選工具：[`src/tools/`](src/tools/)
  - 階段腳本：全數歸檔於 [`archive/history_scripts/`](archive/history_scripts/)
- **成果目錄分層**：`runs/` 整理為 `train/`（最新訓練指標）、`eval_test/`、`eval_overhead/` 與 `archive/`。
- **歷史隔離歸檔**：`legacy_truck_presence/` 移入 [`archive/01_legacy_truck_presence/`](archive/01_legacy_truck_presence/)；`curation_pool_767/` 移入 [`archive/02_curation_pool_767/`](archive/02_curation_pool_767/)。


