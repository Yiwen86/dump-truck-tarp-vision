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
