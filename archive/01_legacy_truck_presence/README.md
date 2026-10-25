# 歷史封存：卡車存在性判定系統 (Legacy Truck Presence Detection)

本資料夾已完整封存先前的「俯視監視器路口卡車存在性判定 (有車 / 無車)」所有舊程式碼、舊資料集與檢測結果。

## 封存內容目錄結構

- `src/`：
  - `04_detect_truck.py`：方案二 (COCO YOLO + 車道虛擬線圈混合檢測)
  - `05_prepare_dataset.py`：舊 421 張路況影像標註與資料集切分
  - `06_train_yolo.py`：舊 YOLO 卡車偵測微調訓練腳本
  - `07_detect_custom_yolo.py`：舊卡車存在微調模型推論腳本
- `data/`：
  - `raw_images/`：舊 421 張監視器連拍原始影像
- `dataset/`：
  - 舊 421 張影像組成的卡車存在 YOLO 標註資料集
- `runs/`：
  - `truck_detection/`：舊 421 張影像雙重檢測結果圖與 CSV
  - `truck_custom_yolo/` & `train/`：舊微調訓練日誌與權重
  - `dataset_preview/`：舊標註預覽
  - `test_custom/`：舊單張測試輸出
  - `reference_background.jpg`：舊道路背景參考基準圖
