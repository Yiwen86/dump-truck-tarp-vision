#!/usr/bin/env python3
"""
腳本: src/app_demo.py
=============================================================================
功能:
卡車車斗帆布覆蓋合規性即時 Web 互動 DEMO 系統
- 支援拖曳 / 點選上傳任意圖片
- 支援一鍵快速載入內建合規/違規範例圖片
- 可動態調節信心度門檻 (Confidence Threshold)
- 視覺化顯示綠框 (Covered) / 紅框 (Uncovered) 與判決橫幅
- 顯示 GPU 推論延遲 (ms) 與結構化判定數據
=============================================================================
"""

import os
import sys
import time
import base64
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

# 模型權重路徑
WEIGHTS_CANDIDATES = [
    Path("weights/best_tarp_v2.pt").resolve(),
    Path("weights/best_tarp_v1.pt").resolve(),
    Path("runs/tarp_v2_training/weights/best.pt").resolve(),
    Path("runs/tarp_training/weights/best.pt").resolve(),
]
WEIGHTS_PATH = next((p for p in WEIGHTS_CANDIDATES if p.exists()), WEIGHTS_CANDIDATES[0])

print(f"[DEMO] Loading model from: {WEIGHTS_PATH}")
model = YOLO(str(WEIGHTS_PATH))

# 測試用範例圖片路徑
SAMPLE_FILES = {
    "covered": Path("dataset_tarp/test/images/17-OT_jpg.rf.841938541284d9b4a89e361079660515.jpg").resolve(),
    "uncovered": Path("dataset_tarp/test/images/05-NT_jpg.rf.77c7e59f483774471f4c7c16f49a8ab5.jpg").resolve(),
    "real_uncovered": Path("dataset_v2/raw_downloaded_57/-1-_png.rf.d5ee89bc015442bdd7019b0f1da86581.jpg").resolve(),
    "real_covered": Path("dataset_v2/raw_downloaded_57/frame_000010_jpg.rf.a065d129db9d8f38fc647e8ad4640d90.jpg").resolve(),
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>砂石車/卡車車斗帆布覆蓋合規性即時判定系統</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0a0e17;
            --card-bg: rgba(18, 26, 43, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.25);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(16, 185, 129, 0.08) 0%, transparent 45%);
            background-attachment: fixed;
        }

        header {
            padding: 24px 32px;
            border-bottom: 1px solid var(--border-color);
            background: rgba(10, 14, 23, 0.85);
            backdrop-filter: blur(16px);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .header-title h1 {
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .model-badge {
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.4);
            color: #60a5fa;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
        }

        .gpu-badge {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
        }

        main {
            flex: 1;
            max-width: 1380px;
            width: 100%;
            margin: 0 auto;
            padding: 32px;
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 28px;
        }

        @media (max-width: 1024px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .panel h2 {
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #e5e7eb;
        }

        /* 檔案上傳區域 */
        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 36px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
            background: rgba(255, 255, 255, 0.02);
            position: relative;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent-blue);
            background: rgba(59, 130, 246, 0.05);
            box-shadow: 0 0 24px var(--accent-glow);
        }

        .dropzone svg {
            width: 44px;
            height: 44px;
            fill: none;
            stroke: var(--text-muted);
            margin-bottom: 12px;
            transition: stroke 0.25s ease;
        }

        .dropzone:hover svg {
            stroke: var(--accent-blue);
        }

        .dropzone-text {
            font-size: 0.95rem;
            color: #d1d5db;
            margin-bottom: 6px;
        }

        .dropzone-hint {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        input[type="file"] {
            display: none;
        }

        /* 範例測試按鈕 */
        .sample-buttons {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .sample-btn {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 12px 14px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.88rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.2s ease;
        }

        .sample-btn:hover {
            background: rgba(255, 255, 255, 0.09);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }

        .sample-tag-covered {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .sample-tag-uncovered {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* 參數滑桿 */
        .setting-item {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .setting-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .setting-val {
            font-family: 'JetBrains Mono', monospace;
            color: #60a5fa;
            font-weight: 600;
        }

        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
            accent-color: var(--accent-blue);
        }

        .btn-predict {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
            padding: 14px 20px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }

        .btn-predict:hover {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
            transform: translateY(-1px);
        }

        .btn-predict:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* 右側結果展示 */
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .status-banner {
            padding: 16px 20px;
            border-radius: 12px;
            font-size: 1.15rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
            transition: all 0.3s ease;
        }

        .status-compliant {
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.25), rgba(16, 185, 129, 0.1));
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399;
        }

        .status-violation {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.25), rgba(239, 68, 68, 0.1));
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
            animation: pulse-border 2s infinite;
        }

        .status-idle {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
        }

        @keyframes pulse-border {
            0%, 100% { border-color: rgba(239, 68, 68, 0.4); }
            50% { border-color: rgba(239, 68, 68, 0.85); box-shadow: 0 0 16px rgba(239, 68, 68, 0.3); }
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px 18px;
        }

        .metric-title {
            font-size: 0.78rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        .metric-val {
            font-size: 1.4rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: #f3f4f6;
        }

        /* 預覽影像容器 */
        .image-viewport {
            background: #000;
            border-radius: 14px;
            border: 1px solid var(--border-color);
            min-height: 460px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }

        .image-viewport img {
            max-width: 100%;
            max-height: 600px;
            object-fit: contain;
            border-radius: 8px;
        }

        .placeholder-text {
            color: var(--text-muted);
            font-size: 0.95rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }

        /* 載入中動畫 */
        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top: 3px solid var(--accent-blue);
            width: 24px;
            height: 24px;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>🚛 砂石車/卡車車斗帆布覆蓋合規性即時判定系統</h1>
            <span class="model-badge">YOLOv8 Tarp V2 (mAP50=95.3%)</span>
            <span class="gpu-badge">RTX 4060 GPU 加速</span>
        </div>
    </header>

    <main>
        <!-- 左側操作面板 -->
        <div class="panel">
            <h2>📷 影像輸入</h2>
            
            <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                <svg viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
                <div class="dropzone-text">點擊選取 或 拖曳卡車照片至此</div>
                <div class="dropzone-hint">支援 JPG, PNG, WEBP 高畫質格式</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>

            <h2>⚡ 快速範例測試</h2>
            <div class="sample-buttons">
                <button class="sample-btn" onclick="loadSample('real_uncovered')">
                    <span>台灣路況 A：路口俯視裸露違規</span>
                    <span class="sample-tag-uncovered">真實違規</span>
                </button>
                <button class="sample-btn" onclick="loadSample('real_covered')">
                    <span>台灣路況 B：路口俯視帆布有蓋</span>
                    <span class="sample-tag-covered">真實合格</span>
                </button>
                <button class="sample-btn" onclick="loadSample('uncovered')">
                    <span>基準範例 C：車斗砂石裸露違規</span>
                    <span class="sample-tag-uncovered">Uncovered</span>
                </button>
                <button class="sample-btn" onclick="loadSample('covered')">
                    <span>基準範例 D：帆布完整覆蓋卡車</span>
                    <span class="sample-tag-covered">Covered</span>
                </button>
            </div>

            <h2>⚙️ 推論設定</h2>
            <div class="setting-item">
                <div class="setting-header">
                    <span>信心度門檻 (Confidence)</span>
                    <span class="setting-val" id="confVal">0.30</span>
                </div>
                <input type="range" id="confSlider" min="0.10" max="0.90" step="0.05" value="0.30" oninput="document.getElementById('confVal').textContent = this.value">
            </div>

            <button class="btn-predict" id="predictBtn" onclick="runInference()" disabled>
                <span>🚀 執行神經網絡合規判定</span>
            </button>
        </div>

        <!-- 右側結果面板 -->
        <div class="panel results-container">
            <div class="status-banner status-idle" id="statusBanner">
                <span id="statusIcon">⚪</span>
                <span id="statusText">等待載入影像並執行檢測</span>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">推論耗時 (GPU)</div>
                    <div class="metric-val" id="latencyVal">-- ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">檢出車斗目標</div>
                    <div class="metric-val" id="targetsVal">-- 個</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">最高信心度</div>
                    <div class="metric-val" id="confScoreVal">--%</div>
                </div>
            </div>

            <div class="image-viewport" id="viewport">
                <div class="placeholder-text" id="placeholder">
                    <svg style="width: 48px; height: 48px; stroke: #4b5563; fill: none;" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
                    <span>請先自左側上傳圖片或點選快速範例</span>
                </div>
                <img id="resultImg" style="display: none;" alt="檢測標註結果">
            </div>
        </div>
    </main>

    <script>
        let currentImageBlob = null;

        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const predictBtn = document.getElementById('predictBtn');
        const resultImg = document.getElementById('resultImg');
        const placeholder = document.getElementById('placeholder');

        // Drag & Drop
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            currentImageBlob = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                resultImg.src = e.target.result;
                resultImg.style.display = 'block';
                placeholder.style.display = 'none';
                predictBtn.disabled = false;
                resetStatus();
            };
            reader.readAsDataURL(file);
        }

        async function loadSample(sampleType) {
            placeholder.innerHTML = '<div class="spinner"></div><span>載入範例中...</span>';
            placeholder.style.display = 'flex';
            resultImg.style.display = 'none';

            try {
                const response = await fetch(`/sample/${sampleType}`);
                const blob = await response.blob();
                currentImageBlob = blob;
                const url = URL.createObjectURL(blob);
                resultImg.src = url;
                resultImg.style.display = 'block';
                placeholder.style.display = 'none';
                predictBtn.disabled = false;
                resetStatus();
                // 自動執行一次辨識
                runInference();
            } catch (err) {
                alert('載入範例失敗: ' + err);
            }
        }

        function resetStatus() {
            const banner = document.getElementById('statusBanner');
            banner.className = 'status-banner status-idle';
            document.getElementById('statusIcon').textContent = '⚪';
            document.getElementById('statusText').textContent = '已載入圖片，請點擊「執行神經網絡合規判定」';
            document.getElementById('latencyVal').textContent = '-- ms';
            document.getElementById('targetsVal').textContent = '-- 個';
            document.getElementById('confScoreVal').textContent = '--%';
        }

        async function runInference() {
            if (!currentImageBlob) return;

            predictBtn.disabled = true;
            predictBtn.innerHTML = '<div class="spinner"></div><span>GPU 推論中...</span>';

            const formData = new FormData();
            formData.append('image', currentImageBlob);
            formData.append('conf', document.getElementById('confSlider').value);

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (data.error) {
                    alert('推論出錯: ' + data.error);
                    return;
                }

                // 顯示標註後影像
                resultImg.src = 'data:image/jpeg;base64,' + data.image_base64;
                resultImg.style.display = 'block';
                placeholder.style.display = 'none';

                // 更新指標
                document.getElementById('latencyVal').textContent = data.latency_ms + ' ms';
                document.getElementById('targetsVal').textContent = data.targets_count + ' 個';
                document.getElementById('confScoreVal').textContent = (data.max_conf * 100).toFixed(1) + '%';

                // 更新狀態橫幅
                const banner = document.getElementById('statusBanner');
                if (data.status === 'VIOLATION') {
                    banner.className = 'status-banner status-violation';
                    document.getElementById('statusIcon').textContent = '🚨';
                    document.getElementById('statusText').textContent = '[VIOLATION] 違規！偵測到未覆蓋車斗 (Uncovered Bed)';
                } else if (data.status === 'COMPLIANT') {
                    banner.className = 'status-banner status-compliant';
                    document.getElementById('statusIcon').textContent = '✅';
                    document.getElementById('statusText').textContent = '[COMPLIANT] 帆布合規覆蓋 (Tarp Properly Covered)';
                } else {
                    banner.className = 'status-banner status-idle';
                    document.getElementById('statusIcon').textContent = '⚪';
                    document.getElementById('statusText').textContent = '[IDLE] 未偵測到車斗目標';
                }

            } catch (err) {
                alert('伺服器連線失敗: ' + err);
            } finally {
                predictBtn.disabled = false;
                predictBtn.innerHTML = '<span>🚀 執行神經網絡合規判定</span>';
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/sample/<sample_type>")
def get_sample(sample_type):
    file_path = SAMPLE_FILES.get(sample_type)
    if not file_path or not file_path.exists():
        return "Sample file not found", 404
    with open(file_path, "rb") as f:
        data = f.read()
    return data, 200, {"Content-Type": "image/jpeg"}

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    conf = float(request.form.get("conf", 0.30))

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400

    t_start = time.time()
    results = model.predict(source=img, conf=conf, device=0, verbose=False)
    t_latency = (time.time() - t_start) * 1000.0

    res = results[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
    clses = res.boxes.cls.cpu().numpy() if res.boxes is not None else []
    confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []

    annotated = img.copy()
    h, w = annotated.shape[:2]

    has_uncovered = False
    has_covered = False
    max_conf = float(max(confs)) if len(confs) > 0 else 0.0

    for box, cls_id, conf_val in zip(boxes, clses, confs):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)
        
        if cls_id == 0:
            color = (0, 200, 0)
            label = f"Covered {conf_val * 100:.1f}%"
            has_covered = True
        else:
            color = (0, 0, 255)
            label = f"UNCOVERED {conf_val * 100:.1f}%"
            has_uncovered = True

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if has_uncovered:
        status = "VIOLATION"
        banner_text = "[VIOLATION] Uncovered Bed Detected (Non-compliant)"
        banner_color = (0, 0, 220)
    elif has_covered:
        status = "COMPLIANT"
        banner_text = "[COMPLIANT] Tarp Properly Covered"
        banner_color = (0, 160, 0)
    else:
        status = "IDLE"
        banner_text = "[IDLE] No Dump Bed Detected"
        banner_color = (80, 80, 80)

    cv2.rectangle(annotated, (0, 0), (w, 36), banner_color, -1)
    cv2.putText(annotated, banner_text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    _, encoded_img = cv2.imencode(".jpg", annotated)
    img_b64 = base64.b64encode(encoded_img).decode("utf-8")

    return jsonify({
        "status": status,
        "latency_ms": round(t_latency, 1),
        "targets_count": len(boxes),
        "max_conf": max_conf,
        "image_base64": img_b64,
    })

if __name__ == "__main__":
    print("=" * 68)
    print(" 🚀 Starting Web DEMO Server on http://127.0.0.1:5000")
    print("=" * 68)
    app.run(host="127.0.0.1", port=5000, debug=False)
