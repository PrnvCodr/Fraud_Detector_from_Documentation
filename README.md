# 🔍 DocFraudDetector — AI-Powered Document Fraud & Tamper Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)

**End-to-end document fraud detection system using deep learning + classical computer vision forensics.**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Docs](#-api) · [Training](#-training) · [Demo](#-demo)

</div>

---

## 🎯 Features

| Stage | Technique | Description |
|-------|-----------|-------------|
| 📄 **Document Detection** | YOLOv8 + OpenCV Contours | Localize documents in arbitrary backgrounds with robust fallback |
| 📐 **Perspective Rectification** | OpenCV `warpPerspective` | Correct rotation, tilt, and perspective distortion |
| 🔬 **Tamper Detection** | EfficientNet-B0 + ELA + Noise Analysis | Multi-technique fraud detection with weighted scoring |
| 🔤 **OCR Extraction** | EasyOCR + Regex Field Parsing | Extract and structure text (Name, DOB, ID Number, etc.) |
| 🧪 **Synthetic Data** | Custom Generator (5 tamper types) | Generate training data with realistic tampering patterns |
| 🌐 **REST API** | FastAPI + Swagger | Production-ready API endpoint for document analysis |
| 💻 **Web Demo** | Streamlit | Interactive step-by-step visualization dashboard |

### Tamper Detection Techniques

- **Error Level Analysis (ELA)** — Detects JPEG re-compression artifacts from spliced regions
- **Noise Consistency Analysis** — Identifies blocks with inconsistent noise patterns
- **Edge Density Analysis** — Detects unnatural boundaries from copy-paste operations
- **CNN Classification** — EfficientNet-B0 trained on synthetic genuine/tampered pairs

### Synthetic Tamper Types

| Type | Description |
|------|-------------|
| `text_replacement` | Whiteout + re-typed text with slight color mismatch |
| `font_mismatch` | Inconsistent typography injected into document |
| `copy_paste` | Region cloned to a different location |
| `blur_injection` | Selective Gaussian blur to hide original content |
| `noise_injection` | Localized noise added to mask modifications |

---

## 🏗️ Architecture

```
Input Image
    │
    ▼
┌────────────────────────────────┐
│   1. Document Detection        │  YOLOv8-nano / OpenCV Contour Fallback
│   • Edge detection (Canny)     │
│   • Contour approximation      │
│   • Corner ordering            │
└──────────┬─────────────────────┘
           │ Cropped ROI + Corners
           ▼
┌────────────────────────────────┐
│   2. Perspective Rectification │  OpenCV getPerspectiveTransform
│   • Auto corner detection      │  + warpPerspective (INTER_CUBIC)
│   • Adaptive dimensions        │
│   • CLAHE enhancement          │
└──────────┬─────────────────────┘
           │ Rectified Document
           ▼
┌────────────────────────────────┐
│   3. Tamper Detection          │  Weighted Multi-Technique Analysis
│   ┌─────────────┐             │
│   │ ELA (0.40)  │─── Score ──▶│
│   ├─────────────┤             │  Weighted
│   │ Noise(0.35) │─── Score ──▶│──▶ Tamper Probability
│   ├─────────────┤             │
│   │ Edge (0.25) │─── Score ──▶│
│   └─────────────┘             │
│   + CNN (0.50) if trained     │
└──────────┬─────────────────────┘
           │ Verdict + Heatmap
           ▼
┌────────────────────────────────┐
│   4. OCR Extraction            │  EasyOCR (English + Hindi)
│   • Preprocessing (CLAHE)      │
│   • Adaptive binarization      │
│   • Regex field parsing        │
│   • Structured JSON output     │
└──────────┬─────────────────────┘
           │
           ▼
    ┌──────────────┐
    │  JSON Report  │  Verdict, fields, confidence, timing
    │  + Heatmap    │  + ELA/noise visualizations
    └──────────────┘
```

---

## 🚀 Quick Start

### 1. Installation

```bash
cd DocFraudDetector
pip install -r requirements.txt
```

### 2. Generate Sample Images

```bash
python data/synthetic_generator.py --samples-only
```

### 3. Run the Pipeline

```bash
# Analyze a single image
python src/pipeline.py data/sample_images/test_doc.png

# Results saved to outputs/
```

### 4. Generate Training Data + Train Model

```bash
# Generate 200 genuine + 200 tampered synthetic documents
python data/synthetic_generator.py

# Train the EfficientNet-B0 tamper classifier
python training/train_tamper.py --epochs 20 --batch-size 16
```

### 5. Launch Web Demo

```bash
streamlit run demo/app.py
```

### 6. Start API Server

```bash
python api/server.py
# Open http://localhost:8000/docs for Swagger UI
```

---

## 🌐 API

### `POST /analyze`

Upload a document image for full analysis.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@document.jpg" \
  -F "include_images=true"
```

**Response:**
```json
{
  "status": "success",
  "summary": {
    "is_tampered": false,
    "tamper_probability": 0.23,
    "verdict": "✅ GENUINE",
    "total_time_ms": 342.5
  },
  "stages": {
    "detection": { "confidence": 0.87, "method": "contour" },
    "rectification": { "output_size": {"width": 600, "height": 400} },
    "tamper_detection": { "tamper_probability": 0.23 },
    "ocr": { "word_count": 15, "structured_fields": {...} }
  },
  "images": {
    "detection": "<base64>",
    "rectified": "<base64>",
    "ela": "<base64>",
    "heatmap": "<base64>"
  }
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

---

## 🧠 Training

### Generate Synthetic Training Data

```bash
python data/synthetic_generator.py \
  --num-genuine 500 \
  --num-tampered 500
```

### Train Tamper Detection Model

```bash
python training/train_tamper.py \
  --epochs 30 \
  --batch-size 16 \
  --lr 0.0001 \
  --early-stopping 7
```

Training produces:
- Model checkpoint: `models/tamper_efficientnet_b0.pth`
- Training history: `models/training_history.json`
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC

---

## 📁 Project Structure

```
DocFraudDetector/
├── config.py                        # Central configuration
├── requirements.txt                 # Dependencies
├── README.md                        # This file
│
├── src/                             # Core pipeline modules
│   ├── __init__.py
│   ├── detector.py                  # Document detection (YOLOv8 + contours)
│   ├── rectifier.py                 # Perspective correction (OpenCV)
│   ├── tamper_detector.py           # Multi-technique tamper detection
│   ├── ocr_engine.py                # OCR text extraction (EasyOCR)
│   ├── pipeline.py                  # End-to-end orchestrator
│   └── utils.py                     # Image utilities & visualization
│
├── data/
│   ├── synthetic_generator.py       # Synthetic document generator
│   └── sample_images/               # Test images
│
├── training/
│   └── train_tamper.py              # Model training script
│
├── api/
│   └── server.py                    # FastAPI REST API
│
├── demo/
│   └── app.py                       # Streamlit web demo
│
├── models/                          # Saved model checkpoints
└── outputs/                         # Analysis results
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **CV / Image Processing** | OpenCV, scikit-image, Pillow, Albumentations |
| **Deep Learning** | PyTorch, timm (EfficientNet-B0), torchvision |
| **Object Detection** | Ultralytics YOLOv8 |
| **OCR** | EasyOCR |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Web Demo** | Streamlit |
| **ML Ops** | scikit-learn, tqdm, matplotlib |

---

## 📊 Resume Bullet Point

> Built an end-to-end document fraud detection system using YOLOv8 + EfficientNet-B0 + EasyOCR; implemented Error Level Analysis, noise consistency, and edge density forensics for multi-technique tamper scoring; generated 1000+ synthetic tampered documents with 5 tamper types; deployed as FastAPI REST API + Streamlit demo

---

## 👤 Author

**Pranav Kashyap**  
IIIT Dharwad  

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
