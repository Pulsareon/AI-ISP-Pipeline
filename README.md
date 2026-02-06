# AI-ISP Pipeline

An end-to-end Image Signal Processing (ISP) pipeline designed for AI-based computational photography.

## 🚀 Features

- **Frame Selection**: Laplacian Variance based sharpness scoring.
- **Alignment**: ECC (Enhanced Correlation Coefficient) image alignment.
- **Demosaicing**: Standard Bayer to RGB conversion.
- **Denoising**: NLM (Non-local Means) with interface for Deep Learning models (DnCNN/UNet).
- **Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) and Texture Sharpening.

## 📂 Project Structure

```
AI-ISP/
├── pipeline.py       # Core ISP logic (Python Prototype)
├── pipeline_lite.py  # High-Performance Tiling Logic
├── simulate_raw.py   # Physical Sensor Simulator
├── ncnn_src/         # C++ Implementation (ncnn + OpenCV)
├── scripts/          # Deep Learning Workflow (PyTorch -> ONNX)
│   ├── model.py      # TinyISPNet Definition
│   ├── export.py     # ONNX Exporter
│   └── inference.py  # ONNX Runtime Inference Engine
└── requirements.txt
```

## 🛠️ Usage

### 1. Python Simulation
```bash
pip install opencv-python numpy
python pipeline.py
```

### 2. Deep Learning Demo (PyTorch -> ONNX)
We provide a complete workflow to train, export, and run a TinyISPNet.

**Export to ONNX:**
```bash
python scripts/export.py
```

**Run Inference:**
```bash
python scripts/inference.py
```
This runs the ISP pipeline on `input_bayer.png` (auto-generated) and saves `output.png`.

## 📜 License
MIT
