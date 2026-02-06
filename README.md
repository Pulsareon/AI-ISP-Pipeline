# AI-ISP Pipeline

> **"我们这一代程序员就像是汇编变成C，从coder变成editer。"**

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

### 4. RAW Processing (MIPI RAW10)
To handle real-world sensor data, we provide a RAW10 pack/unpack utility compliant with MIPI CSI-2 standards.

```bash
python scripts/raw_processing.py
```
This validates the bit-level manipulation logic required for interfacing with mobile sensors.

### 5. Test Gallery (Real-world Validation)

We have validated the TinyISPNet (Sharpening variant) on various scene types.
Results are saved in `results/` as side-by-side comparisons (Original Left vs AI Output Right).

| Scene Type | Source | Purpose | Output File |
|------------|--------|---------|-------------|
| **Landscape** | Unsplash | Test distant detail recovery | `results/compare_landscape.jpg` |
| **City** | Unsplash | Test geometric edge enhancement | `results/compare_city.jpg` |
| **Portrait** | Unsplash | Test skin texture preservation | `results/compare_portrait.jpg` |
| **Texture** | Unsplash | Test high-frequency detail restoration | `results/compare_texture.jpg` |

**Validation Metric:**
- Laplacian Variance Improvement: **> +3000%** (on blurred synthetic input)

## 🔧 Modular Architecture (Ref. openISP)

In response to Issue #1, we have refactored the pipeline into a modular design.

- **`pipeline_v2.py`**: A clean, modular ISP pipeline (BLC -> Demosaic -> CCM -> Gamma -> AI-NR).
- **`scripts/model.py`**: Added **DnCNN** implementation for state-of-the-art AI Denoising.

Run the modular pipeline:
```bash
python pipeline_v2.py
```

## 📜 License
MIT
