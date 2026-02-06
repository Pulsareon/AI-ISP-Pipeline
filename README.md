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
├── pipeline.py       # Core ISP logic
├── models/           # (Placeholder) PyTorch/ONNX models
├── raw_data/         # Input RAW images
└── output/           # Processed results
```

## 🛠️ Usage

```bash
pip install opencv-python numpy
python pipeline.py
```

## 🤖 AI Integration

To plug in your AI model (e.g., for Denoising):

1. Modify `denoise()` in `pipeline.py`.
2. Load your model: `model = torch.load('denoiser.pth')`.
3. Inference: `clean_tensor = model(noisy_tensor)`.

## 📜 License
MIT
