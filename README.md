# AI-ISP Pipeline (Performance Edition)

## 🎯 Hardware Targets
- **Platform**: Qualcomm Snapdragon 8 Gen 2 / 8 Gen 3
- **Resolution**: 4K (3840 x 2160)
- **Latency**: < 800ms (End-to-End)
- **Memory**: < 800MB Peak

## ⚡ Optimization Strategies

### 1. Tiling Architecture (分块处理)
To keep memory usage under 800MB, we process the 4K image in **512x512 tiles**.
- **Memory Footprint**: ~10MB per tile (FP16) vs ~200MB full frame.
- **L2 Cache Friendly**: Smaller tensors fit in NPU cache.

### 2. INT8 Quantization (量化)
Models must be quantized to INT8 for Snapdragon Hexagon NPU.
- **Toolchain**: SNPE (Snapdragon Neural Processing Engine) / QNN SDK.
- **Speedup**: 4x vs FP32.

### 3. Zero-Copy Pipeline
- Use `AHardwareBuffer` on Android to share memory between Camera, GPU (OpenCL), and NPU without CPU copying.

## 🧪 Benchmark (Python Simulation)

Run `pipeline_lite.py` to simulate the logic flow and check CPU overhead.

```bash
python pipeline_lite.py
```

*Note: Real performance requires C++ implementation on Android.*
