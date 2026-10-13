# Depth Anything 3 C++ Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

C++ inference engine for **[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)** monocular depth estimation, with **multiple inference backends** (ONNX Runtime and TensorRT) and OpenCV for image I/O and visualization.

🎬 **Demo video**: [monocular depth on street footage — TensorRT 11 on RTX 3060](https://www.youtube.com/watch?v=7wUFLR9P0p4) (side-by-side RGB vs predicted depth, 195 ms/frame; replication: `docs/video-inference.md`)

Other useful upstream links:

- **Project page**: <https://depth-anything-3.github.io/> (abilities, demos, benchmark)
- **Upstream repo**: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Citation**: Depth Anything 3, arXiv:2511.10647 (<https://arxiv.org/abs/2511.10647>)
- **Hugging Face collection**: [depth-anything/depth-anything-3](https://huggingface.co/collections/depth-anything/depth-anything-3) — 11 items: `depth-anything-3-nested-giant-large-1.1`, `depth-anything-3-giant-1.1`, `depth-anything-3-large-1.1`, `depth-anything-3-base`, `depth-anything-3-small`, `depth-anything-3-metric-large`, `depth-anything-3-mono-large`, `depth-anything-3-nested-giant-large`, `depth-anything-3-giant`, `depth-anything-3-large`, plus the online demo space and the `DA3-BENCH` benchmark
- **Camera depth models**: [depth-anything/camera-depth-models](https://huggingface.co/collections/depth-anything/camera-depth-models) — 7 items: `d405`, `d435`, `l515`, `kinect`, `zed2i-neural`, `zed2i-quality`, `base`


---

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Model Setup](#model-setup)
- [Parity With the Python Reference](#parity-with-the-python-reference)
- [Scope: Multi-View Mode (Deferred)](#scope-multi-view-mode-deferred)
- [Installation](#installation)
- [Building](#building)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Testing](#testing)
- [Docker](#docker)
- [Code Quality Tools](#code-quality-tools)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Monocular depth estimation** from a single RGB image
- **Multiple inference backends** selectable at compile time:
  - ONNX Runtime (CPU/GPU, cross-platform, auto-downloaded)
  - TensorRT (NVIDIA GPU, auto-downloaded)
- **ImageNet normalization** matching the Python reference implementation
- **Aspect-ratio-preserving resize** with DINOv2 patch-size divisibility (multiples of 14)
- **Depth colormap visualization** using OpenCV's INFERNO colormap
- **Optional confidence map** output
- **Unit tests** using Google Test (no model required)
- **Google Benchmark** support for preprocessing benchmarks

---

## Dependencies

### Required (All Backends)
- **C++20 Compiler**: Clang 15+ or GCC 12+
- **CMake**: Version 3.12 or higher
- **OpenCV**: Version 4.x (`core`, `imgproc`, `imgcodecs`, `highgui`)
- **Google Test**: Version 1.12.1 (automatically fetched during build)
- **Ninja**: Optional but recommended (`sudo apt-get install ninja-build`)

### Backend-Specific Dependencies

#### ONNX Runtime Backend (Default)
- **ONNX Runtime**: Version 1.21.0 (automatically downloaded during CMake configure)
- **Platform**: Linux, Windows, macOS
- **Acceleration**: CPU and GPU

#### TensorRT Backend (Optional)
- **TensorRT**: Version 11.2.1.2 (same pin as rf-detr-cpp-inference) + CUDA 13.3 toolkit (automatically downloaded during CMake configure; strongly typed — engine precision follows the ONNX graph)
- **CUDA Toolkit**: Must be installed manually
- **Platform**: Linux with NVIDIA GPU

---

## Model Setup

1. **Install the Depth Anything 3 Python package** in a virtual environment:
   ```bash
   pip install xformers "torch>=2" torchvision
   pip install -e /path/to/Depth-Anything-3
   ```

2. **Export the model to ONNX** using the provided script:
   ```bash
   # Export DA3-Large (recommended for a good speed/quality trade-off)
   python export_onnx.py \
       --model-name da3-large \
       --output model.onnx \
       --process-res 504

   # Export DA3Mono-Large (monocular-only variant, slightly faster)
   python export_onnx.py \
       --model-name da3mono-large \
       --output model_mono.onnx

   # Export from a local directory
   python export_onnx.py \
       --model-dir /path/to/local/model \
       --output model.onnx
   ```
   The script exports the model with **dynamic spatial axes** by default so it
   accepts images of any resolution (as long as H and W are divisible by 14).

3. **Supported model variants** for monocular depth:

   | Model | Params | License | Notes |
   |-------|--------|---------|-------|
   | `DA3-Small` | ~90M | Apache-2.0 | Fastest |
   | `DA3-Base` | ~190M | Apache-2.0 | Balanced |
   | `DA3-Large` | 350M | CC BY-NC 4.0 | Good quality — prefer the retrained `-1.1` checkpoint (`da3-large-1.1`) |
   | `DA3-Giant` | 1.15B | CC BY-NC 4.0 | Best quality — prefer the retrained `-1.1` checkpoint (`da3-giant-1.1`) |
   | `DA3-Nested-Giant-Large` | ~1.25B | CC BY-NC 4.0 | Nested multi-view variant — prefer `da3-nested-giant-large-1.1`; already outputs metric depth (see metric note below) |
   | `DA3Mono-Large` | 350M | Apache-2.0 | Monocular-optimized |
   | `DA3Metric-Large` | 350M | Apache-2.0 | Metric depth (real-world scale) |

   > **License note**: the Small/Base/Mono/Metric and camera-base checkpoints are **Apache-2.0**;
   > the Large/Giant/Nested checkpoints are **CC BY-NC 4.0** (non-commercial). For Large, Giant and
   > Nested, prefer the retrained `-1.1` checkpoints for better quality.

4. **Camera-depth model variants** (trained per camera model, on
   [depth-anything/camera-depth-models](https://huggingface.co/collections/depth-anything/camera-depth-models)):

   | Model | Target device | Notes |
   |-------|--------------|-------|
   | `d405` | Intel RealSense D405 | Camera-specific depth |
   | `d435` | Intel RealSense D435 | Camera-specific depth |
   | `l515` | Intel RealSense L515 | Camera-specific depth |
   | `kinect` | Microsoft Kinect | Camera-specific depth |
   | `zed2i-neural` | Stereolabs ZED 2i | Neural (stacked) variant |
   | `zed2i-quality` | Stereolabs ZED 2i | Quality (stacked) variant |
   | `base` | Generic RGB camera | Camera-base model (Apache-2.0) |

---

## Installation

### Install Dependencies (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y cmake libopencv-dev

# Compiler — any C++20-capable version:
sudo apt-get install -y clang-18
# or: gcc (default on Ubuntu 22.04+ is GCC 11+)

# Optional (faster builds):
sudo apt-get install -y ninja-build

# Optional (linting):
sudo apt-get install -y clang-format-18 clang-tidy-18
```

---

## Building

### Backend Selection

**Important**: Only ONE backend can be enabled at a time.

### Build with ONNX Runtime (Default)

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
```

With explicit compiler:
```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18

cmake --build build --parallel
```

### Build with TensorRT Backend

> **Static shapes required**: the TensorRT backend sets no optimization
> profile, so the ONNX model must have static spatial dims. Export with
> `--fixed-size` for this path:
> ```bash
> python export_onnx.py --model-name da3-small --output model_fixed.onnx --fixed-size --process-res 504
> ```
> (Dynamic-axes exports are rejected at engine-build time with a clear error.)

```bash
cmake -S . -B build -G Ninja \
  -DUSE_ONNX_RUNTIME=OFF \
  -DUSE_TENSORRT=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `-DUSE_ONNX_RUNTIME=ON/OFF` | ON | Enable ONNX Runtime backend |
| `-DUSE_TENSORRT=ON/OFF` | OFF | Enable TensorRT backend |
| `-DCMAKE_BUILD_TYPE=Release/Debug` | — | Build configuration |
| `-DSANITIZERS=ON/OFF` | OFF | Enable AddressSanitizer + UBSan |
| `-DWERROR=ON/OFF` | OFF | Treat warnings as errors |
| `-DBENCHMARKS=ON/OFF` | OFF | Build Google Benchmark targets |

---

## Usage

### Single Image Inference

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg
```

The output depth colormap is saved as **`output_depth.jpg`** in the current directory.

### Native Task Selection (`--task`)

```bash
# Monocular relative depth (default — previous commands keep working)
./build/inference_app model.onnx image.jpg --task mono
# Metric depth in meters (DA3METRIC-LARGE; Nested is already meters: add --already-metric)
./build/inference_app model.onnx image.jpg --task metric --focal 500
# Multi-view: pass a directory (sorted *.jpg/*.jpeg/*.png → output_depth_%03d.jpg)
./build/inference_app model.onnx ./views/ --task multiview
# Pose: poses.txt (N×12 w2c) + intrinsics.txt (N×4 fx fy cx cy)
./build/inference_app model.onnx ./views/ --task pose [--use-ray-pose]
# Dense reconstruction → output_cloud.ply (ASCII, x y z r g b)
./build/inference_app model.onnx ./views/ --task reconstruction
# 3D Gaussians → output_gaussians.ply (depth-lifted placeholder; full head needs gsplat export)
./build/inference_app model.onnx ./views/ --task gaussian
```

| `--task` | Outputs | Best checkpoints |
|----------|---------|------------------|
| `mono` | `output_depth.jpg` | DA3-SMALL / BASE / LARGE / GIANT |
| `metric` | `output_depth.jpg` + `output_metric.npy` | DA3METRIC-LARGE |
| `multiview` | `output_depth_%03d.jpg` per frame | DA3-LARGE / GIANT |
| `pose` | `poses.txt` + `intrinsics.txt` | DA3-LARGE / GIANT (not METRIC) |
| `reconstruction` | `output_cloud.ply` | DA3-GIANT; NESTED-GIANT-LARGE for metric |
| `gaussian` | `output_gaussians.ply` (placeholder) | DA3-LARGE / GIANT + Gaussian head |

Notes: per-frame multi-view is a baseline — spatially consistent fusion needs an
any-view ONNX (`python export_onnx.py --task multiview|pose|… --use-ray-pose`).
Tasks whose head the loaded ONNX lacks degrade gracefully (identity poses,
depth-lifted cloud/gaussians) with a warning, never a silent wrong result.
License: Small/Base/Mono/Metric Apache-2.0; Large/Giant/Nested CC BY-NC 4.0 (prefer `-1.1`).

### With Confidence Map

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg --conf
```

Saves `output_depth.jpg` and `output_conf.jpg`.

### Custom Processing Resolution

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg --process-res 756
```

### Keep Depth at Model Resolution (no upscale)

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg --no-upscale
```

### Full Usage

```
Usage: ./build/inference_app <path_to_model> <path_to_image> [--conf] [--no-upscale] [--process-res N] [--task TASK] [--focal PX] [--input-dir DIR] [--output-dir DIR] [--already-metric] [--use-ray-pose]

Arguments:
  path_to_model   ONNX model (.onnx) or TensorRT engine (.engine / .trt)
  path_to_image   Input image (JPG, PNG, ...) or directory of images

Options:
  --conf          Also save the confidence map as output_conf.jpg
  --no-upscale    Keep depth map at model output resolution (no resize to original)
  --process-res N Resize the longest side to N before inference (default: 504)
  --task TASK     Native DA3 task: mono (default), metric, multiview, pose, reconstruction, gaussian
  --focal PX      Focal length in pixels for --task metric (default: 500)
  --input-dir DIR Directory of input images (overrides positional image)
  --output-dir DIR Directory for output files (default: current directory)
  --already-metric Depth already in meters (Nested); skip focal*net/300 scaling
  --use-ray-pose  Ray-based pose estimation for --task pose (needs pose-capable export)
```

---

## Configuration

The `Config` struct controls inference behaviour:

```cpp
Config config;
config.process_res = 504;     // Longest-side target resolution
config.patch_size  = 14;      // DINOv2 patch divisibility requirement
config.output_conf = false;   // Also extract confidence map
// ImageNet normalization (same as Python reference):
config.means = {0.485f, 0.456f, 0.406f};
config.stds  = {0.229f, 0.224f, 0.225f};
```

---

## Technical Details

### Preprocessing Pipeline

1. Load BGR image via OpenCV
2. Compute output spatial size:
   - Scale longest side to `process_res`
   - Round each dimension to nearest multiple of `patch_size` (14)
3. Resize to `(out_h, out_w)` with bilinear interpolation
4. Convert BGR → RGB
5. Scale to [0, 1] (divide by 255)
6. Arrange in CHW layout
7. Apply ImageNet normalization: `(x − mean) / std`

### Model Input / Output

| Tensor | Shape | Description |
|--------|-------|-------------|
| `image` (input) | `[1, 3, H, W]` | Preprocessed CHW float image |
| `depth` (output 0) | `[1, H, W]` | Depth map (relative or metric) |
| `conf` (output 1) | `[1, H, W]` | Confidence map (optional) |

> **Note**: The spatial dimensions `H` and `W` must be divisible by 14.
> With dynamic-axes export (default), the model accepts any compliant resolution.

### Postprocessing

The depth tensor is:
1. Reshaped from `[1, H, W]` to a `cv::Mat` of type `CV_32F`
2. Optionally upscaled back to the original image resolution with bilinear interpolation
3. Colorized using OpenCV's `COLORMAP_INFERNO` for visualization

### Metric Depth

For the metric models (`DA3Metric-Large`, camera-depth variants), the real-world
metric depth is obtained from the network output as:

```
metric_depth = focal * net_output / 300.
```

where `focal` is the focal length in pixels from the camera intrinsics. The
nested multi-view variant (`DA3-Nested-Giant-Large`) already outputs meters and
requires no conversion.

---

## Parity With the Python Reference

The C++/ONNX path is parity-checked against the upstream Python reference:

- **Python forward call** (MonocularWrapper):
  `model.model(imgs.unsqueeze(1), None, None, None, False, False, "saddle_balanced")`
- **ONNX graph**: input image `[1, 3, H, W]` → outputs depth `[1, H, W]` and
  optional confidence `[1, H, W]`; exported with **opset 17** and **dynamic axes**
  (any H, W divisible by 14)
- **C++ preprocessing** (`Config{process_res = 504, patch_size = 14}`): the input
  image's longest side is resized to `process_res`, `H` and `W` are rounded to be
  divisible by 14, and ImageNet normalization
  (`means = {0.485, 0.456, 0.406}`, `stds = {0.229, 0.224, 0.225}`) is applied.

---

## Scope: Multi-View Mode (Deferred)

The upstream DA3 package also supports **any-view (multi-view) reconstruction**,
which is out of scope for this C++ inference engine and **deferred to Phase 3**:

- Depth/confidence maps of shape `[N, H, W]`, extrinsics `[N, 3, 4]`, and
  intrinsics `[N, 3, 3]` for N input views
- Camera options such as `use_ray_pose` and `ref_view_strategy`
- Streaming inference for scenes that do not fit in `<12GB` of GPU memory
- 3D asset exports: `glb` / `npz` / `ply` / `3DGS video`

This repository currently covers the **monocular** depth-estimation path only.

---

## Testing

### Unit Tests

Unit tests use Google Test and run entirely without a model file:

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R UnitTests
```

Tests cover:
- `normalize_image` — channel-wise normalization
- `compute_output_size` — aspect-ratio-preserving resize with patch alignment
- `round_to_multiple` — patch divisibility rounding
- `preprocess_frame` — full CHW tensor pipeline
- `apply_depth_colormap` — visualization
- `DepthAnything3Inference` — constructor, preprocess, run+postprocess with mock backend

### Benchmarks

```bash
cmake -S . -B build -DBENCHMARKS=ON
cmake --build build --target benchmarks --parallel
./build/benchmarks
```

### Sanitizers

```bash
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DSANITIZERS=ON
cmake --build build-san --parallel
ctest --test-dir build-san --output-on-failure
```

---

## Docker

### ONNX Runtime (CPU)

```bash
docker build -t da3-onnx .
docker run -v $(pwd)/data:/data da3-onnx /data/model.onnx /data/image.jpg
```

---

## Code Quality Tools

| Tool | Purpose | How to run |
|------|---------|------------|
| `clang-format-18` | Code formatting | `find src tests -name '*.cpp' -o -name '*.hpp' \| xargs clang-format-18 -i` |
| `clang-tidy-18` | Static analysis | `find src -name '*.cpp' \| xargs clang-tidy-18 -p build` |
| `cppcheck` | Flow-based static analysis | `cppcheck --enable=all --std=c++20 -I src src/` |
| AddressSanitizer + UBSan | Runtime memory/UB detection | `-DSANITIZERS=ON` at configure time |

---

## Acknowledgements

- **Depth Anything 3** model: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
