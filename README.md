# Depth Anything 3 C++ Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

C++ inference engine for **[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)** monocular depth estimation. It builds with one compile-time backend — ONNX Runtime or TensorRT — and uses OpenCV for image I/O and visualization.

🎬 **Demo video**: [monocular depth on street footage — TensorRT 11 on RTX 3060](https://www.youtube.com/watch?v=7wUFLR9P0p4) (side-by-side RGB vs predicted depth, 195 ms/frame; replication: `docs/video-inference.md`)

Other useful upstream links:

- **Project page**: <https://depth-anything-3.github.io/> (abilities, demos, benchmark)
- **Upstream repo**: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Citation**: Depth Anything 3, arXiv:2511.10647 (<https://arxiv.org/abs/2511.10647>)
- **Hugging Face collection**: [depth-anything/depth-anything-3](https://huggingface.co/collections/depth-anything/depth-anything-3) — 12 items: 10 model checkpoints (`depth-anything-3-nested-giant-large-1.1`, `depth-anything-3-giant-1.1`, `depth-anything-3-large-1.1`, `depth-anything-3-base`, `depth-anything-3-small`, `depth-anything-3-metric-large`, `depth-anything-3-mono-large`, `depth-anything-3-nested-giant-large`, `depth-anything-3-giant`, `depth-anything-3-large`) plus the [online demo space](https://huggingface.co/spaces/depth-anything/depth-anything-3) and the [DA3-BENCH dataset](https://huggingface.co/datasets/depth-anything/DA3-BENCH)
- **Camera depth models**: [depth-anything/camera-depth-models](https://huggingface.co/collections/depth-anything/camera-depth-models) — 7 items: `d405`, `d435`, `l515`, `kinect`, `zed2i-neural`, `zed2i-quality`, `base`


---

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Model Setup](#model-setup)
- [Installation](#installation)
- [Building](#building)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Parity With the Python Reference](#parity-with-the-python-reference)
- [Any-View Support](#any-view-support)
- [Testing](#testing)
- [Docker](#docker)
- [Continuous Integration](#continuous-integration)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Monocular depth estimation** from a single RGB image
- **Backend choice at compile time** (one per binary):
  - ONNX Runtime (CPU/GPU, cross-platform, auto-downloaded)
  - TensorRT (NVIDIA GPU, auto-downloaded)
- **ImageNet normalization** matching the Python reference implementation
- **Aspect-ratio-preserving resize** with DINOv2 patch-size divisibility (multiples of 14)
- **Depth colormap visualization** using OpenCV's INFERNO colormap
- **Optional live capture**: Intel RealSense color plus aligned depth, captured
  in-process (no Python) when built with `-DUSE_REALSENSE=ON`
- **Optional confidence map** output
- **Unit tests** using Google Test (no downloaded model required)
- **Google Benchmark** support for preprocessing benchmarks

---

## Dependencies

### Required (All Backends)
- **C++20 Compiler**: Clang 15+ or GCC 12+
- **CMake**: Version 3.12 or higher
- **OpenCV**: Version 4.x (`core`, `imgproc`, `imgcodecs`, `highgui`, `videoio`)
- **Google Test**: Version 1.12.1 (automatically fetched during build)
- **Ninja**: Optional but recommended (`sudo apt-get install ninja-build`)

### Backend-Specific Dependencies

#### ONNX Runtime Backend (Default)
- **ONNX Runtime**: Version 1.21.0 (automatically downloaded during CMake configure)
- **Platform**: Linux, Windows, macOS
- **Acceleration**: CPU and GPU

#### TensorRT Backend (Optional)
- **TensorRT**: Version 11.2.1.2 (same pin as rf-detr-cpp-inference; downloaded during CMake configure; strongly typed — engine precision follows the ONNX graph)
- **CUDA Toolkit**: Version 13.3 must be installed; CMake locates it with `find_package(CUDAToolkit)`
- **Platform**: Linux with NVIDIA GPU

#### RealSense Capture (Optional)
- **librealsense2**: Version 2.59.1. CMake uses a system package when present, otherwise builds the pinned source automatically. Requires `libusb-1.0-0-dev` and `libudev-dev`.
- **Enabled by**: `-DUSE_REALSENSE=ON` (default `OFF`). Default builds, Docker, and CI need no RealSense dependency.
- **udev rules**: required once for non-root USB access on Linux.

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
   The script declares dynamic spatial axes by default, but the current traced exports bake the RoPE tables for the resolution used during export. Treat the exported graph as shape-locked: run it at the trace resolution, or export with `--fixed-size` and use that exact size. See [Native Task Selection](#native-task-selection---task) for the runtime guidance.

3. **Supported model variants** (monocular, plus the any-view Nested checkpoint):

   | Model | Params | License | Notes |
   |-------|--------|---------|-------|
   | `DA3-Small` | ~90M | Apache-2.0 | Fastest |
   | `DA3-Base` | ~190M | Apache-2.0 | Balanced |
   | `DA3-Large` | 350M | CC BY-NC 4.0 | Good quality — prefer the retrained `-1.1` checkpoint (`da3-large-1.1`) |
   | `DA3-Giant` | 1.15B | CC BY-NC 4.0 | Best quality — prefer the retrained `-1.1` checkpoint (`da3-giant-1.1`) |
   | `DA3-Nested-Giant-Large` | ~1.25B | CC BY-NC 4.0 | Nested multi-view variant — prefer `da3-nested-giant-large-1.1`; already outputs metric depth (see metric note below) |
   | `DA3Mono-Large` | 350M | Apache-2.0 | Monocular-optimized |
   | `DA3Metric-Large` | 350M | Apache-2.0 | Metric depth (real-world scale) |

   > **License note**: the DA3 Small/Base/Mono/Metric checkpoints are **Apache-2.0**;
   > the DA3 Large/Giant/Nested checkpoints are **CC BY-NC 4.0** (non-commercial). For Large, Giant and
   > Nested, prefer the retrained `-1.1` checkpoints for better quality. The camera-depth
   > checkpoints below are a separate **CC BY-NC 4.0** family (see their note).

4. **Camera-depth model variants** (a separate Camera Depth Model family, trained per camera
   model, on
   [depth-anything/camera-depth-models](https://huggingface.co/collections/depth-anything/camera-depth-models)):

   | Model | Target device | Notes |
   |-------|--------------|-------|
   | `d405` | Intel RealSense D405 | Camera-specific depth |
   | `d435` | Intel RealSense D435 | Camera-specific depth |
   | `l515` | Intel RealSense L515 | Camera-specific depth |
   | `kinect` | Microsoft Kinect | Camera-specific depth |
   | `zed2i-neural` | Stereolabs ZED 2i | Neural (stacked) variant |
   | `zed2i-quality` | Stereolabs ZED 2i | Quality (stacked) variant |
   | `base` | Generic RGB camera | Camera-base variant |

   > **Camera-depth note**: these are **Camera Depth Models (CDM)**, not Depth Anything 3
   > checkpoints. They take RGB **plus sensor depth** (4-channel input), ship as `.ckpt`/`.pth`
   > with no DA3 `config.json`, and are **CC BY-NC 4.0**. They cannot be exported with
   > `export_onnx.py` or loaded by `inference_app`; `export_onnx.py --model-name d435` exits
   > with that explanation. `--camera` in the CLI is advisory: it selects the recommended HF
   > identifier, drives the metric-scaling default, and never changes the loaded graph.

---

## Installation

### Install Dependencies (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y cmake libopencv-dev
# Component-only alternative (avoids the VTK/HIP chain that libopencv-dev pulls):
# sudo apt-get install -y libopencv-core-dev libopencv-imgproc-dev \
#   libopencv-imgcodecs-dev libopencv-highgui-dev libopencv-videoio-dev

# Compiler — any C++20-capable version:
sudo apt-get install -y clang-18
# or: gcc (GCC 12+ is required; Ubuntu 24.04 ships GCC 13)

# Optional (faster builds):
sudo apt-get install -y ninja-build

# Optional (linting):
sudo apt-get install -y clang-format-18 clang-tidy-18
```

### RealSense udev Rules (once, for non-root access)

The D435's USB node is not writable by a normal user until the librealsense udev
rules are installed. The rules ship with the SDK source:

```bash
sudo cp build-rs/_deps/librealsense-src/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
# replug the camera
```

With a system librealsense install the same file is usually at
`/etc/udev/rules.d/99-realsense-libusb.rules` already after the package installs
its rules.

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

### Build with RealSense Capture (Optional)

RealSense support is additive: the backend choice is unchanged.

```bash
cmake -S . -B build-rs -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_REALSENSE=ON
cmake --build build-rs --parallel
```

CMake first looks for a system `realsense2` package, then builds pinned
librealsense v2.59.1 into `build-rs/_deps`. The first configure downloads the
source, so it needs network access once. A binary built without the flag reports
that `--input-realsense` needs `-DUSE_REALSENSE=ON`.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `-DUSE_ONNX_RUNTIME=ON/OFF` | ON | Enable ONNX Runtime backend |
| `-DUSE_TENSORRT=ON/OFF` | OFF | Enable TensorRT backend |
| `-DUSE_REALSENSE=ON/OFF` | OFF | Enable Intel RealSense capture (`--input-realsense`) |
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
# Video input (frames extracted via OpenCV, then the directory path runs)
./build/inference_app model.onnx --input-video clip.mp4 --fps 5 --task multiview
# Camera-depth preset (Hugging Face depth-anything/camera-depth-models; exact fx/fy via --focal)
./build/inference_app model.onnx image.jpg --task metric --camera d435 --focal 600
# Multi-view: pass a directory (sorted *.jpg/*.jpeg/*.png → output_depth_%03d.jpg)
./build/inference_app model.onnx ./views/ --task multiview
# RealSense: capture 3 frames + aligned depth, then run the metric path with
# the device's own focal length (requires -DUSE_REALSENSE=ON)
./build-rs/inference_app model.onnx --input-realsense 3 --camera d435 --task metric \
  --capture-dir captures/d435
# Pose: poses.txt (N×12 world-to-camera) + intrinsics.txt (N×4 fx fy cx cy)
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

Notes:

- Per-frame multi-view is a baseline. Spatially consistent fusion needs an any-view ONNX graph (`python export_onnx.py --task multiview|pose|… --use-ray-pose`).
- Dynamic-axes exports are trace-shape-locked (finding F-1, proven 2026-10-13: a 504-trace model fails on a 252×504 input in the rotary position embedding (RoPE) broadcast). Match `--process-res` to the trace shape, or export with `--fixed-size` (required for TensorRT).
- Tasks whose head the loaded ONNX graph lacks degrade gracefully with a warning: identity poses, a depth-lifted cloud, or placeholder Gaussians. They never produce a silent wrong result.
- Licenses: Small, Base, Mono, and Metric are Apache-2.0. Large, Giant, and Nested are CC BY-NC 4.0; prefer the `-1.1` checkpoints.

### RealSense Capture

`--input-realsense N` captures `N` frames into a structured directory and then
runs the normal directory path, so `mono`, `metric`, `multiview`, `pose`,
`reconstruction`, and `gaussian` all work unchanged:

```bash
./build-rs/inference_app <model.onnx> --input-realsense 3 \
  --camera d435 --task metric --capture-dir captures/d435
```

`<model.onnx>` is your exported model (`export_onnx.py --output model.onnx`;
for a smoke test without real weights,
`tests/fixtures/tiny_depth.onnx` loads and runs). Capture happens before
the model is opened, so a bad model path still captures frames and then
exits non-zero — pass a valid model to get depth outputs too. `captures/`
is created by the run; it is not in the repo.

The capture directory holds:

```
captures/d435/
  color/000.png … 002.png    lossless BGR
  depth/000.png … 002.png    16-bit millimetres, aligned to color
  intrinsics.txt              fx fy cx cy per line (existing format)
  capture.txt                 device, size, counts, timestamps
```

`--focal` is optional with a RealSense source: when omitted, the metric focal is
`mean(fx, fy)` from the active color stream. An explicit `--focal` still wins.
Only the mean focal carries over — per-frame pose/fusion intrinsics reuse it
with a centred principal point, not the device `cx/cy` (exact device values
stay in the capture directory's `intrinsics.txt`).
Depth and color are captured in the same pass, so the hardware depth can be used
as a later reference for the DA3 metric output.

Use `--rs-size WxH` (default `1280x720`) and `--rs-fps K` (default `30`) to pick
the color stream. Depth uses the device default profile at the same rate and is
resampled to the color resolution. A directory passed with `--capture-dir` is
created if needed; otherwise captures go under
`<output-dir>/realsense_<timestamp>/`.

**Troubleshooting**

- **"No device connected" with the camera plugged in:** both the USB device node
  and the UVC `/dev/video*` nodes must be accessible. Install the librealsense
  udev rules for the USB node, and grant the user access to the video nodes
  (`video` group or a `video4linux` udev rule).
- **"Frame didn't arrive" or incomplete frames:** check the negotiated link with
  `lsusb -t`. D400 cameras are USB 3 devices; on a USB 2 port they may enumerate
  and expose profiles but fail to stream. Use a USB 3 port and a USB 3 cable.

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
Usage: ./build/inference_app <path_to_model> [<path_to_image|dir>] [options]

Arguments:
  path_to_model   ONNX model (.onnx) or TensorRT engine (.engine / .trt)
  path_to_image   Input image (JPG, PNG, ...) or directory of images

Options:
  --task TASK       Native DA3 task: mono, metric, multiview, pose,
                    reconstruction, gaussian (default: mono)
  --camera ID       Camera-depth preset: d405, d435, l515, kinect,
                    zed2i-neural, zed2i-quality, base (default: none)
  --focal PX        Focal length in pixels for metric scaling (default: 500;
                    pose and reconstruction default to 1.2*W)
  --input-dir DIR   Directory of input images (overrides positional image)
  --output-dir DIR  Directory for output files (default: current directory)
  --already-metric  Depth is already in meters; skip focal*net/300 scaling
  --use-ray-pose    Use ray-based pose estimation for the pose task
  --conf            Also save the confidence map as output_conf.jpg
  --no-upscale      Keep depth at model output resolution (no resize)
  --process-res N   Resize the longest side to N before inference (default: 504)
  --dump-raw PATH   Write the raw float32 depth map to PATH as a .npy file
  --views-per-batch K
                    Stream views in chunks of K (default: trace N for static
                    batch models, frame-by-frame otherwise)
  --ply-stride S    Subsample Polygon File Format (PLY) outputs every S-th
                    pixel (default: 1)
  --input-video PATH
                    Extract frames via cv::VideoCapture into a temporary
                    directory, then run the directory path (pairs with --fps)
  --fps K           Sample K frames per second from --input-video (default: all)
  --input-realsense N
                    Capture N frames from an Intel RealSense camera, then run
                    the directory path. Requires -DUSE_REALSENSE=ON. Reads
                    exact color intrinsics and aligned depth; auto-fills
                    --focal when absent
  --rs-size WxH     RealSense color size (default: 1280x720)
  --rs-fps K        RealSense frame rate (default: 30)
  --capture-dir DIR Keep captured color/ and depth/ in DIR (default:
                    <output-dir>/realsense_<timestamp>)
  --help            Show help and exit
```

The `--help` output also prints the task, checkpoint, camera, and license tables.

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
> Dynamic axes are declared in the ONNX graph, but a traced export is locked to
> the shape used during tracing. Use the trace resolution, or export with
> `--fixed-size` and run that exact size.

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
  optional confidence `[1, H, W]`; exported with **opset 17** and dynamic axes.
  The axes are declared, but current traced exports remain shape-locked to the
  trace resolution (see [Native Task Selection](#native-task-selection---task)).
- **C++ preprocessing** (`Config{process_res = 504, patch_size = 14}`): the input
  image's longest side is resized to `process_res`, `H` and `W` are rounded to be
  divisible by 14, and ImageNet normalization
  (`means = {0.485, 0.456, 0.406}`, `stds = {0.229, 0.224, 0.225}`) is applied.

---

## Any-View Support

Upstream DA3 also supports any-view (multi-view) reconstruction. This C++ engine
implements the front end and the standard-file outputs, while true N-view fusion
remains deferred.

**Implemented in the CLI:** the `multiview`, `pose`, `reconstruction`, and
`gaussian` tasks; directory input and `--views-per-batch` chunking; `poses.txt`
and `intrinsics.txt`; the point-cloud and Gaussian PLY writers; and the
`--input-video` frame extractor. True spatially consistent fusion still needs an
any-view ONNX graph.

**Deferred:** streaming windows below 12 GB; glTF and NumPy exports; 3D Gaussian
Splatting (3DGS) video rendering; long-sequence SLAM; and Gradio parity. The
upstream I/O remains the reference: depth and confidence `[N, H, W]`, extrinsics
`[N, 3, 4]`, intrinsics `[N, 3, 3]`, `use_ray_pose`, and `ref_view_strategy`.

The monocular path remains the default (`--task mono`). For details, see the [native tasks CLI requirements](specs/2026-10-12-da3-native-tasks/requirements.md), the [any-view fusion validation](specs/2026-10-13-da3-anyview-fusion/validation.md), and the [deferred revisit validation](specs/2026-10-13-da3-deferred-revisit/validation.md).

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

## Continuous Integration

GitHub Actions runs `.github/workflows/ci.yml` for pushes and pull requests targeting `master`, plus manual dispatch. Every job is blocking and reports into a single `Quality Gate` status. `master` requires that check with strict up-to-date branches and blocks force pushes and deletions.

| Job | What it enforces |
|-----|------------------|
| `Format` | `clang-format-18` across the whole tree |
| `Build & Test (clang, gcc)` | Release build with `-DWERROR=ON`, unit tests, benchmark smoke test |
| `Sanitizers` | Debug build with ASan + UBSan, unit tests |
| `Clang-Tidy` | `clang-analyzer-*` and `bugprone-*` findings fail; the full configured report is uploaded as an artifact |
| `Cppcheck` | `warning`, `performance`, and `portability` findings fail |
| `Coverage` | gcov line coverage floor on the tested library surface, plus diff coverage on changed lines |
| `Benchmark regression` | Google Benchmark medians compared with the tracked `benchmarks/data.json` baseline; 150% of the stored median fails |
| `Docker build` | Builds the multi-stage image on Ubuntu 24.04 and smoke-tests `inference_app --help` |
| `Quality Gate` | Aggregates the jobs above; required status check |

### Gate behavior

The gate fails closed: a skipped or cancelled dependency is a failure, not a pass. Branch protection is a repository setting, not a file. `master` currently requires `Quality Gate` with up-to-date branches, and blocks force pushes and deletions. Direct pushes by admins are still possible; enable “Require a pull request before merging” in **Settings → Branches** to block those too.

### Coverage

The coverage gate is calibrated to the model-free test surface. It enforces 80% line coverage on `depth_anything3_inference`, `processing_utils`, and both backend implementations, plus 70% diff coverage on changed lines. The ONNX Runtime backend is exercised end-to-end with `tests/fixtures/tiny_depth.onnx` (216 bytes), so CI needs no model weights. `main.cpp` is reported but not gated until CLI tests exist.

### Benchmarks

Benchmark data lives in the tracked file `benchmarks/data.json`; there is no data branch and nothing is pushed. Pull requests and `master` pushes compare against that file and fail at 150% of the stored median. Refresh it from the `benchmark-results` artifact when the baseline legitimately changes. The threshold is deliberately loose because hosted runners are noisy; tighten it after collecting stable history.

### Formatting and static analysis

The whole C++ tree is formatted and checked on every run. Format files with `clang-format-18 -i`; run `scripts/ci/check-format.sh --all` to check without modifying them.

For a strict static-analysis audit, run the workflow manually with `strict_tidy=true` (Actions → CI → Run workflow).

### Local quality run

The same blocking checks run locally, using a separate `build-ci` directory:

```bash
scripts/ci/run_quality.sh              # whole-tree format + build + tests + tidy + cppcheck
scripts/ci/run_quality.sh --skip-tests # skip ctest and the benchmark smoke test
```

Coverage needs `gcovr` and `diff-cover` on `PATH` (for example in a virtualenv), then:

```bash
scripts/ci/run-coverage.sh
```

For a local benchmark run with the same median-of-five format:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBENCHMARKS=ON
cmake --build build --target benchmarks --parallel
./build/benchmarks --benchmark_min_time=0.2s --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=build/benchmark.json --benchmark_out_format=json
python3 scripts/ci/benchmark-to-json.py build/benchmark.json build/benchmark-action.json
```

### Individual tools

| Tool | Purpose | How to run |
|------|---------|------------|
| `clang-format-18` | Code formatting | `scripts/ci/check-format.sh --all` (check) or `clang-format-18 -i <files>` (fix) |
| `clang-tidy-18` | Static analysis | `scripts/ci/run-clang-tidy.sh` (requires `build/compile_commands.json`) |
| `cppcheck` | Flow-based static analysis | `scripts/ci/run-cppcheck.sh` |
| `gcovr` + `diff-cover` | Line and changed-line coverage | `scripts/ci/run-coverage.sh` |
| AddressSanitizer + UBSan | Runtime memory/UB detection | `-DSANITIZERS=ON` at configure time |

`clang-tidy` uses the full check set from `.clang-tidy`, but only `clang-analyzer-*` and `bugprone-*` are blocking by default; the other configured checks are reported in `build/clang-tidy.log`. Override locally with `TIDY_WARNINGS_AS_ERRORS=<glob>` or `STRICT_TIDY=true`. The TensorRT backend is excluded because the default configure does not install TensorRT.

---

## Acknowledgements

- **Depth Anything 3** model: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
