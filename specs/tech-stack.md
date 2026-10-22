# Tech Stack — Depth Anything 3 C++ Inference

This document records the fixed stack, the preferred implementation patterns, and the explicit non-choices. A fixed item changes only with a recorded decision.

## Fixed choices

- **Language.** C++20. Clang 15+ and GCC 12+ are supported; CI builds with Clang 18 and GCC.
- **Build system.** CMake 3.12 or newer, with Ninja recommended. The build produces the static library `da3_inference_lib`, the `inference_app` executable, and the `unit_tests` executable.
- **Libraries.** OpenCV 4.x (`core`, `imgproc`, `imgcodecs`, `highgui`, `videoio`) and the platform thread library.
- **Capture (optional).** Intel RealSense through librealsense2 2.59.1, gated behind `-DUSE_REALSENSE=ON` (default `OFF`). CMake uses a system package when found and otherwise builds the pinned source. Capture is a source adapter and never changes the inference backend. Non-root Linux access requires the librealsense udev rules, a one-time host step.
- **Backends.** Exactly one compile-time backend. ONNX Runtime 1.21.0 is downloaded automatically and supports Linux, Windows, and macOS on CPU or GPU. TensorRT 11.2.1.2 is downloaded automatically and requires the CUDA 13.3 toolkit; it supports Linux with an NVIDIA GPU only. The TensorRT pin matches `rf-detr-cpp-inference`.
- **Tests.** GoogleTest 1.12.1 via FetchContent. `ctest -R UnitTests` runs without downloading a model file. Benchmarks are opt-in with `-DBENCHMARKS=ON` and use google/benchmark 1.9.1.
- **Linting.** `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` and related warnings. `-DWERROR=ON` and `-DSANITIZERS=ON` are opt-in. The lint toolchain is `clang-format-18`, `clang-tidy-18`, and `cppcheck`.
- **Docker.** A multi-stage Ubuntu 24.04 build. The runtime stage installs the OpenCV runtime libraries, copies the ORT shared library into the image, and runs `ldconfig`.

## Preferred patterns

- **Backend abstraction.** Use the strategy interface `da3::backend::InferenceBackend` and the `create_backend()` factory. Processing helpers live in `da3::processing`: `normalize_image`, `compute_output_size`, `preprocess_frame`, and the colormap helpers.
- **Capture boundary.** Hardware sources implement `da3::camera::Source` and return one `Frame` at a time. `capture_to_directory()` owns file layout, depth encoding, and intrinsics output, so capture logic is testable with a synthetic source and inference never depends on a camera SDK.
- **Configuration defaults.** `Config{process_res=504, patch_size=14, means={0.485,0.456,0.406}, stds={0.229,0.224,0.225}}`.
- **Preprocessing.** Resize so the longest side equals `process_res`, round each dimension to the nearest multiple of 14, convert BGR to RGB, divide by 255, convert to CHW layout, and apply ImageNet normalization.
- **ONNX contract.** The input is `image [1,3,H,W]` as float. The outputs are `depth [1,H,W]` and an optional `conf [1,H,W]`. Height and width must be divisible by 14. Dynamic axes are declared by default, but traced exports remain shape-locked; see the camera-depth packet for the trace-shape limit.
- **Postprocessing.** Depth is `CV_32F`. Upscaling to the original size is an optional bilinear resize. Depth visualization uses `COLORMAP_INFERNO`; confidence uses `COLORMAP_JET`.
- **Export wrapper.** `export_onnx.py` wraps the model in a thin `MonocularWrapper`: `unsqueeze(1)` → `model.model(imgs, None, None, None, False, False, "saddle_balanced")`, then squeeze `[B,N,1,H,W]` to `[1,H,W]`. The export uses opset 17.

## Open decisions

These are decided per feature:

- N-view input layout.
- Depth and confidence `[N,H,W]`; extrinsics `[N,3,4]`; intrinsics `[N,3,3]`.
- `use_ray_pose` and `ref_view_strategy`.
- Streaming windows.
- glTF, PLY, NumPy, and 3DGS exports.

## Explicit non-choices

- **No Python at runtime.** The product is a portable edge and production binary.
- **No multi-backend binary.** ORT and TensorRT conflict at link and runtime time. The backend is selected with `USE_ONNX_RUNTIME` or `USE_TENSORRT`.
- **No bundled weights.** Licensing and size make that impractical. Users export a model with the script or download one from Hugging Face.
- **No training, streaming, or SLAM in this slice.** Those need the any-view design spike first.

## Cross-cutting constraints

- Height and width must be divisible by 14 because the DINOv2 backbone uses 14-pixel patches. `compute_output_size` enforces this.
- **Licensing.** Upstream code is Apache-2.0. The weights are split: DA3-Small, DA3-Base, DA3Mono-Large, DA3Metric-Large, and the camera-depth base are Apache-2.0. DA3-Large, DA3-Giant, and DA3-Nested are CC BY-NC 4.0. Prefer the retrained `-1.1` checkpoints for Large, Giant, and Nested.
- **Metric conversion.** `metric_depth = focal * net_output / 300`, where focal is the mean of `fx` and `fy`. Nested models already output meters.
- **Performance.** Keep the preprocessing allocation at `3*H*W` and avoid per-frame heap churn in the hot path. Benchmarks currently cover preprocessing only.

## Open questions

- **Canonical C++ struct for any-view predictions (Q-1).** The struct must carry depth, confidence, pose, and intrinsics. Owner: maintainer.

_Revision: 2026-10-11 — grounded in `CMakeLists.txt`, `src/backends/`, `src/processing_*`, `tests/unit/`, `Dockerfile`, and `export_onnx.py`._
_Revision: 2026-10-21 — added optional RealSense capture (`USE_REALSENSE`, librealsense2 2.59.1) as a source adapter; default builds are unchanged._
