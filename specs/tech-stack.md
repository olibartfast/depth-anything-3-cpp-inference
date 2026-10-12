# Tech Stack — Depth Anything 3 C++ Inference

## Fixed (must not be changed per feature)
- Language: C++20 (Clang 15+/GCC 12+, CI uses clang-18)
- Build: CMake >=3.12, Ninja recommended; `da3_inference_lib` static + `inference_app` + `unit_tests`
- Libs: OpenCV 4.x (`core, imgproc, imgcodecs, highgui`), Threads
- Backends (compile-time, exactly one): ONNX Runtime 1.21.0 auto-download (Linux/Win/macOS, CPU/GPU) OR TensorRT 11.2.1.2 auto-download + CUDA 13.3 toolkit (Linux+NVIDIA only; same pin as rf-detr-cpp-inference)
- Tests: GoogleTest 1.12.1 via FetchContent, `ctest -R UnitTests` model-free; Benchmarks opt-in (`-DBENCHMARKS=ON`, google/benchmark v1.9.1)
- Lint: `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` etc.; opt-in `-DWERROR=ON`, `-DSANITIZERS=ON`; `clang-format-18`, `clang-tidy-18`, `cppcheck`
- Docker: `ubuntu:22.04` builder + runtime, ORT `.so` copied next to binary

## Preferred (follow unless recorded reason not to)
- Strategy pattern `da3::backend::InferenceBackend` + `create_backend()`; `da3::processing` for `normalize_image`, `compute_output_size`, `preprocess_frame`, colormaps
- `Config{process_res=504, patch_size=14, means={0.485,0.456,0.406}, stds={0.229,0.224,0.225}}`
- Preprocess: longest-side→`process_res`, round each dim to nearest multiple of 14, BGR→RGB, /255, CHW, ImageNet norm
- ONNX I/O: input `image [1,3,H,W]` float; outputs `depth [1,H,W]`, optional `conf [1,H,W]`; H,W divisible by 14; dynamic axes by default
- Post: `CV_32F` depth, optional bilinear upscale to orig, `COLORMAP_INFERNO` depth / `COLORMAP_JET` conf
- Export: `export_onnx.py` thin `MonocularWrapper`: `unsqueeze(1)` → `model.model(imgs,None,None,None,False,False,"saddle_balanced")`, squeeze `[B,N,1,H,W]`→`[1,H,W]`, opset 17

## Open (decide per feature)
- N-view input layout, pose/intrinsics I/O (`[N,H,W]`, extrinsics `[N,3,4]`, intrinsics `[N,3,3]`), `use_ray_pose`, `ref_view_strategy`, streaming windows, glb/ply/npz/3DGS exports

## Explicit Non-Choices
- No Python at runtime — rationale: portable edge/production binary
- No multi-backend binary — rationale: ORT vs TRT link/runtime conflict; selected via `USE_ONNX_RUNTIME`/`USE_TENSORRT`
- No bundled weights — rationale: licensing + size; user exports via script or HF download
- No training/streaming/SLAM in this slice — rationale: needs any-view design spike first

## Constraints (cross-feature)
- H,W must be divisible by 14 (DINOv2 patch); `compute_output_size` guarantees it
- Licenses: upstream code Apache-2.0; weights split — `DA3-Small/Base`, `DA3Mono-Large`, `DA3Metric-Large`, camera-depth base Apache-2.0; `DA3-Large/Giant/Nested` CC BY-NC 4.0; prefer `-1.1` retrained checkpoints for Large/Giant/Nested
- Metric: `metric_depth = focal * net_output / 300.` (focal = mean fx,fy); Nested output already meters
- Perf: keep preprocess allocation `3*H*W`, no per-frame heap churn in hot path; benchmarks cover preprocess only

## Open Questions
- [Q-1] Canonical C++ struct for any-view prediction (depth/conf/pose/intrinsics) — owner: maintainer

_Revision: 2026-10-11 — grounded in CMakeLists.txt, src/backends/, src/processing_*, tests/unit/, Dockerfile, export_onnx.py._
