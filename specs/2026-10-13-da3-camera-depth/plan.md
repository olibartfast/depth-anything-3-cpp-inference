# Plan — DA3 Camera-Depth (Phase A)

## Phases (thin)
### A1 — `--camera` preset + checkpoint map
`parse_camera`/`camera_to_string`/`camera_checkpoint` in header; `Options.camera` + `--camera` parse in main; preset implies metric-mode default focal handling (explicit `--focal` wins). Writable: `src/depth_anything3_inference.hpp`, `src/main.cpp`.

### A2 — Export + docs + tests
`export_onnx.py`: camera ids in `--model-name` help + shape-lock warning text; README camera table; R-4 unit tests. Writable: `export_onnx.py`, `README.md`, `tests/unit/test_depth_anything3_inference.cpp`.

## Integration
- After A1+A2: `cmake --build build --parallel && ctest --test-dir build -R UnitTests` + `./build/inference_app --help` shows cameras + `python3 export_onnx.py --help`.

# Validation — DA3 Camera-Depth (Phase A)

| ID | Requirement | Check |
|----|-------------|-------|
| V-1 | R-1 | `--help` lists 7 camera presets; `--task metric --camera d435` runs (falls back gracefully without weights: parse-level) |
| V-2 | R-2 | README camera table matches HF collection ids; no hardcoded fx in code (`grep focal src/main.cpp` shows only `--focal`/1.2*W) |
| V-3 | R-3 | `export_onnx.py --help` + README document trace-shape-lock + `--fixed-size` guidance |
| V-4 | R-4 | New unit tests pass model-free |
| V-5 | R-5 | `--model-name` help mentions camera ids |

## Evidence
- 2026-10-13: build clean; `ctest -R UnitTests` pass (CameraParse/CameraCheckpoint/CameraFocalPrecedence); `py_compile` OK; `--help` shows 7 presets; `--task metric --camera base` end-to-end on da3-small: prints preset id + skip-scaling note + writes metric npy. Camera repo ids verified against HF collection page (`camera-depth-model-<slug>`); no fx/fy invented (`grep` shows only `--focal`/1.2*W). F-1 shape-lock documented in export `--help` + README.
