# Validation — DA3 Camera-Depth (Phase A)

Defined before implementation (SDD). Evidence recorded with dates on execution.

| ID | Requirement | Check | Expected |
|----|-------------|-------|----------|
| V-1 | R-1 camera flag | `./build/inference_app --help` | Lists 7 presets + collection note; `--task metric --camera d435` runs |
| V-2 | R-2 no invented focals | `grep focal src/main.cpp` + README table | Only `--focal`/1.2*W; table matches HF `camera-depth-model-<slug>` ids (verified 2026-10-13) |
| V-3 | R-3 shape-lock | `export_onnx.py --help` + README | Trace-shape-lock + `--fixed-size` guidance present |
| V-4 | R-4 unit tests | `ctest -R UnitTests` | CameraParse/CameraCheckpoint/CameraFocalPrecedence pass model-free |
| V-5 | R-5 export ids | `export_onnx.py --help` | `--model-name` help mentions camera ids; `_resolve_model_id` maps to `depth-anything/camera-depth-model-<slug>` |

## Evidence
- 2026-10-13: build clean; `ctest -R UnitTests` pass (CameraParse/CameraCheckpoint/CameraFocalPrecedence); `py_compile` OK; `--help` shows 7 presets; `--task metric --camera base` end-to-end on da3-small: prints preset id + skip-scaling note + writes metric npy. Camera repo ids verified against HF collection page (`camera-depth-model-<slug>`); no fx/fy invented (`grep` shows only `--focal`/1.2*W). F-1 shape-lock documented in export `--help` + README.
