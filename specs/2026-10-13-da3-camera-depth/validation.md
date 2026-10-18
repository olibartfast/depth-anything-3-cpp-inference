# Validation — DA3 Camera-Depth and Export Shape-Lock

These checks were defined before implementation (spec-driven development, SDD). Evidence is recorded with dates when the checks run.

## Checks

| ID | Requirement | Check | Expected |
|---|---|---|---|
| V-1 | R-1 camera flag | `./build/inference_app --help` | Lists seven presets and the collection note. `--task metric --camera d435` runs. |
| V-2 | R-2 no invented focal lengths | `grep focal src/main.cpp` plus the README table | Only `--focal` and `1.2*W` appear. The table matches the Hugging Face IDs `camera-depth-model-<slug>`, verified 2026-10-13. |
| V-3 | R-3 shape lock | `export_onnx.py --help` plus the README | Explains the trace-shape lock and gives `--fixed-size` guidance. |
| V-4 | R-4 unit tests | `ctest -R UnitTests` | `CameraParse`, `CameraCheckpoint`, and `CameraFocalPrecedence` pass model-free. |
| V-5 | R-5 export IDs | `export_onnx.py --help` | `--model-name` mentions camera IDs, and `_resolve_model_id` maps them to `depth-anything/camera-depth-model-<slug>`. |

## Evidence

**2026-10-13:** The build was clean, and `ctest -R UnitTests` passed, including `CameraParse`, `CameraCheckpoint`, and `CameraFocalPrecedence`. `py_compile` succeeded. `--help` shows seven presets. `--task metric --camera base` ran end to end on `da3-small`: it printed the preset ID and the skip-scaling note, and wrote the metric NumPy file. Camera repository IDs were verified against the Hugging Face collection page (`camera-depth-model-<slug>`). No `fx` or `fy` values were invented; `grep` shows only `--focal` and `1.2*W`. The F-1 shape lock is documented in the export `--help` and the README.
