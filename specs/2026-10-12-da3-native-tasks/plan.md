# Plan — DA3 Native Tasks CLI

## Phases (thin, each runnable)
### Phase A — Task enum + CLI parse + help (walk)
Deliver: `TaskType` in `depth_anything3_inference.hpp`, `parse_task`/`task_to_string`/`recommended_checkpoint`, `main.cpp --task/--focal/--input-dir/--output-dir/--already-metric/--use-ray-pose/--help` wired; default `mono` keeps old invocations working.
Tasks: A1 header enum+helpers, A2 main arg parse + usage, A3 README usage snippet.

### Phase B — Metric + multiview loop (run)
Deliver: `--focal` scaling via `metric_depth_from_relative`, `--already-metric` skip; dir glob helper `collect_image_paths`, per-frame loop saving indexed outputs.
Tasks: B1 metric wiring, B2 multiview input+loop, B3 export script `--task mono/metric/multiview` names.

### Phase C — Pose + reconstruction + gaussian files (run)
Deliver: `poses.txt`/`intrinsics.txt` writers (identity fallback + model-output extraction by name when available), depth→cloud `save_ply`, gaussians `save_gaussian_ply`; export script `--task pose/reconstruction/gaussian` + `--use-ray-pose` passthrough.
Tasks: C1 pose I/O, C2 PLY cloud, C3 gaussian PLY, C4 export modes, C5 unit tests (R-8).

## Integration checkpoints
- After A: `cmake --build build && ./build/inference_app --help` shows tasks; `ctest -R UnitTests` passes.
- After B+C: full `validation.md` V-1..V-8 green.

## Task groups for delegation (orchestration skill)
- Packet 1 (implementer): Phase A+B — writable `src/*`, read-only `tests/unit/mock_backend.hpp`, `src/backends/*`; acceptance: build + `ctest -R UnitTests`.
- Packet 2 (implementer): Phase C + `export_onnx.py` — same paths + `export_onnx.py`; acceptance: same + `python3 -m py_compile export_onnx.py`.
- Reviewer: read-only diff review vs `requirements.md` R-1..R-8, zero writes.
