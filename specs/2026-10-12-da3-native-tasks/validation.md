# Validation — DA3 Native Tasks CLI

These checks were defined before implementation, following the project’s spec-driven development (SDD) practice. Evidence is recorded with dates when the checks run.

## Checks

| ID | Requirement | Check | Expected |
|---|---|---|---|
| V-1 | R-1 task flag | `./build/inference_app --help` | Lists `mono`, `metric`, `multiview`, `pose`, `reconstruction`, and `gaussian` with their checkpoints. A no-argument run exits 1 and prints usage. |
| V-2 | R-1 default compatibility | `ctest -R UnitTests` plus the parse test | `parse_task("")` returns `mono`; an invalid task throws. |
| V-3 | R-2 metric path | Unit test `MetricTaskScaling` | `focal * net / 300` goes through the helper, and `--already-metric` skips scaling. |
| V-4 | R-3 multi-view input | Unit test `CollectImages` plus a manual directory run | Sorted JPG/PNG glob, with indexed `output_depth_%03d.jpg` output. |
| V-5 | R-4 pose output | Pose-format unit test and a `--task pose` run on a mono ONNX model when available, otherwise a mock backend | `poses.txt` has N×12 floats; `intrinsics.txt` has N×4; a warning appears when the pose head is missing. |
| V-6 | R-5 reconstruction | PLY-header unit test and a `--task reconstruction` run | ASCII PLY with `vertex` count equal to `H*W*N` and fields `x y z r g b`. |
| V-7 | R-6 Gaussian output | Gaussian-PLY unit test and a `--task gaussian` run | `output_gaussians.ply` contains opacity and scale columns and is non-empty. |
| V-8 | R-7 export | `python3 -m py_compile export_onnx.py` and `--help` | `--task` and `--use-ray-pose` are present, and output names are documented. |

## Evidence

- **2026-10-12:** `cmake --build build --parallel` completed cleanly. `ctest -R UnitTests` passed 100%, including the new `TaskParse`, `CollectImages`, `PoseTxt`, `PoseIntrinsics`, `PlyHeader`, and `GaussianPly` tests. `python3 -m py_compile export_onnx.py` passed. `./build/inference_app --help` lists six tasks and their checkpoints. A no-argument run exits 1 with usage, and `--task bogus` reports the valid task list.
- **2026-10-12 (review):** R-5 was corrected to the `1.2*W` default through `tasks::default_focal`, with the `PoseIntrinsics.DefaultFocal` test. R-8 was corrected by moving the writers into `da3::tasks` in the header so tests call production code. Export any-view heads now receive dynamic `num_views` axes.
- **Pre-existing scope note:** The `export_onnx.py --dynamo` change, the RoPE patch, and the `MonocularWrapper.squeeze()` change predate this packet. They were uncommitted GPU-gate work and were kept as-is rather than reverted. Pose-head matching is a shape heuristic using trailing `[3,4]` and `[3,3]` dimensions, because the C++ backend exposes output indices and shapes rather than names.
- **2026-10-13 (RTX 3060 Laptop, ORT CPU, `/tmp/da3-small-dyn.onnx`, two square frames at 1008²):** all six tasks ran end to end. Mono, metric, and multiview produced per-frame depths. `poses.txt` contained two identity 12-value rows and the expected warning. `intrinsics.txt` contained `1209.6 = 1.2*1008`. `output_cloud.ply` reported vertex count `2032128 = 1008*1008*2`. `output_gaussians.ply` contained opacity, scale, and rotation columns.

## Follow-on findings

- **F-1 — Dynamic-axes export is trace-shape-locked.** A 252×504 input fails in the rotary position embedding (RoPE) broadcast (`Add` of 649 by 1297); 504×504 works. The export path must document and enforce the trace shape.
- **F-2 — ASCII per-pixel PLY files are large.** Two frames produced a 76 MB cloud and 179 MB of Gaussians. Binary PLY or stride subsampling is a follow-on.
