# Plan — DA3 Native Tasks CLI

The work is split into three thin stages, each producing a runnable result. The acceptance checks are in [validation.md](validation.md).

## Stage A — Task enum, CLI parsing, and help

Add `TaskType` to `depth_anything3_inference.hpp`, along with `parse_task`, `task_to_string`, and `recommended_checkpoint`. Wire `main.cpp` for `--task`, `--focal`, `--input-dir`, `--output-dir`, `--already-metric`, `--use-ray-pose`, and `--help`. The default `mono` task keeps old invocations working.

| ID | Task |
|---|---|
| A1 | Add the task enum and helper functions to the header. |
| A2 | Add argument parsing and usage text to `main.cpp`. |
| A3 | Add a README usage snippet. |

## Stage B — Metric and multi-view processing

Wire `--focal` scaling through `metric_depth_from_relative` and support `--already-metric`. Add the `collect_image_paths` directory helper and the per-frame loop that writes indexed outputs.

| ID | Task |
|---|---|
| B1 | Wire metric scaling. |
| B2 | Add multi-view input and the per-frame loop. |
| B3 | Extend the export script with `--task mono`, `metric`, and `multiview`, including output names. |

## Stage C — Pose, reconstruction, and Gaussian files

Write `poses.txt` and `intrinsics.txt`, with identity fallback and model-output extraction when available. Add the depth-to-cloud `save_ply` writer and the Gaussian `save_gaussian_ply` writer. Extend the export script with `--task pose`, `reconstruction`, and `gaussian`, plus `--use-ray-pose` passthrough.

| ID | Task |
|---|---|
| C1 | Add pose I/O. |
| C2 | Add the point-cloud PLY writer. |
| C3 | Add the Gaussian PLY writer. |
| C4 | Add the export modes. |
| C5 | Add unit tests for R-8. |

## Integration checkpoints

- After Stage A, `cmake --build build && ./build/inference_app --help` shows the tasks, and `ctest -R UnitTests` passes.
- After Stages B and C, checks V-1 through V-8 in [validation.md](validation.md) are green.

## Delegation groups

- **Implementation packet 1:** Stages A and B. Writable paths are `src/*`; read-only paths are `tests/unit/mock_backend.hpp` and `src/backends/*`. Acceptance: build plus `ctest -R UnitTests`.
- **Implementation packet 2:** Stage C and `export_onnx.py`. Writes are limited to the same paths plus `export_onnx.py`. Acceptance: the same tests plus `python3 -m py_compile export_onnx.py`.
- **Reviewer:** Read-only diff review against R-1 through R-8 in [requirements.md](requirements.md). No writes.
