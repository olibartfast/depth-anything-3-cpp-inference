# Validation — DA3 Native Tasks CLI

Defined BEFORE implementation (SDD). Evidence recorded with dates on execution.

| ID | Requirement | Check | Expected |
|----|-------------|-------|----------|
| V-1 | R-1 task flag | `./build/inference_app --help` | Lists `mono,metric,multiview,pose,reconstruction,gaussian` + checkpoints; no-arg exit 1 with Usage |
| V-2 | R-1 default compat | `ctest -R UnitTests` + parse test | `parse_task("")==mono`; invalid task throws |
| V-3 | R-2 metric | unit `MetricTaskScaling` | `focal*net/300` via helper; `--already-metric` skips |
| V-4 | R-3 multiview | unit `CollectImages` + manual dir run | Sorted jpg/png glob; indexed `output_depth_%03d.jpg` |
| V-5 | R-4 pose | unit pose-format + run ` --task pose` on mono onnx (if available) or mock | `poses.txt` N×12 floats, `intrinsics.txt` N×4; warns when head missing |
| V-6 | R-5 reconstruction | unit PLY header + run `--task reconstruction` | ASCII `ply` `vertex` count == H*W*N, `x y z r g b` |
| V-7 | R-6 gaussian | unit gaussian PLY + run `--task gaussian` | `output_gaussians.ply` with opacity/scale cols, non-empty |
| V-8 | R-7 export | `python3 -m py_compile export_onnx.py` + `--help` | `--task` + `--use-ray-pose` present, output names documented |

## Evidence
- 2026-10-12: `cmake --build build --parallel` clean; `ctest -R UnitTests` 100% pass (incl. new TaskParse/CollectImages/PoseTxt/PoseIntrinsics/PlyHeader/GaussianPly tests); `python3 -m py_compile export_onnx.py` OK; `./build/inference_app --help` lists 6 tasks + checkpoints; no-arg exits 1 with Usage; `--task bogus` errors with valid list.
- 2026-10-12 (review): R-5 fixed to 1.2*W default (`tasks::default_focal` + `PoseIntrinsics.DefaultFocal` test); R-8 fixed — writers moved to `da3::tasks` in header, tests call production code; export any-view heads get dynamic `num_views` axes.
- Pre-existing scope note: `export_onnx.py --dynamo` + RoPE patch + `MonocularWrapper.squeeze()` change predate this packet (uncommitted GPU-gate work, kept as-is, not reverted). Pose-head matching is shape-heuristic ([3,4]/[3,3] trailing dims) because the C++ backend interface exposes indices + shapes, not output names.
