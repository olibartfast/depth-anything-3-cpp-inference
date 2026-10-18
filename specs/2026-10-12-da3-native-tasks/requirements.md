# Requirements — DA3 Native Tasks CLI

This packet adds a `--task` switch to `inference_app` so users can select each native DA3 task from C++. The C++ dispatch must match the graphs produced by `export_onnx.py --task`.

The scope was clarified in an interview on 2026-10-12: multi-view input is an image directory, outputs are standard files, and the export script is extended to match.

- **Spec:** `specs/2026-10-12-da3-native-tasks/`
- **Roadmap:** Phase 3 thin slice; it un-defers the any-view front end.

## Goal

Add `--task` to `inference_app` with working C++ dispatch and corresponding `export_onnx.py --task` graphs.

## Native tasks

These are the upstream task definitions this CLI exposes. The last column names the recommended checkpoints.

| ID | Task | Output | Recommended checkpoints |
|---|---|---|---|
| T-1 | `mono` | Dense relative depth `[1,H,W]` from one image. | DA3-SMALL, DA3-BASE, DA3-LARGE, DA3-GIANT |
| T-2 | `metric` | Depth in meters from one image (`focal * net / 300`). | DA3METRIC-LARGE |
| T-3 | `multiview` | Spatially consistent depth `[N,H,W]` across views. | DA3-LARGE, DA3-GIANT |
| T-4 | `pose` | Extrinsics `[N,3,4]` and intrinsics `[N,3,3]`. | DA3-LARGE, DA3-GIANT (not DA3METRIC) |
| T-5 | `reconstruction` | A fused point cloud or point maps via the depth-ray representation. | DA3-GIANT; NESTED-GIANT-LARGE for metric |
| T-6 | `gaussian` | Per-pixel 3D Gaussians for novel-view synthesis. | DA3-LARGE or DA3-GIANT with a Gaussian head |

The upstream API exposes `prediction.depth`, `prediction.conf`, `prediction.extrinsics`, and `prediction.intrinsics` from `DepthAnything3.inference`. Its CLI provides `da3 auto` and `da3 video`. “Nested” means an any-view Giant model with a metric scaler.

## In scope

- **Task switch.** Add `--task {mono,metric,multiview,pose,reconstruction,gaussian}` to `inference_app`, defaulting to `mono` for backward compatibility. `--help` lists the tasks and recommended checkpoints. (R-1)
- **Metric path.** `--focal PX` (default 500) applies the existing `metric_depth_from_relative` helper. Nested models skip scaling because they already output meters; this is controlled by `--already-metric`. The metric path saves `output_depth.jpg` and `output_metric.npy`. (R-2)
- **Multi-view input.** The positional `<image>` argument accepts a file or a directory, and `--input-dir DIR` overrides it. Images matching `*.jpg`, `*.png`, and `*.jpeg` are sorted, processed one frame at a time, and saved as `output_depth_%03d.jpg`; `--dump-raw` also writes `depth_%03d.npy`. The per-frame loop is a baseline; consistent fusion needs an any-view ONNX graph, tracked as R-7. (R-3)
- **Pose outputs.** Write `poses.txt` with one line per view: `r11 r12 r13 tx r21 ...` in world-to-camera (w2c) form, using COLMAP-compatible rows. Write `intrinsics.txt` with `fx fy cx cy` per view. Extract pose values from model outputs by trailing shape (`[3,4]` for extrinsics and `[3,3]` for intrinsics), because the C++ backend exposes output indices and shapes rather than names. If no pose head exists — as with the mono graph — fall back to identity poses and emit a clear warning. The default intrinsics use `--focal` when supplied; otherwise the focal length is `1.2*W`, with `cx=W/2` and `cy=H/2`. (R-4)
- **Reconstruction.** Back-project depth into `output_cloud.ply`, an ASCII PLY with vertex fields `x y z r g b`, using pinhole intrinsics: `--focal` when supplied, otherwise `1.2*W`, with `cx=W/2` and `cy=H/2`. In multi-view mode, append all frames and apply the per-view pose transform when poses are known; otherwise use identity. (R-5)
- **Gaussian output.** Save `output_gaussians.ply` with a per-pixel placeholder derived from depth and color: position, RGB, opacity, scale, and rotation. The full Gaussian head — spherical-harmonic scale and rotation — needs the upstream gsplat export. Never crash when the loaded graph is monocular. (R-6)
- **Export modes.** Extend `export_onnx.py` with `--task {mono,metric,multiview,pose,reconstruction,gaussian}` and `--use-ray-pose`, emitting matching output names: `depth[,conf][,extrinsics,intrinsics][,pointmap][,gaussian_params]`. Mono and metric keep the current graph. Other modes wrap `model.model(imgs, ...)` with an N-view batch dimension and document the shapes. (R-7)
- **Model-free tests.** Add unit tests for task parsing round-trips, invalid-task rejection, metric-formula wiring, PLY writers, pose-text formatting, and multi-view directory globbing. Put the writers in the `da3::tasks` header so tests exercise production code rather than a mirror. (R-8)

## Out of scope

- True N-view consistent fusion, streaming windows below 12 GB, glTF, NumPy, and 3DGS-video renderers, long-sequence SLAM, and Gradio parity. These remain Phase 3 follow-ons.
- Downloading weights and changing the one-backend-per-binary rule.
- Video decoder input. Use an image directory; video extraction via `docs/video-inference.md` remains a separate workflow.

## Decisions

- **Default to `mono`.** Existing commands keep working. (D-1)
- **Accept a directory, not a video.** This matches the upstream `da3 auto <dir>` workflow and the interview decision. (D-2)
- **Write standard files.** Produce `poses.txt`, `intrinsics.txt`, and PLY rather than raw-only output, as requested in the interview. (D-3)
- **Degrade gracefully.** When a task needs a head that the loaded ONNX graph does not provide, warn and produce the best-effort output — identity pose, a depth-lifted point cloud, or placeholder Gaussians — rather than failing silently. The current C++ runtime ships mono graphs, and users iterate toward any-view exports. (D-4)

## Constraints

- C++20, one backend per binary, and height and width divisible by 14.
- `ctest -R UnitTests` remains model-free.
- Licenses are surfaced in `--help` and the README: Small, Base, Mono, and Metric are Apache-2.0; Large, Giant, and Nested are CC BY-NC 4.0; prefer `-1.1` checkpoints.

## Dependencies

- **Code:** `src/main.cpp:47-96`, `src/depth_anything3_inference.{hpp,cpp}`, `src/processing_utils.{hpp,cpp}:99-103`, `export_onnx.py:160-193`, and `tests/unit/test_depth_anything3_inference.cpp`.
- **Upstream:** the `DepthAnything3.inference` I/O, the Model Zoo table, and the FAQ metric formula.

## Open questions

- **Gaussian PLY schema (Q-1).** The canonical schema versus the gsplat format is unresolved. Owner: maintainer. Resolve after upstream Gaussian-head export lands. The current PLY is a documented placeholder.
- **Any-view ONNX dynamic-N export shape (Q-2).** Owner: maintainer. This needs access to a GPU-gate model.

## Definition of done

- Every requirement (R-1 through R-8) is implemented or explicitly deferred with a location.
- All checks in [validation.md](validation.md) pass with recorded evidence.
