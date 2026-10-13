# Requirements — DA3 Native Tasks CLI

Spec: `specs/2026-10-12-da3-native-tasks/` · Roadmap: Phase 3 thin slice (un-defer any-view front-end)
Interview answers (2026-10-12): multi-view input = image directory; outputs = standard files; export script extended.

## Goal
Add `--task` to `inference_app` selecting each upstream DA3 native task, with working C++ dispatch + matching `export_onnx.py --task` graphs.

## Native tasks (upstream truth)
| # | Task | What you get | Best checkpoints |
|---|------|--------------|------------------|
| T-1 | `mono` | Dense relative depth `[1,H,W]` from 1 image | DA3-SMALL/BASE/LARGE/GIANT |
| T-2 | `metric` | Depth in meters from 1 image (`focal*net/300`) | DA3METRIC-LARGE |
| T-3 | `multiview` | Spatially consistent depth `[N,H,W]` across views | DA3-LARGE/GIANT |
| T-4 | `pose` | Extrinsics `[N,3,4]` + intrinsics `[N,3,3]` | DA3-LARGE/GIANT (not METRIC) |
| T-5 | `reconstruction` | Fused point cloud / point maps via depth-ray rep. | DA3-GIANT; NESTED-GIANT-LARGE for metric |
| T-6 | `gaussian` | Per-pixel 3D Gaussians for novel-view synthesis | DA3-LARGE/GIANT + Gaussian head |

Upstream API reference: `prediction.{depth, conf, extrinsics, intrinsics}` (`DepthAnything3.inference`), CLI `da3 auto/video`, `use_ray_pose`, nested = any-view giant + metric scaler.

## In Scope
- [R-1] `--task {mono,metric,multiview,pose,reconstruction,gaussian}` on `inference_app`, default `mono` (backward compat). `--help` lists tasks + recommended checkpoints.
- [R-2] Metric path: `--focal PX` (default 500) applies existing `metric_depth_from_relative`; Nested models skip scaling (already meters) via `--already-metric` flag. Saves `output_depth.jpg` + metric `.npy`.
- [R-3] Multi-view input: positional `<image>` accepts file OR directory; plus `--input-dir DIR` override. Glob `*.jpg/png/jpeg` sorted, run each frame, save `output_depth_%03d.jpg` (+ `depth_%03d.npy` with `--dump-raw` as prefix). Warn that per-frame loop is baseline; consistent fusion needs any-view ONNX (R-7).
- [R-4] Pose outputs: `poses.txt` (N lines: `r11 r12 r13 tx r21 ...` w2c, COLMAP-compatible rows) + `intrinsics.txt` (`fx fy cx cy` per view). Extract from model outputs by trailing shape (`[3,4]` extrinsics / `[3,3]` intrinsics — the C++ backend exposes indices + shapes, not names); else identity-pose fallback with clear warning (mono graph has no pose head). Default intrinsics: `--focal` when passed, else 1.2*W, cx=W/2, cy=H/2.
- [R-5] Reconstruction: back-project depth → `output_cloud.ply` (ASCII PLY, vertex x y z r g b) using pinhole intrinsics (`--focal` when passed, else 1.2*W, cx=W/2, cy=H/2). Multi-view appends all frames with per-view pose transform when poses known, else identity.
- [R-6] Gaussian: save `output_gaussians.ply` (per-pixel: x y z + rgb + opacity + scale + rotation placeholder from depth+color) + note that full Gaussian head (scale/rotation SH) needs upstream gsplat export. Never crash on mono graph.
- [R-7] `export_onnx.py --task {mono,metric,multiview,pose,reconstruction,gaussian}` (+ `--use-ray-pose` flag) emitting matching output names: `depth[,conf][,extrinsics,intrinsics][,pointmap][,gaussian_params]`. Mono/metric keep current graph; other modes wrap `model.model(imgs,...)` with N-view batch dim and document shapes.
- [R-8] Unit tests model-free for: task parse round-trip, invalid task rejection, metric formula wiring, PLY writers, pose-txt formatting, multiview dir glob. Writers live in `da3::tasks` (header) so tests call production code, not mirrors.

## Out of Scope
- True N-view consistent fusion / streaming windows (<12GB), `glb/npz/3DGS-video` renderers, SLAM-long, Gradio parity — remain Phase 3 follow-ons.
- Downloading weights; changing one-backend-per-binary rule.
- Video decoder input (use image dir; video via `docs/video-inference.md` ffmpeg extraction stays).

## Decisions
- [D-1] Default `mono` so existing commands keep working.
- [D-2] Directory input (not video) per interview — matches upstream `da3 auto <dir>`.
- [D-3] Standard files (poses.txt/intrinsics.txt/PLY) per interview — not raw-only.
- [D-4] Graceful degradation: tasks requiring heads the loaded ONNX lacks must warn + produce best-effort output (identity pose, depth-lifted cloud/gaussians), never hard-fail without message. Rationale: current C++ only ships mono graphs; users iterate toward any-view exports.

## Constraints
- C++20, one backend per binary, H/W % 14 == 0, `ctest -R UnitTests` model-free.
- Licenses surfaced in `--help`/README: Small/Base/Mono/Metric Apache-2.0; Large/Giant/Nested CC BY-NC 4.0; prefer `-1.1`.

## Dependencies
- Code: `src/main.cpp:47-96`, `src/depth_anything3_inference.{hpp,cpp}`, `src/processing_utils.{hpp,cpp}:99-103`, `export_onnx.py:160-193`, `tests/unit/test_depth_anything3_inference.cpp`.
- Upstream: `DepthAnything3.inference` I/O, Model Zoo table, FAQ metric formula.

## Open Questions
- [Q-1] Canonical Gaussian PLY schema vs gsplat format — owner: maintainer, follow-on after upstream Gaussian-head export lands. Current PLY is documented placeholder.
- [Q-2] Any-view ONNX dynamic-N export shape — owner: maintainer, needs GPU-gate model access.

## Definition of Done
- [ ] Every [R-n] implemented or explicitly deferred with location
- [ ] `validation.md` checks all pass with evidence
