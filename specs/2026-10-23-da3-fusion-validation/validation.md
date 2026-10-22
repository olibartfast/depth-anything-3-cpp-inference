# Validation — Fusion Validation and Tracking

Validation is defined before implementation. Every row is either green with a committed command
and captured output, or `BLOCKED` with a dated reason. A static capture is not valid evidence
for V-6: at zero baseline fusion collapses to denoising.

| ID | Requirement | Check | Pass condition |
|----|-------------|-------|----------------|
| V-1 | R-1 focal provenance | Unit test: `base` preset with RealSense auto-filled focal; then explicit `--focal`; then `--camera base` with no camera | Auto-fill skips scaling; explicit focal scales; documented behaviour matches, no silent 1.82x |
| V-2 | R-2 pose providers | Unit test: external, model, and ICP providers return w2c; identity fallback flagged false | All three produce invertible w2c rows; `has_model_poses` is false when the graph exposes no head |
| V-3 | R-3 ICP tracker | Synthetic depth ramp with a known rigid transform; plus a non-rigid outlier scene | Recovered pose within 1 mm and 0.1 degrees; outlier scene rejected |
| V-4 | R-4 Open3D backend | CI builds `-DWITH_FUSION=ON` against the pinned release; inspect output | Build and link green; `fused_mesh.ply` has a face element; default build has no Open3D on the link line |
| V-5 | R-5 confidence weighting | Synthetic two-surface scene, high and low confidence | Fused surface follows the high-confidence surface; binary-mask mode still available |
| V-6 | R-6 hand-held A/B/C | 30–60 frame D435 sequence, path length at least 0.2 m; run modes a, b, c | Three metric rows recorded (Chamfer, ATE, abs_rel 0.3–3.0 m); static run rejected |
| V-7 | R-7 camera-depth input | Read upstream `depth_anything_3` model wrapper for the `d435` checkpoint | Finding recorded: RGB-only or RGB+sensor depth; if supported, wired and tested; if not, question closed |

## Evidence

| Date | Check | Result | Notes |
|------|-------|--------|-------|
| 2026-10-23 | V-1 | **PASS** | Fixed: `focal_from_device` separated from `focal_set` (`src/main.cpp`); RealSense auto-fill no longer marks the focal as user-specified, so `camera_implies_already_metric()` sees only `--focal`. `CameraFocalProvenance` tests cover base+device (skips), base+user (scales), d435+user (scales). Hardware check: `--camera base --input-realsense 1 --task metric` printed "Focal from device intrinsics: 910.259 px (… device-derived, not an explicit --focal)" then "preset 'base' already outputs meters; skipping focal*net/300 scaling"; adding `--focal 500` suppressed the skip note and scaled the output (1.68x on the tiny fixture's noisy range; scaling is per-pixel, so the max ratio is content-dependent). |
| 2026-10-23 | V-2 | PENDING | No `PoseProvider` exists; modes b/c consume `FrameData::pose` directly with an unchecked identity fallback |
| 2026-10-23 | V-3 | PENDING | No ICP tracker exists; "D435+ICP" is currently a CLI label for external poses |
| 2026-10-23 | V-4 | BLOCKED (pre-existing) | Open3D is not installed on this machine; `-DWITH_FUSION=ON` continues with a warning and the fallback, so no mesh has been produced |
| 2026-10-23 | V-5 | PENDING | Binary confidence mask only; no per-sample weighting |
| 2026-10-23 | V-6 | PENDING | No hand-held sequence captured; every capture so far is a static tripod shot |
| 2026-10-23 | V-7 | **CLOSED — RGB_PLUS_DEPTH (not DA3)** | Primary sources: HF API for `camera-depth-model-d435` lists only `README.md` + `cdm_d435.ckpt` (5.4 GB), CC BY-NC 4.0; `camera-depth-model-base` likewise `cdm_base.pth`, CC BY-NC 4.0. Upstream code `ByteDance-Seed/manip-as-in-sim-suite/cdm/infer.py` requires `--depth-image` and passes inverse depth to `model.infer_image`; `cdm/rgbddepth/dpt.py` splits `x[:, :3]` / `x[:, 3:]` (4-channel input). DA3 itself stays RGB-only (`Depth-Anything-3/src/depth_anything_3/api.py`: `(B, N, 3, H, W)`). Decision: do not wire depth into the DA3 path; `export_onnx.py --model-name <camera-id>` now fails fast with exit 2; README and `--help` corrected (CDM family, CC BY-NC 4.0, advisory `--camera`). |

## Out-of-scope confirmations

These are recorded as decisions, not deferrals, per the
[deferred revisit](../2026-10-13-da3-deferred-revisit/validation.md):

- 3DGS head: CONDITIONAL/`BLOCKED` — no Gaussian-head weights, gsplat build required.
- Long-sequence SLAM: DROPPED — external system; chunked streaming covers the C++ need.
- Video reconstruction: video input KEPT and shipped; video-to-3D stays out with the 3DGS head.
- Gradio/CLI parity: DROPPED — Python application layer.

## Attempt log

- **2026-10-23 — spec authoring.** Packet created from the roadmap review. No code changed.
- **2026-10-23 — prior evidence audit.** RealSense V-2..V-5 rows were re-verified live; the
  `output_metric.npy` filename in V-4 was wrong and was corrected to `output_metric_000.npy`.
