# Requirements — DA3 Upstream Alignment

Spec: `specs/2026-10-11-da3-upstream-alignment/requirements.md` · Branch: `spec/da3-upstream-alignment` · Roadmap: Phase 1

## Goal
Make this C++ repo traceable to upstream DA3 (site, code, weights) and lock monocular parity scope before any any-view work.

## In Scope
- [R-1] Constitution exists and matches repo: `specs/mission.md`, `tech-stack.md`, `roadmap.md`.
- [R-2] Docs traceability: README links official site `https://depth-anything-3.github.io/`, upstream code `ByteDance-Seed/Depth-Anything-3`, citation `arXiv:2511.10647`, demo space, model cards with `-1.1` preference note.
- [R-3] Weight catalog: HF `depth-anything/depth-anything-3` collection (Nested-Giant-Large-1.1, Giant-1.1, Large-1.1, Base, Small, Metric-Large, Mono-Large, Nested-Giant-Large, Giant, Large + space + DA3-BENCH) and HF `depth-anything/camera-depth-models` (d405, d435, l515, kinect, zed2i-neural, zed2i-quality, base) listed with params/licenses where known.
- [R-4] Monocular parity statement: `export_onnx.py` wrapper (`unsqueeze(1)` → `model.model(..., "saddle_balanced")`, `[1,3,H,W]`→`[1,H,W]`+`conf`, opset 17, dynamic axes) and C++ preprocess (longest-side 504, patch-14, ImageNet norm) verified against current code.
- [R-5] Metric handling documented: `metric_depth = focal * net_output / 300.`, Nested already meters.
- [R-6] Any-view gap explicitly deferred with upstream I/O noted (`depth/conf [N,H,W]`, `extrinsics [N,3,4]`, `intrinsics [N,3,3]`, `use_ray_pose`, `ref_view_strategy`, streaming <12GB, exports glb/npz/ply/3DGS-video).

## Out of Scope
- Implementing multi-view/pose/3DGS/streaming/video/SLAM in C++ — deferred to Phase 3 spike.
- Downloading/mirroring weights, retraining, benchmark reproduction.
- Changing backend selection (still one compile-time backend).

## Decisions
- [D-1] Prefer `-1.1` checkpoints for Large/Giant/Nested — Rationale: upstream retrained after training bug, better street scenes.
- [D-2] Keep monocular ONNX shape `[1,H,W]` as C++ contract — Rationale: matches current backends/tests; N-view needs new packet.
- [D-3] Apache-2.0 vs CC BY-NC 4.0 surfaced in docs — Rationale: Small/Base/Mono/Metric/camera-base are commercial-friendly; Large/Giant/Nested are not.

## Constraints
- Constitution `tech-stack.md` fixed section; H,W divisible by 14; one backend per binary.
- No weight binaries committed.

## Dependencies
- Codebase: `export_onnx.py:91-119`, `src/processing_utils.cpp:31-68`, `src/depth_anything3_inference.cpp:83-147`, `CMakeLists.txt:41-52`, `README.md:69-107` (verified by read).
- External: official site abilities/benchmark claims; GitHub Model Zoo/API/FAQ; HF collections (fetched 2026-10-11).

## Context
- Upstream finding: single plain transformer (DINOv2) + depth-ray representation; SOTA vs VGGT +35.7% pose / +23.6% geometry; DA2 parity for mono (site + GitHub).
- Upstream API finding: `prediction.{processed_images, depth, conf, extrinsics, intrinsics}`; CLI `da3 auto/video/backend`; nested = any-view giant + metric scaler.
- Repo inference: current C++ only single-view depth+conf; no pose/intrinsics path.

## Assumptions & Open Questions
- [A-1] Upstream `saddle_balanced` + ImageNet stats remain canonical — Basis: current export script + README.
- [Q-1] Do camera-depth models need separate export flags? Owner: maintainer — resolve in Phase 2.

## Definition of Done (requirements level)
- [ ] Every [R-n] is implemented or explicitly deferred with tracked location
- [ ] No In Scope behavior silently dropped; no Out of Scope work smuggled in
