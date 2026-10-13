# Roadmap

Status values: `idea` · `planned` · `in progress` · `done` · `blocked` · `deferred`

## Phase 1 — Upstream alignment + monocular parity docs
- Status: in progress
- Outcome: constitution + one feature packet proving docs link official site, GitHub, both HF collections, and monocular parity path
- Proves: stack can stay truthful to upstream without scope creep
- Spec: `specs/2026-10-11-da3-upstream-alignment/` Branch: `spec/da3-upstream-alignment`

## Phase 2 — Camera-depth + metric handling
- Status: planned
- Outcome: metric formula + camera-specific models (D405/D435/L515/Kinect/ZED2I-neural/ZED2I-quality/base) documented, unit-tested scaling helper, README table updated
- Proves: metric users get correct real-world scale
- Spec: `specs/2026-10-11-da3-upstream-alignment/` (R-5) or follow-on slice

## Phase 3 — Any-view design spike (deferred)
- Status: in progress (thin front-end slice: `--task` CLI landed 2026-10-12 — mono/metric fully wired, multiview per-frame baseline loop, pose/reconstruction/gaussian standard-file outputs with graceful degradation; true N-view consistent fusion, streaming, glb/npz/3DGS-video still deferred)
- Outcome: design note for N-view tensors, extrinsics/intrinsics, `use_ray_pose`, `ref_view_strategy`, streaming, glb/ply export; no implementation
- Proves: whether C++ any-view is a thin next slice or new project
- Spec: future dated packet

## Phase 1-GPU — GPU gate (Brev full + Colab partial)
- Status: in progress
- Outcome: TRT 11.2.1.2 (build + onnx→engine + engine reuse) and ORT proven on RTX 3060 sm_86 and Brev L4 sm_89 with runbook + evidence (`evidence/2026-10-12-local-3060-trt11/`, `evidence/2026-10-12-brev-l4/`); Colab partial still pending (user-run notebook)
- Proves: rented-GPU pattern from tritonic/rf-detr transfers to DA3
- Spec: `specs/2026-10-11-da3-gpu-gate/` Branch: `spec/da3-gpu-gate`

## Deferred / Revisit
- [3DGS head, SLAM-Long, video reconstruction, Gradio/CLI parity — needs Phase 3 design + licensing review for Giant/Nested CC BY-NC 4.0]
- [Benchmark pipeline DA3-BENCH — research tooling, not C++ runtime]

_Revision: 2026-10-11 — initial thin slicing; replan after Phase 1 validation._
