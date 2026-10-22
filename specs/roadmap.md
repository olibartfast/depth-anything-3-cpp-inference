# Roadmap

Status values: `idea`, `planned`, `in progress`, `done`, `blocked`, `deferred`.

## Phase 1 — Upstream alignment and monocular parity

- **Status:** In progress.
- **Outcome:** A project constitution and one feature packet that prove the documentation links the official site, GitHub repository, both Hugging Face collections, and the monocular parity path.
- **Why it matters:** It shows the stack can stay faithful to upstream without expanding scope.
- **Spec:** [Upstream alignment](2026-10-11-da3-upstream-alignment/requirements.md) · branch `spec/da3-upstream-alignment`.

## Phase 2 — Camera-depth models and metric handling

- **Status:** Done (2026-10-13).
- **Outcome:** The metric formula and the camera-specific models (D405, D435, L515, Kinect, ZED 2i neural, ZED 2i quality, and base) are documented. A unit-tested scaling helper exists and the README model table is current.
- **Why it matters:** Metric users get correct real-world scale.
- **Spec:** [Camera-depth and export shape-lock](2026-10-13-da3-camera-depth/requirements.md) · branch `spec/da3-camera-depth`. This supersedes the R-5 stub in the upstream-alignment packet.

## Phase 3 — Any-view design spike

- **Status:** In progress.
- **Delivered so far:** A thin front-end slice landed on 2026-10-12. The `--task` CLI supports mono and metric end to end, a per-frame multi-view baseline loop, and pose, reconstruction, and Gaussian standard-file outputs with graceful degradation.
- **Still deferred:** True N-view consistent fusion, streaming, and glTF binary (`.glb`), NumPy (`.npz`), and 3DGS-video exports.
- **Outcome:** A design note covering N-view tensors, extrinsics and intrinsics, `use_ray_pose`, `ref_view_strategy`, streaming, and glTF/PLY export. The design spike itself produces no implementation; the front-end slice above is separate follow-on work.
- **Why it matters:** It determines whether C++ any-view support is a thin next slice or a separate project.
- **Spec:** A future dated packet.

## Phase 4 — RealSense capture and metric reference

- **Status:** In progress.
- **Outcome:** `inference_app` captures color plus aligned depth from an Intel RealSense camera in C++ (`-DUSE_REALSENSE=ON`), writes exact intrinsics, and feeds the existing metric pipeline.
- **Why it matters:** The D435 provides real metric depth on the same frames, which lets the camera-depth checkpoints be checked against physical measurements without a Python runtime.
- **Spec:** [RealSense capture](2026-10-21-da3-realsense-capture/requirements.md) · landed on `master`.

## Phase 1-GPU — GPU gate (Brev full gate, Colab partial)

- **Status:** In progress.
- **Delivered so far:** TensorRT 11.2.1.2 (build, ONNX-to-engine conversion, and engine reuse) and ORT were proven on an RTX 3060 (sm_86) and a Brev L4 (sm_89). The runbook and evidence are in [the GPU gate packet](2026-10-11-da3-gpu-gate/requirements.md), under `evidence/2026-10-12-local-3060-trt11/` and `evidence/2026-10-12-brev-l4/`.
- **Pending:** The Colab partial gate. The notebook was authored on 2026-10-13 and is pending a user run.
- **Why it matters:** It shows the rented-GPU pattern from the tritonic and rf-detr projects transfers to DA3.
- **Spec:** [GPU gate](2026-10-11-da3-gpu-gate/requirements.md) · branch `spec/da3-gpu-gate`.

## Deferred and revisit work

These efforts keep unresolved items visible rather than silently dropping them. Every item ends either green or `BLOCKED` with a dated reason.

| Effort | Status | Scope |
|---|---|---|
| Any-view fusion | In progress | Small-model fusion attempt, streaming chunks, and the PLY size follow-up (F-2). See [requirements](2026-10-13-da3-anyview-fusion/requirements.md) · branch `spec/da3-anyview-fusion`. |
| Colab gate | In progress | Notebook authoring; execution is pending a user run. See [requirements](2026-10-13-da3-colab-gate/requirements.md). |
| Deferred revisit | In progress | Video input, the 3DGS-head verdict, and keep-or-drop decisions for SLAM, Gradio, and DA3-BENCH. See [requirements](2026-10-13-da3-deferred-revisit/requirements.md). |

The following remain out of scope until the Phase 3 design and a licensing review for the CC BY-NC 4.0 Giant and Nested checkpoints:

- 3DGS head, long-sequence SLAM, video reconstruction, and Gradio/CLI parity.
- The DA3-BENCH benchmark pipeline, which is research tooling rather than part of the C++ runtime.

_Revision: 2026-10-11 — initial thin slicing; replan after Phase 1 validation._
_Revision: 2026-10-21 — added Phase 4 (RealSense capture) after the D435 hardware arrived._
