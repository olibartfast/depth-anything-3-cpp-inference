# Requirements — DA3 Upstream Alignment

This packet makes the C++ repository traceable to upstream DA3 and locks the monocular parity scope before any any-view work begins. It is part of [Phase 1 of the roadmap](../roadmap.md) and builds on the [mission](../mission.md) and [tech stack](../tech-stack.md).

- **Spec:** `specs/2026-10-11-da3-upstream-alignment/`
- **Branch:** `spec/da3-upstream-alignment`

## Goal

Connect this repository to the upstream DA3 site, source code, and weight catalog, and define exactly what “monocular parity” means for this slice.

## In scope

- **Project constitution.** The constitution exists, is committed, and matches the repository: [mission](../mission.md), [tech stack](../tech-stack.md), and [roadmap](../roadmap.md). (R-1)
- **Documentation traceability.** The README links the official site at <https://depth-anything-3.github.io/>, the upstream code at `ByteDance-Seed/Depth-Anything-3`, the citation `arXiv:2511.10647`, the Hugging Face collection, the demo space, and the model cards. It also explains the preference for `-1.1` checkpoints. (R-2)
- **Weight catalog.** The documentation lists the models in the Hugging Face collection `depth-anything/depth-anything-3` — Nested-Giant-Large-1.1, Giant-1.1, Large-1.1, Base, Small, Metric-Large, Mono-Large, Nested-Giant-Large, Giant, and Large, plus the demo space and DA3-BENCH — with parameters and licenses where known. It also lists the camera-depth collection `depth-anything/camera-depth-models`: d405, d435, l515, kinect, zed2i-neural, zed2i-quality, and base. (R-3)
- **Monocular parity statement.** The packet documents the `export_onnx.py` wrapper (`unsqueeze(1)` → `model.model(..., "saddle_balanced")`, `[1,3,H,W]` → `[1,H,W]` plus optional confidence, opset 17, declared dynamic axes) and the C++ preprocessing (longest side 504, patch-14 alignment, ImageNet normalization). Both are checked against the current code; the camera-depth packet records the trace-shape limit of dynamic-axes exports. (R-4)
- **Metric handling.** The metric conversion `metric_depth = focal * net_output / 300` is documented, including the fact that Nested models already output meters. (R-5)
- **Any-view gap.** The deferral explicitly records the upstream I/O: depth and confidence `[N,H,W]`, extrinsics `[N,3,4]`, intrinsics `[N,3,3]`, `use_ray_pose`, `ref_view_strategy`, streaming below 12 GB, and glTF, NumPy, PLY, and 3DGS-video exports. (R-6)

## Out of scope

- Implementing multi-view, pose, 3DGS, streaming, video, or SLAM in C++. That work belongs to the Phase 3 spike.
- Downloading or mirroring weights, retraining models, or reproducing benchmarks.
- Changing backend selection. The binary still uses one compile-time backend.

## Decisions

- **Prefer `-1.1` checkpoints for Large, Giant, and Nested.** Upstream retrained these models after a training bug, and the retrained versions perform better on street scenes. (D-1)
- **Keep the monocular ONNX shape `[1,H,W]` as the C++ contract.** It matches the current backends and tests. N-view support needs a new packet. (D-2)
- **Surface the license split in the documentation.** Small, Base, Mono, Metric, and the camera-depth base are commercial-friendly under Apache-2.0. Large, Giant, and Nested are CC BY-NC 4.0. (D-3)

## Constraints

- The fixed section of the [tech stack](../tech-stack.md) applies. Height and width must be divisible by 14, and each binary has one backend.
- No weight binaries are committed.

## Dependencies

- **Codebase:** `export_onnx.py:91-119`, `src/processing_utils.cpp:31-68`, `src/depth_anything3_inference.cpp:83-147`, `CMakeLists.txt:41-52`, and `README.md:69-107`. These were verified by reading the files.
- **External sources:** the official site’s abilities and benchmark claims, the GitHub Model Zoo/API/FAQ, and the Hugging Face collections, fetched on 2026-10-11.

## Context from the upstream review

- **Architecture.** Upstream uses a single plain transformer (DINOv2) with a depth-ray representation. It reports state-of-the-art results over VGGT: +35.7% on pose and +23.6% on geometry. DA2 parity holds for monocular depth according to the site and GitHub.
- **API and CLI.** A prediction exposes `processed_images`, `depth`, `conf`, `extrinsics`, and `intrinsics`. The CLI provides `da3 auto`, `da3 video`, and `da3 backend`. “Nested” means an any-view Giant model with a metric scaler.
- **Current C++ scope.** The repository currently supports single-view depth and confidence only. It has no pose or intrinsics path.

## Assumptions and open questions

- **Canonical preprocessing (A-1).** Upstream `saddle_balanced` and the ImageNet statistics remain canonical. Basis: the current export script and README.
- **Camera-depth export flags (Q-1).** Do camera-depth models need separate export flags? Owner: maintainer. Resolve in Phase 2.

## Definition of done

- Every requirement (R-1 through R-6) is implemented or explicitly deferred with a tracked location.
- No in-scope behavior is silently dropped, and no out-of-scope work is smuggled in.
