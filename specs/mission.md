# Mission — Depth Anything 3 C++ Inference

## Problem
Python DA3 is research-first; C++/edge/robotics users need a small, backend-portable inference engine (ONNX Runtime / TensorRT + OpenCV) without Python at runtime.

## Audience
For: C++ developers shipping monocular depth on CPU/GPU, ROS/edge, offline batch tools.
Explicitly not for: training, model research, in-browser viewers, full any-view SLAM/3DGS pipelines (yet).

## Product Promise
Single-image depth + confidence from any DA3-monocular ONNX export with preprocessing parity to upstream Python, one compile-time backend, reproducible Docker/C++ tests. Any-view (multi-view, pose, metric-nested, streaming) is documented as a follow-on, not smuggled into this slice.

## Success Criteria
- Monocular ONNX from `export_onnx.py` runs in C++ and reproduces upstream preprocessing (longest-side + patch-14 + ImageNet norm).
- Upstream refs are traceable: official site, GitHub, HF DA3 collection, HF camera-depth-models collection, citation, licenses.
- `ctest -R UnitTests` passes with no model file.

## Product Principles
- Parity before expansion: match upstream observable behavior first.
- One backend at a time; deny-by-default paths in handoffs.
- Thin slices with runnable output; validation before code.

## Open Questions
- [Q-1] Any-view C++ API shape (N-view tensors, pose I/O) — owner: maintainer, resolve before Phase 3 spike.

_Revision: 2026-10-11 — initial constitution from official site + GitHub + HF collections._
