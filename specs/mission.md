# Mission — Depth Anything 3 C++ Inference

Depth Anything 3 (DA3) is a research-first Python project. This repository provides a small C++ inference engine for teams that need monocular depth estimation on CPUs, GPUs, edge devices, or robots without a Python runtime. The engine uses OpenCV for image I/O and supports exactly one compile-time inference backend: ONNX Runtime (ORT) or TensorRT (TRT).

## Who this is for

The primary audience is C++ developers shipping monocular depth on CPU or GPU, including ROS and edge deployments and offline batch tools.

This project is explicitly not for training, model research, in-browser viewers, or full any-view SLAM and 3D Gaussian Splatting (3DGS) pipelines.

## Product promise

The library produces single-image depth and confidence from any DA3 monocular ONNX export, with preprocessing that matches the upstream Python implementation. It offers one compile-time backend and reproducible Docker and C++ tests.

Any-view capabilities — multi-view input, pose estimation, metric-nested models, and streaming — are documented as a follow-on. They are not part of this slice.

## Success criteria

- A monocular ONNX model exported by `export_onnx.py` runs in C++ and reproduces upstream preprocessing: longest-side resize, patch-14 alignment, and ImageNet normalization.
- Upstream references remain traceable: the official site, GitHub repository, Hugging Face DA3 collection, Hugging Face camera-depth-models collection, citation, and licenses.
- `ctest -R UnitTests` passes without a downloaded model file.

## Principles

- **Parity before expansion.** Match upstream observable behavior before adding new capabilities.
- **One backend at a time.** Each binary selects ONNX Runtime or TensorRT at compile time.
- **Safe handoffs.** Default to read-only paths unless write access is explicitly granted.
- **Thin, runnable slices.** Prefer small changes with runnable output, and validate before writing code.

## Open questions

- **Any-view C++ API shape (Q-1).** The canonical representation for N-view tensors and pose I/O is still open. Owner: maintainer. Resolve before the Phase 3 spike.

_Revision: 2026-10-11 — initial constitution based on the official site, GitHub repository, and Hugging Face collections._
