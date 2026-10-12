# Requirements — DA3 GPU Gate (Brev + Colab)

Spec: `specs/2026-10-11-da3-gpu-gate/requirements.md` · Roadmap: Phase 1-GPU
Refs: tritonic `specs/2026-10-01-gpu-preprocess-alternatives/` (runbook + evidence pattern), rf-detr `specs/features/2026-10-08-colab-gpu-gate/validation.md` + `specs/rented-gpu-runbook.md`

## Goal
Prove DA3 C++ `inference_app` runs on a real NVIDIA GPU via rented Brev (full gate) and Colab (cheap arch cover), for both ORT-CUDA and TensorRT backends, with evidence recorded like tritonic/rf-detr.

## In Scope
- [R-1] Brev full gate on one card: `g5.xlarge` A10G ($1.21/h, 4 vCPU, 24GB) or cheaper L4 `g2-standard-4` ($0.85/h) — build ORT + TRT, run `inference_app` on exported ONNX + TRT engine, record `environment.txt` + outputs.
- [R-2] ORT CUDA EP path: same ONNX as CPU, providers `[CUDA, CPU]`, fallback to CPU logged when no GPU.
- [R-3] TensorRT path: ONNX→engine via bundled `libnvonnxparser`, `inference_app model.engine image.jpg` produces `output_depth.jpg` + depth range.
- [R-4] Colab partial gate: T4/L4 free tier, no Docker — CMake build + `ctest -R UnitTests` + `inference_app` smoke on `DA3-Small` ONNX, notebook drives it, results copied to `evidence/`.
- [R-5] Parity: GPU depth vs CPU depth on same image within tolerance (max-abs recorded, gate `mean ≤ 1e-3` after Issue #TBD), same `compute_output_size` (504, patch-14) + ImageNet norm.
- [R-6] Runbook + evidence layout mirrors tritonic: `gpu-validation-runbook.md`, `evidence/<date>-brev-<gpu>/` with `STATUS`, logs, `comparison.md` snippet.
- [R-7] Safety: no `brev delete/stop` or `--count>1` or H100 without explicit confirm; cost shown before `brev create`; `brev login` required (current host is logged out).

## Out of Scope
- Multi-view/pose/3DGS/streaming (Phase 3 spike); DALI/Triton/custom CUDA kernels — DA3 has none.
- CI GPU runners; batch>1, video, PNG-only, multi-GPU.
- Mirroring weights in repo; retraining; DA3-BENCH reproduction.

## Decisions
- [D-1] Small model first (`DA3-Small` ~90M, Apache-2.0) — Rationale: fastest export + fits T4 16GB + L4/A10G; Large/Giant are CC BY-NC + slower.
- [D-2] Brev over fixed provider — Rationale: user asked `brev dev like tritonic`; `brev search --sort price` picks cheapest A10G/L4.
- [D-3] Colab supplements, not replaces Brev — Rationale: rf-detr pattern; Colab has no Docker/TRT full pin, Brev proves TRT parser + CUDA forward-compat.

## Constraints
- H,W %14; one backend per binary (`USE_ONNX_RUNTIME` vs `USE_TENSORRT`); CUDA 13.3 toolkit needed for TRT configure (auto-download at CMake time; Colab gate pre-installs it).
- TRT engine is fixed-shape: export with `--fixed-size` and run on a square image at the export resolution (gate uses `test504.png`).
- TRT 11.2.1.2 / CUDA 13.3 — same pin as rf-detr-cpp-inference `versions.env`.
- Disk ≥100GB on Brev (ORT 1.21.0 + TRT 11.2.1.2 + OpenCV builds + ONNX).

## Dependencies
- Codebase: `src/backends/onnx_runtime_backend.cpp`, `tensorrt_backend.cpp`, `src/main.cpp`, `export_onnx.py:91-119`, `CMakeLists.txt:41-52`.
- External: `brev` CLI (login needed), Docker + NVIDIA toolkit on Brev image, HF `DA3-Small` weights, pinned ORT 1.21.0 / TRT 11.2.1.2 (CUDA 13.3).

## Assumptions & Open Questions
- [A-1] Brev Deep Learning AMI has driver ≥R580 + CUDA 13.x with forward-compat for ORT/TRT containers — Basis: tritonic A10G run driver 595.91.
- [Q-1] Which card funds this run (A10G g5.xlarge vs L4 g2-standard-4)? Owner: user — must confirm before `brev create`.

## Definition of Done
- [ ] Every R-n implemented or deferred with location; no silent scope change.
