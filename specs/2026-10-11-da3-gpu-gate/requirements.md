# Requirements — DA3 GPU Gate (Brev and Colab)

This packet proves that the C++ `inference_app` runs on real NVIDIA GPUs through the rented Brev service and the cheaper Colab path. It covers both the ORT CUDA execution provider (EP) and the TensorRT backend, and records evidence in the same style as the tritonic and rf-detr projects.

- **Spec:** `specs/2026-10-11-da3-gpu-gate/`
- **Roadmap:** Phase 1-GPU
- **Reference patterns:** tritonic `specs/2026-10-01-gpu-preprocess-alternatives/` for the runbook and evidence layout; rf-detr `specs/features/2026-10-08-colab-gpu-gate/validation.md` and `specs/rented-gpu-runbook.md`.

## Goal

Show that DA3 C++ inference works on a real NVIDIA GPU with both backends, and capture reproducible evidence.

## In scope

- **Brev full gate on one card.** Use an A10G `g5.xlarge` (about $1.21/hour, 4 vCPU, 24 GB GPU memory) or a cheaper L4 `g2-standard-4` (about $0.85/hour). Build both ORT and TensorRT, run `inference_app` on the exported ONNX model and the TensorRT engine, and record `environment.txt` and the outputs. (R-1)
- **ORT CUDA execution provider.** Use the same ONNX model as the CPU path, with providers `[CUDA, CPU]`. Log the fallback to CPU when no GPU is present. (R-2)
- **TensorRT path.** Convert ONNX to an engine with the bundled `libnvonnxparser`. `inference_app model.engine image.jpg` must produce `output_depth.jpg` and print a depth range. (R-3)
- **Colab partial gate.** On a free T4 or L4 runtime, without Docker, build with CMake, run `ctest -R UnitTests`, and smoke-test `inference_app` on a DA3-Small ONNX model. A notebook drives the run, and the results are copied into `evidence/`. (R-4)
- **Parity.** Compare GPU depth with CPU depth on the same image. Record the maximum absolute difference and require the mean to be at most `1e-3` after Issue #TBD. Both paths must use the same `compute_output_size` (504, patch 14) and ImageNet normalization. (R-5)
- **Runbook and evidence layout.** Mirror the tritonic pattern: `gpu-validation-runbook.md` and `evidence/<date>-brev-<gpu>/` containing `STATUS`, logs, and a `comparison.md` summary. (R-6)
- **Safety.** Do not run `brev delete`, `brev stop`, `--count>1`, or H100 instances without explicit confirmation. Show the cost before `brev create`. `brev login` is required; the host was logged out when this packet was written. (R-7)

## Out of scope

- Multi-view, pose, 3DGS, and streaming work (Phase 3 spike).
- DALI, Triton, and custom CUDA kernels. DA3 has none.
- CI GPU runners, batch sizes above one, video, PNG-only input, and multi-GPU runs.
- Mirroring weights in the repository, retraining, and DA3-BENCH reproduction.

## Decisions

- **Start with the Small model.** DA3-Small has about 90 million parameters and is Apache-2.0. It exports fastest, fits a T4 with 16 GB, and fits the L4 and A10G. Large and Giant are CC BY-NC and slower. (D-1)
- **Use Brev rather than a fixed provider.** The user asked for a Brev development flow like tritonic, and `brev search --sort price` finds the cheapest A10G or L4. (D-2)
- **Treat Colab as a supplement, not a replacement.** This follows the rf-detr pattern. Colab has no Docker and no full TensorRT pin; Brev proves the TensorRT parser and CUDA forward compatibility. (D-3)

## Constraints

- Height and width must be divisible by 14.
- One backend per binary: `USE_ONNX_RUNTIME` or `USE_TENSORRT`.
- TensorRT configuration needs the CUDA 13.3 toolkit installed; CMake locates it with `find_package(CUDAToolkit)`. The Colab gate pre-installs it.
- A TensorRT engine is fixed-shape. Export with `--fixed-size` and run a square image at the export resolution; the gate uses `test504.png`.
- TensorRT 11.2.1.2 and CUDA 13.3 match the `versions.env` pin in `rf-detr-cpp-inference`.
- The Brev host needs at least 100 GB of disk for ORT 1.21.0, TensorRT 11.2.1.2, OpenCV builds, and ONNX files.

## Dependencies

- **Codebase:** `src/backends/onnx_runtime_backend.cpp`, `tensorrt_backend.cpp`, `src/main.cpp`, `export_onnx.py:91-119`, and `CMakeLists.txt:41-52`.
- **External:** the `brev` CLI (login required), Docker and the NVIDIA toolkit on the Brev image, the Hugging Face `DA3-Small` weights, ORT 1.21.0, and TensorRT 11.2.1.2 with CUDA 13.3.

## Assumptions and open questions

- **Brev image compatibility (A-1).** The Brev Deep Learning AMI has driver R580 or newer and CUDA 13.x with forward compatibility for the ORT and TensorRT containers. Basis: the tritonic A10G run used driver 595.91.
- **Card choice (Q-1).** Which card funds this run — an A10G `g5.xlarge` or an L4 `g2-standard-4`? Owner: user. Confirm before `brev create`.

## Definition of done

- Every requirement is implemented or deferred with a location.
- No scope change is silent.
