# Plan — DA3 GPU Gate

This plan produces the [requirements](requirements.md) and records the proof in [validation.md](validation.md), following the [GPU validation runbook](gpu-validation-runbook.md).

## Group 1 — Runbook and harness (no GPU needed)

| ID | Task | Deliverable | Check |
|---|---|---|---|
| T-1 | Add `gpu-validation-runbook.md` for the Brev full gate and the Colab partial gate, adapted from the tritonic runbook and the rf-detr rented-GPU runbook. | This packet’s runbook. | `ls specs/2026-10-11-da3-gpu-gate/` |
| T-2 | Add `scripts/run_gpu_gate.sh` and a `scripts/colab/gpu_gate.ipynb` skeleton. The harness probes `nvidia-smi`, the driver, CUDA, the ORT and TensorRT versions, and the commit; builds and tests ORT-CPU; builds and runs ORT-CUDA; builds and runs TensorRT; and compares parity. | Executable scripts whose `--help` works locally. | `shellcheck scripts/run_gpu_gate.sh` |

## Group 2 — Brev full gate (GPU required)

| ID | Task | Deliverable |
|---|---|---|
| T-3 | Provision the host: `brev login`, `brev search --gpu-name L4/A10G --sort price`, `brev create da3-gate --type g5.xlarge --min-disk 100` (or the L4 equivalent), and wait for `SHELL READY`. | `evidence/<date>-brev-<gpu>/environment.txt` |
| T-4 | ORT path: build the default backend, run `ctest -R UnitTests`, export the DA3-Small ONNX model or copy a pre-exported one, and run `inference_app model.onnx image.jpg --conf`. | `output_depth.jpg` and the depth-range log |
| T-5 | TensorRT path: configure with `-DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON`, build, run `inference_app model.onnx` to let the parser build the engine, then rerun with `inference_app model.engine`. | Engine and depth outputs |
| T-6 | Parity and collection: compare CPU, CUDA, and TensorRT depth with max and mean absolute difference, copy the results back with `brev copy`, and delete the instance after confirmation. | `evidence/.../STATUS` with PASS or FAIL per validation row |

## Group 3 — Colab partial gate (cheap GPU)

| ID | Task | Deliverable | Check |
|---|---|---|---|
| T-7 | Notebook cells for setup (CUDA pin, OpenCV, CMake), ORT-CPU build, Small export, smoke inference, and a parity snippet. | A Drive directory per card (T4 or L4) with copied logs. | V-10 through V-12 in [validation.md](validation.md) |

## Group 4 — Report

| ID | Task | Deliverable |
|---|---|---|
| T-8 | Write the `comparison.md` summary, update the validation evidence log, update the roadmap status and CHANGELOG, and note any deviations. | A closed packet with deviations recorded |

## Notes

- This work is additive. No `src/` behavior changes are allowed merely to pass the gate. If code must change, replan on the same branch.
- T-3 through T-6 are maintainer-run worker packets. Writes are limited to `~/da3-gate` and `evidence/**`. The acceptance run uses `scripts/run_gpu_gate.sh` once. `brev delete` and `brev stop` require explicit user confirmation, following the Brev CLI safety rules.
