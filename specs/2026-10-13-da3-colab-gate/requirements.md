# Requirements — DA3 Colab GPU Gate

This packet authors the user-runnable Colab notebook that closes the pending Colab portion of the GPU-gate runbook. The full GPU-gate context is in [the GPU-gate requirements](../2026-10-11-da3-gpu-gate/requirements.md).

- **Spec:** `specs/2026-10-13-da3-colab-gate/`
- **Branch:** `spec/da3-colab-gate`

## Goal

Produce a Colab notebook that installs the build dependencies, builds this repository, exports a small model, and runs the ORT inference path on a stock Colab GPU runtime. The TensorRT path is documented as best-effort because it needs an exact CUDA and TensorRT pin.

## In scope

- **Notebook.** Create `docs/colab-inference.ipynb` (or place it under `notebooks/`). It must install CMake and OpenCV, note the CUDA toolkit requirement for the TensorRT path, and support the ORT-only CPU or GPU path. It must clone the current branch, configure and build, download or export a Small ONNX model, run `--task mono` and `--task pose` on sample images, and display the outputs. The ORT path must work on a stock Colab GPU runtime. The TensorRT path is best-effort and must state its CUDA and TensorRT pin. (R-1)
- **Runbook update.** Update `specs/2026-10-11-da3-gpu-gate/gpu-validation-runbook.md` (or this packet’s validation document) to mark the notebook as authored. Execution evidence remains `PENDING-USER-RUN`, with exact steps for pasting results back. (R-2)

## Out of scope

- Executing the notebook here. That is impossible in this environment.
- Merging to `master`.

## Related deferred work

The video-input, 3DGS-head, and keep-or-drop work moved to [Deferred revisit](../2026-10-13-da3-deferred-revisit/requirements.md). That packet carries the deferred requirements (R-1 through R-3) and checks (V-1 through V-3) from the original combined draft, including the F-2 follow-up. The Colab checks remain here.

## Validation

The acceptance checks and their current status are in [validation.md](validation.md): V-1 covers notebook validity and cell review, and V-2 covers the runbook update.
