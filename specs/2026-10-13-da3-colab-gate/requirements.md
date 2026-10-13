# Requirements — DA3 Colab GPU Gate (Phase C)

Spec: `specs/2026-10-13-da3-colab-gate/` · Branch: `spec/da3-colab-gate`

## Goal
Author the user-runnable Colab notebook that closes the GPU-gate runbook's pending Colab-partial item.

## In Scope
- [R-1] `docs/colab-inference.ipynb` (or `notebooks/`): install (cmake, opencv, CUDA toolkit note for TRT path OR ORT-only CPU/GPU path), clone this repo at the current branch, configure + build, download a Small ONNX (or export), run `--task mono` + `--task pose` on sample images, display outputs. ORT path must work on stock Colab GPU runtime; TRT path documented as best-effort (needs matching CUDA/TRT pin).
- [R-2] Runbook update in `specs/2026-10-11-da3-gpu-gate/gpu-validation-runbook.md` (or validation.md here) marking notebook authored; execution evidence stays PENDING-USER-RUN with exact steps to paste back.

## Out of Scope
- Executing Colab here (impossible); merge to master.

## Validation
| ID | Check |
|----|-------|
| V-1 | `jupyter nbconvert --to notebook --execute` dry-run of non-GPU cells where possible, else `json` validity + cell review checklist |
| V-2 | Runbook updated with user-run instructions |

# Requirements — DA3 Deferred Revisit (Phase D)

Spec: `specs/2026-10-13-da3-deferred-revisit/` · Branch: `spec/da3-deferred-revisit`

## Goal
Leave nothing silently deferred: video input, F-2 follow-through (if not done in B), 3DGS-head/SLAM/BENCH verdicts.

## In Scope
- [R-1] Video input: `--input-video PATH --fps K` (ffmpeg extract to temp dir → reuse multiview dir loop) OR documented refusal with reason. Attempt ffmpeg-based; OpenCV `VideoCapture` needs no new dep (opencv videoio present? verify — `imgcodecs/highgui` only in CMake; if videoio missing, doc-only + blocked reason).
- [R-2] 3DGS-head verdict: attempt gsplat install + `--task gaussian` export probe; record WORKS or BLOCKED (weights/head availability) with reason.
- [R-3] SLAM-Long / Gradio parity / DA3-BENCH: explicit keep-or-drop decision per item with one-paragraph rationale in validation.md (a decision is an outcome, not a deferral).

## Validation
| ID | Check |
|----|-------|
| V-1 | Video path exercised on a sample mp4 (or refusal documented + tested at parse level) |
| V-2 | gsplat probe outcome recorded |
| V-3 | Keep/drop table filled |
