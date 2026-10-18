# Requirements — DA3 Deferred Revisit

This packet closes out work that earlier phases left unresolved. Its purpose is to ensure nothing is silently deferred: each item ends with a shipped result, an explicit decision, or a `BLOCKED` status with a dated reason.

- **Spec:** `specs/2026-10-13-da3-deferred-revisit/`
- **Branch:** `spec/da3-deferred-revisit`

## Goal

Resolve the remaining deferred items: video input, the F-2 follow-through if the any-view packet did not complete it, and the verdicts for the 3DGS head, SLAM, Gradio, and DA3-BENCH.

## In scope

- **Video input.** Add `--input-video PATH --fps K`, extracting frames with ffmpeg into a temporary directory and reusing the multi-view directory loop, or document a refusal with a reason. The original plan preferred ffmpeg; OpenCV `VideoCapture` was the fallback and needs no new dependency. The validation records the path taken. Verify whether `videoio` is available, since CMake originally requested only `imgcodecs` and `highgui`. If `videoio` is missing, document the blocker and the reason. (R-1)
- **3DGS-head verdict.** Attempt to install gsplat and probe `--task gaussian` export. Record whether it `WORKS` or is `BLOCKED` by weights or head availability, with a reason. (R-2)
- **Keep-or-drop decisions.** Make an explicit decision for SLAM-Long, Gradio parity, and DA3-BENCH, with a one-paragraph rationale for each in [validation.md](validation.md). A decision is an outcome, not a deferral. (R-3)

## Validation

The checks are defined in [validation.md](validation.md):

- V-1: exercise the video path on a sample MP4, or document and test a refusal at parse level.
- V-2: record the gsplat probe outcome.
- V-3: fill the keep-or-drop table.
