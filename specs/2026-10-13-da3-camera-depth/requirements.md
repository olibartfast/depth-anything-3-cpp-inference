# Requirements — DA3 Camera-Depth + Export Shape-Lock (Phase A)

Spec: `specs/2026-10-13-da3-camera-depth/` · Branch: `spec/da3-camera-depth` (from `spec/da3-native-tasks`)

## Goal
Close roadmap Phase 2 and finding F-1: first-class camera presets + honest export shape-lock handling.

## In Scope
- [R-1] `--camera {d405,d435,l515,kinect,zed2i-neural,zed2i-quality,base}` on `inference_app`: selects recommended checkpoint name + metric mode (camera models output metric-ish depth; Nested `base` already meters → implies `--already-metric` semantics unless overridden). `--focal` still wins when explicitly passed (record `focal_set` already exists).
- [R-2] No fabricated focal numbers: presets carry the HF collection id + qualitative default guidance; exact fx/fy remain a user `--focal` input. Document per-camera notes (device, baseline use) in README table from HF `depth-anything/camera-depth-models` (d405, d435, l515, kinect, zed2i-neural, zed2i-quality, base).
- [R-3] F-1 export shape-lock: `export_onnx.py` warns at export time that dynamic-axes graphs are trace-shape-locked (RoPE broadcast fails off-shape, e.g. 252×504 vs 504×504 trace); C++ prints the traced-vs-requested shape hint when an ORT shape/broadcast error occurs is out of scope — instead document + add `--fixed-size` guidance in README/export `--help`.
- [R-4] Model-free unit tests: camera preset parse round-trip, invalid rejection, preset→checkpoint mapping, focal-override precedence (`focal_set` wins).
- [R-5] `export_onnx.py --model-name` accepts camera model ids (passes through to `depth-anything/<ID>` pretraining path with guidance).

## Out of Scope
- Downloading camera weights; per-device calibration files; changing metric formula.
- True N-view fusion (Phase B), Colab (Phase C), video/3DGS/BENCH (Phase D).

## Constraints
- C++20, one backend per binary, model-free `ctest -R UnitTests`, no invented intrinsics.

## Definition of Done
- [ ] R-1..R-5 implemented, validation.md V-1..V-5 green with evidence
