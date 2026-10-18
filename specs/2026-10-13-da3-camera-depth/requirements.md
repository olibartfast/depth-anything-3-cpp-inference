# Requirements — DA3 Camera-Depth and Export Shape-Lock

This packet closes Phase 2 of the [roadmap](../roadmap.md) and addresses finding F-1: it adds first-class camera presets and handles the export shape lock honestly.

- **Spec:** `specs/2026-10-13-da3-camera-depth/`
- **Branch:** `spec/da3-camera-depth`, branched from `spec/da3-native-tasks`.

## Goal

Add camera presets to `inference_app` and make the trace-shape limitation of dynamic-axes ONNX exports explicit in the export tooling and documentation.

## In scope

- **Camera presets.** Add `--camera {d405,d435,l515,kinect,zed2i-neural,zed2i-quality,base}` to `inference_app`. A preset selects the recommended checkpoint name and metric mode. Camera models are metric-oriented and output approximately metric depth. The `base` preset already outputs meters, so it implies `--already-metric` semantics; the other presets still apply the focal scaling. An explicit `--focal` still wins, and the existing `focal_set` flag records that precedence. (R-1)
- **No fabricated focal lengths.** Presets carry the Hugging Face collection identifier and qualitative default guidance. Exact `fx` and `fy` values remain a user-provided `--focal` input. Document each camera’s device and intended use in the README table, based on the `depth-anything/camera-depth-models` collection: d405, d435, l515, kinect, zed2i-neural, zed2i-quality, and base. (R-2)
- **Export shape lock.** `export_onnx.py` warns at export time that dynamic-axes graphs are locked to the shape used during tracing: a rotary position embedding (RoPE) broadcast fails off-shape, for example at 252×504 when the trace used 504×504. The C++ side will not add a traced-vs-requested shape hint for ORT shape errors in this packet. Instead, document the limitation and add `--fixed-size` guidance to the README and the export `--help`. (R-3)
- **Model-free tests.** Add unit tests for camera-preset parse round-trips, invalid-preset rejection, preset-to-checkpoint mapping, and focal-override precedence. (R-4)
- **Export model IDs.** `export_onnx.py --model-name` accepts camera model IDs and passes them through to the `depth-anything/<ID>` pretraining path with user guidance. (R-5)

## Out of scope

- Downloading camera weights and per-device calibration files.
- Changing the metric formula.
- True N-view fusion (the any-view packet), the Colab gate, and video, 3DGS, or DA3-BENCH work (the deferred-revisit packet).

## Constraints

- C++20, one backend per binary, and model-free `ctest -R UnitTests`.
- Do not invent intrinsics.

## Definition of done

- R-1 through R-5 are implemented.
- Checks V-1 through V-5 in [validation.md](validation.md) are green with evidence.
