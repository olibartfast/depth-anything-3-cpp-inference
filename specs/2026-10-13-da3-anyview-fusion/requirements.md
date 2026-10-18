# Requirements — DA3 Any-View Fusion

This packet tests whether the C++ engine can consume a true any-view model: spatially consistent N-view depth, not just a loop over independent frames. It is the follow-on to the native-tasks CLI.

- **Spec:** `specs/2026-10-13-da3-anyview-fusion/`
- **Branch:** `spec/da3-anyview-fusion`

## Goal

Attempt true spatially consistent N-view depth. Download the DA3-SMALL any-view weights (Apache-2.0, about 80 million parameters, fits in 6 GB), export them through `AnyViewWrapper` with `--task multiview` or `--task pose`, run the C++ N-view batch path, and compare the result with the per-frame baseline.

## In scope

- **Weight download.** Attempt to download `depth-anything/DA3-SMALL` from Hugging Face, installing the Python `depth_anything_3` package if needed. Record the exact outcome: success or blocked with a reason. (R-1)
- **Any-view export and C++ batch.** If R-1 succeeds, export with `export_onnx.py --task multiview` or `--task pose` at `--process-res 280` (14-divisible and small). Run `inference_app --task multiview` on a two-view directory. If the exported graph requires a stacked `[N,3,H,W]` input rather than the current per-frame `[1,3,H,W]` loop, extend the C++ path to feed an N-view batch and document which form the graph accepts. (R-2)
- **Streaming and chunking.** Process directories in `--views-per-batch K` chunks, defaulting to all views, so long sequences never hold N frames in RAM. Document the pattern for GPU memory below 12 GB. (R-3)
- **PLY size follow-up.** Address finding F-2 by adding binary PLY output or a `--ply-stride S` subsampling option for the cloud and Gaussian writers. The ASCII files reached 76 MB and 179 MB for two frames, which is not viable at scale. (R-4)

## Out of scope

These items are attempted only if practical, then recorded as blocked with a reason. Silent deferral is not acceptable.

- Giant and Nested weights. They are CC BY-NC 4.0 and need more than 6 GB of VRAM for inference at 504; record them as blocked with a reason if attempted.
- glTF binary (`.glb`) export. It needs a renderer; record the decision.

## Constraints

- No weight binaries are committed.
- Only Apache-2.0 models are downloaded: Small, Base, Mono, and Metric.
- C++20 and the model-free unit tests continue to pass.

## Definition of done

- Each requirement is either green with evidence or `BLOCKED` with a dated reason in [validation.md](validation.md).
