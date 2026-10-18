# Validation — DA3 Any-View Fusion

This document records the outcome of the [requirements](requirements.md). The work ran on 2026-10-13 on an RTX 3060 Laptop with ORT CPU.

## Outcomes

| ID | Requirement | Outcome |
|---|---|---|
| V-1 | R-1 weight download | **GREEN.** `depth-anything/DA3-SMALL` came from the Hugging Face cache into `/tmp/da3-small-weights` as a 131 MiB safetensors file. Environment fixes were needed: `pip install pycolmap` for the upstream `api` import, a namespace-package fallback to `.api` in the exporter, and a local safetensors-directory fallback in `_load_model` (preset plus `strict=False`, six missing auxiliary keys). |
| V-2 | R-2 eager probe, export, and C++ batch | **GREEN.** An eager N-view forward produced `depth` and `depth_conf` at `[1,2,280,280]`, `extrinsics` at `[1,2,3,4]`, and `intrinsics` at `[1,2,3,3]`. The legacy tracer was **BLOCKED** because it does not support `aten::mT`; the `--dynamo` pose export succeeded (`/tmp/da3-small-anyview-pose.onnx`, 1.9 MB). ONNX Runtime validated it: depth and confidence `[2,280,280]`, extrinsics `[2,3,4]`, intrinsics `[2,3,3]`, all finite. The C++ N-view batch path (`run_inference_batch`, `postprocess_view`, and `get_pose_view`) was implemented with mock-backed unit tests. |
| V-3 | R-3 streaming chunks | **GREEN.** `--views-per-batch K` was added, with `K` defaulting to the traced N. Static models reject a mismatched N. Directories stream in trace-N chunks. |
| V-4 | R-4 PLY size | **GREEN.** `--ply-stride S` was added with a default of 1. A stride-4 reconstruction cloud is 373 KB; the native-tasks run measured roughly 76 MB for a full-resolution per-pixel ASCII cloud. |

## Detailed evidence

- `./build/inference_app /tmp/da3-small-anyview-pose.onnx /tmp/da3smoke/nv --task pose --process-res 280` reported “Batch model: 2 view(s) per batch at 280x280”. Chunk inference took 2.1 s. `poses.txt` contained a near-identity view 0 and a rotated and translated view 1 from the real model, with no fallback warning. `intrinsics.txt` contained properly decomposed `fx ≠ fy` values; the old code wrote `fy=0`, which was fixed in `get_pose_view`.
- `--task reconstruction --ply-stride 4` saved a 373 KB cloud.
- F-1 enforcement: `--process-res 504` on the 280-trace model fails with a clear error and fix guidance.
- `ctest -R UnitTests` passed, including the `BatchInference`, `BatchPoseView`, and `PlyStride` tests.

## Attempt log

- **R-1 download:** immediate from the Hugging Face cache.
- **R-2 legacy pose export:** **BLOCKED** by unsupported `aten::mT`; the dynamo path is **GREEN**.
- **Giant and Nested weights:** not attempted. They are CC BY-NC 4.0 and need more than 6 GB of VRAM at 504. Recorded rather than silently dropped.
- **glTF export:** **DROPPED.** It needs a renderer, and PLY is the interchange format. The decision is recorded in the keep-or-drop table above.
