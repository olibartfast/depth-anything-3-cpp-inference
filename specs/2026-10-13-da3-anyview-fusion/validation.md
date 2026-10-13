# Validation — DA3 Any-View Fusion (Phase B)

| ID | Requirement | Outcome |
|----|-------------|---------|
| V-1 (R-1) | Weights | GREEN: `depth-anything/DA3-SMALL` from HF cache → `/tmp/da3-small-weights` (safetensors 131MiB). Env fixes needed: `pip install pycolmap` (upstream `api` import); upstream namespace-package quirk → exporter now tries `.api` fallback; local safetensors-dir fallback added (`_load_model`: preset + strict=False, 6 missing aux keys). |
| V-2 (R-2) | Eager probe + export + C++ batch | GREEN: eager N-view forward gives `depth/depth_conf [1,2,280,280]`, `extrinsics [1,2,3,4]`, `intrinsics [1,2,3,3]`. Legacy tracer BLOCKED (`aten::mT` unsupported) → `--dynamo` pose export OK (`/tmp/da3-small-anyview-pose.onnx` 1.9MB, ORT-validated: depth/conf `[2,280,280]`, extrinsics `[2,3,4]`, intrinsics `[2,3,3]`, all finite). C++ N-view batch path (`run_inference_batch` + `postprocess_view` + `get_pose_view`) implemented with mock-backed unit tests. |
| V-3 (R-3) | Streaming chunks | GREEN: `--views-per-batch K` (auto = trace N; mismatch errors on static models); dir streamed in trace-N chunks. |
| V-4 (R-4) | PLY size | GREEN: `--ply-stride S` (default 1); stride-4 recon cloud 373KB vs ~76MB-equivalent stride-1 scale. |

## Evidence (2026-10-13, RTX 3060 laptop, ORT CPU)
- `./build/inference_app /tmp/da3-small-anyview-pose.onnx /tmp/da3smoke/nv --task pose --process-res 280`: "Batch model: 2 view(s) per batch at 280x280", chunk infer 2.1s, `poses.txt` shows view0 ~identity + view1 rotated/translated (REAL model poses, no fallback warning), `intrinsics.txt` fx≠fy properly K-decomposed (old code wrote fy=0 — fixed in `get_pose_view`).
- `--task reconstruction --ply-stride 4`: cloud saved 373KB.
- F-1 enforcement: `--process-res 504` on the 280-trace model errors clearly with fix guidance.
- `ctest -R UnitTests` green (BatchInference, BatchPoseView, PlyStride tests).

## Attempt log
- R-1 download: instant (HF cache). R-2 legacy pose export: BLOCKED `aten::mT` → dynamo path GREEN. Giant/Nested: NOT attempted (CC BY-NC 4.0 + >6GB VRAM at 504 — recorded, not silent). glb export: DROPPED (needs renderer; PLY is the interchange format — decision, Phase D table).
