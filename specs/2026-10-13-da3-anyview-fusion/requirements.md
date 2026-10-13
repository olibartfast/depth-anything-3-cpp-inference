# Requirements — DA3 Any-View Fusion (Phase B)

Spec: `specs/2026-10-13-da3-anyview-fusion/` · Branch: `spec/da3-anyview-fusion`

## Goal
Attempt true spatially-consistent N-view depth: download DA3-SMALL any-view weights (Apache-2.0, ~80M params, fits 6GB), export with `AnyViewWrapper` (`--task multiview/pose`), run C++ N-view batch, compare vs per-frame baseline.

## In Scope
- [R-1] Attempt HF download `depth-anything/DA3-SMALL` (+ python `depth_anything_3` install if missing); record exact outcome (success or blocked-with-reason).
- [R-2] If R-1 succeeds: `export_onnx.py --task multiview --process-res 280` (14-divisible, small) → run `inference_app --task multiview` on 2-view dir; extend C++ to feed an N-view batch input when the ONNX input rank/batch semantics require it (today C++ runs per-frame `[1,3,H,W]` loops — document whether the exported any-view graph accepts that or needs a stacked `[N,3,H,W]` input + new batch path).
- [R-3] Streaming/chunked pattern for <12GB: process dirs in `--views-per-batch K` chunks (default all) so long sequences never hold N frames in RAM; document.
- [R-4] F-2: binary PLY or `--ply-stride S` subsampling option for cloud/gaussian writers (76MB/179MB ASCII for 2 frames is untenable at scale).

## Out of Scope (attempt, then mark blocked honestly — no silent deferral)
- Giant/Nested weights (CC BY-NC 4.0 + >6GB VRAM for inference at 504 — record as blocked-with-reason if attempted).
- glb export (needs renderer; record decision).

## Constraints
- No weight binaries committed; Apache-2.0 only for downloads (Small/Base/Mono/Metric); C++20; model-free unit tests keep passing.

## Definition of Done
- [ ] Each R-n either green with evidence or BLOCKED with a dated reason in validation.md
