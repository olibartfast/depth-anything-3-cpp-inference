# 2026-10-12 GPU gate — backend comparison (Brev L4 vs local RTX 3060)

Model: `da3-small-fixed.onnx` (DA3-Small, traced 504x504, opset 17, `--fixed-size`). Image: `test504.png`.
Brev host: `g2-standard-4` NVIDIA L4 **sm_89**, driver 595.91.07, CUDA 13.3 toolkit, TRT 11.2.1.2 (pre-staged via `scripts/colab/setup_colab.sh`).
Local host: RTX 3060 Laptop **sm_86**, driver 610.43.02, CUDA UMD 13.3, TRT 11.2.1.2 (`TENSORRT_ROOTDIR=/home/oli/dependencies/TensorRT-11.2.1.2`).

## Depth ranges (min, max)

| Backend | Host | Depth range | Compared to | min Δ | max Δ |
|---|---|---|---|---|---|
| ORT 1.21.0 | Brev L4 | [0.417894, 1.80673] | — | — | — |
| TRT 11.2.1.2 | Brev L4 | [0.417865, 1.80692] | ORT on L4 | 2.9e-05 | 1.9e-04 |
| TRT 11.2.1.2 | RTX 3060 | [0.417915, 1.80673] | TRT on L4 (cross-arch) | 5.0e-05 | 1.9e-04 |

- ORT vs TRT on the same L4 host agree to ≤ 1.9e-04 (range-level).
- TRT cross-arch, L4 (sm_89) vs RTX 3060 (sm_86), agree to ≤ 1.9e-04 (range-level).
- Local ORT-CPU range [0.417894, 1.80673] is identical to the L4 ORT range.
- Range-level only: the binaries write colorized JPGs, so raw-tensor max/mean-abs is **UNRUN** (see below).

## Gate rows (PASS / FAIL / UNRUN)

| Row | Check | Result |
|---|---|---|
| V-0 | Env identity (`nvidia-smi`, driver/CUDA, ORT/TRT pins, commit) | **PASS** — both hosts; `environment.txt` written |
| V-1 | ORT-CPU build + `UnitTests` | **PASS** — both hosts (1/1 tests) |
| V-2 | ORT infer depth + conf | **PASS** — both hosts; depth range finite |
| V-3 | TRT build (`-DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON`) | **PASS** — both hosts |
| V-4 | TRT infer: onnx→engine, then `.engine` reuse | **PASS** — both hosts (engine built + saved + deserialized; reuse identical) |
| V-5 | Parity CPU vs CUDA vs TRT (gate mean ≤ 1e-3) | **UNRUN** for raw-tensor max/mean-abs — range-level check PASS locally (Δ ≤ 1.9e-04). Brev STATUS: `UNRUN V-5 parity (single run only)` |
| V-6 | Sanitizers CPU-only (`-DSANITIZERS=ON` + ctest) | **UNRUN** |
| V-10 | Colab T4 smoke | **PENDING** — user-run notebook (T-7) |
| V-11 | Colab L4 smoke (optional) | **UNRUN** |
| V-12 | Colab limits noted | **UNRUN** |
| M-1 | Visual check of `output_depth.jpg` | **PENDING** |
| M-2 | `brev ls` confirms instance deleted | **PENDING** in validation.md; instance deletion is recorded in the V-3/V-4 note |

## Cost & cleanup

- Host: L4 `g2-standard-4` at **~$0.85/h**. (A prior 2026-10-11 Brev run closed without a pass — fail-fast pip chain, ~1 h / ~$0.85 wasted; fixed before this run.)
- **Instance deleted** after results were collected — no lingering billing.

## Status

- Brev L4 STATUS: `PASS env, V-1, V-2, V-3, V-4 (onnx→engine), V-4 (engine reuse)`; `UNRUN V-5`.
- Local RTX 3060 STATUS: same set PASS; V-5 raw-tensor parity UNRUN; V-6 UNRUN.
- Evidence: `evidence/2026-10-12-brev-l4/`, `evidence/2026-10-12-local-3060-trt11/`.
