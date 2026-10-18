# Validation — DA3 GPU Gate

These checks were defined before implementation. GPU rows run on a rented or Colab card. The local host had no GPU when the packet was written; it gained an RTX 3060 on 2026-10-12, as recorded in the deviations.

## Automated checks (Brev full gate)

Run on the gate host inside `tmux`:

```bash
DEADLINE_HOURS=2 CUDA_ARCH=86 MODEL=~/da3-small.onnx IMAGE=~/test.jpg \
  ./scripts/run_gpu_gate.sh 2>&1 | tee ~/gate.log
```

| ID | Check | Pass condition |
|---|---|---|
| V-0 | Environment identity: `nvidia-smi`, driver and CUDA versions, ORT 1.21.0 and TensorRT 11.2.1.2 digests, and the commit. | Recorded in `environment.txt`. |
| V-1 | ORT CPU build: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest -R UnitTests` | Exit code 0. |
| V-2 | ORT CUDA inference: `inference_app model.onnx image.jpg --conf` | `output_depth.jpg` and `output_conf.jpg` exist, and the depth range is finite. |
| V-3 | TensorRT build: `cmake -S . -B build-trt -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON` plus build. | Exit code 0. |
| V-4 | TensorRT inference: `inference_app model.onnx image.jpg`, then `inference_app model.engine image.jpg`. | Both produce depth, and the second run reuses the engine. |
| V-5 | Parity across CPU, CUDA, and TensorRT on the same image. Record max and mean absolute difference. | Mean ≤ `1e-3`; max recorded. This is decoder-free, so it is tighter than the tritonic D-5 check. |
| V-6 | CPU-only sanitizers locally: `cmake -DSANITIZERS=ON` plus `ctest`. | No new failures. GPU memory checking is deferred because there is no custom kernel. |

## Colab partial gate

| ID | Check | Pass condition |
|---|---|---|
| V-10 | T4 smoke: the notebook builds ORT-CPU, exports Small, and runs `inference_app`. | PASS recorded with the card, `sm_75`, driver, and CUDA version. |
| V-11 | L4 smoke (optional, same as V-10 on `sm_89`). | PASS recorded with the card details. |
| V-12 | Colab limits noted: no Docker, no TensorRT parser pin, and `compute-sanitizer` may be `UNRUN` under forward compatibility. | Limits recorded as `UNRUN`, not `FAIL`. |

## Manual checks

| ID | Check | Pass condition |
|---|---|---|
| M-1 | View `output_depth.jpg` next to the Python reference colormap. | Visually plausible, with near and far correctly ordered. |
| M-2 | `brev ls` shows the instance deleted or stopped after the run. | No lingering billing. |

## Evidence log

| ID | Command or check | Result | Date | Notes |
|---|---|---|---|---|
| V-1 | Local ORT-CPU build and UnitTests | pass | 2026-10-11 | User run: `scripts/run_gpu_gate.sh` → PASS V-1; V-2 and V-5 `UNRUN` (no `MODEL` or `IMAGE`). |
| V-2/V-5 | Local inference and parity | pass (square 504) | 2026-10-12 | RTX 3060: `inference_app` on `/tmp/da3-small.onnx` and `/tmp/test504.png` → depth `[0.546271,1.50737]` plus confidence. Eager-vs-ONNX max-abs `5.662e-06`, mean-abs `4.057e-07` (gate mean ≤ `1e-3` PASS). Evidence: `evidence/2026-10-12-local-3060/`. Fixed-shape caveat below. |
| V-3/V-4 | Brev L4 full gate (TensorRT 11.2.1.2) | pass | 2026-10-12 | `g2-standard-4` L4 `sm_89` at $0.85/h, driver 595.91, CUDA 13.3 toolkit plus pre-staged TensorRT via `scripts/colab/setup_colab.sh`. `run_gpu_gate.sh` STATUS: PASS for env, V-1, V-2, V-3, V-4 (ONNX→engine), and V-4 (engine reuse). Ranges: ORT `[0.417894,1.80673]`, TensorRT `[0.417865,1.80692]` (min Δ `2.9e-05`, max Δ `1.9e-04`). Cross-architecture vs local 3060 TensorRT `[0.417915,1.80673]`: min Δ `5e-05`. Evidence: `evidence/2026-10-12-brev-l4/`. Instance deleted after collection. |
| V-3/V-4 | Local TensorRT 11.2.1.2 build and inference (same pin as rf-detr) | pass | 2026-10-12 | RTX 3060 `sm_86`, driver 610.43 (CUDA 13.3 UMD), `TENSORRT_ROOTDIR=/home/oli/dependencies/TensorRT-11.2.1.2`, `cudart`/`cublas13`/`cudnn9` from pip. `cmake -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON` plus build 17/17 clean. Fixed-size export (`--fixed-size`, 100 MB, validated) → engine built and saved; inference depth `[0.417915,1.80673]`; `.engine` rerun identical (deserialize path). Negatives: non-square image → clear size-mismatch error; dynamic-axes ONNX → parser rejects (squeeze needs static shape), no engine written. Evidence: `evidence/2026-10-12-local-3060-trt11/`. |
| V-5 | TensorRT-vs-ORT parity (local) | pass (range level) | 2026-10-12 | Same fixed ONNX and `test504.png`: ORT CPU `[0.417894,1.80673]` vs TensorRT `[0.417915,1.80673]` (min Δ `2.1e-05`, max identical). Raw-tensor max/mean-abs still `UNRUN` because the binaries save colorized JPG only; Brev T-6. |
| V-5 | Numeric parity, raw vs raw | pass | 2026-10-12 | `--dump-raw` `.npy` dumps plus `scripts/compare_depth.py`: ORT vs TensorRT on `test504.png` → shapes `(504,504)`, max-abs `1.701e-03`, mean-abs `4.087e-05` ≤ `1e-3` PASS. The gate V-5 hook is wired in `run_gpu_gate.sh` and compares only when `IMAGE == TRT_IMAGE`. |
| V-10 | Colab T4 | skipped for this gate; notebook pending user run | 2026-10-12 | TensorRT 11 was proven on `sm_86` (3060) and `sm_89` (L4) with numeric parity, so T4 `sm_75` would add architecture coverage only. The notebook was authored on 2026-10-13 and is `PENDING-USER-RUN`; see [the Colab gate validation](../2026-10-13-da3-colab-gate/validation.md). |
| M-1/M-2 | Manual | pending | — | — |
| — | Local pre-check | pass | 2026-10-11 | `nvidia-smi` absent → confirms the need for a rented GPU. `brev search` works logged out (A10G `g5.xlarge` $1.21/h, L4 `g2-standard-4` $0.85/h). |
| C-1 | Gate skeleton | pass | 2026-10-11 | `scripts/run_gpu_gate.sh` and `setup_colab.sh` pass `bash -n` and shellcheck; `gpu_gate.ipynb` is valid JSON. |
| C-2 | Skeleton re-audit (orchestrated) | pass with note | 2026-10-12 | `bash -n` clean on `run_gpu_gate.sh`, `gpu_host_setup.sh`, and `setup_colab.sh`; `gpu_gate.ipynb` is valid JSON. shellcheck is not installed locally (`UNRUN`). A local GPU is now present (see deviations). |

## Deviations

### 2026-10-15 Brev L4 re-run aborted by user decision

`da3-gate` (`g2-standard-4:nvidia-l4:1`, $0.85/h) rebooted three times in about two hours due to spot preemption (`last reboot` 14:34, 15:40, +1), which killed the gate before V-1. The image also ships no toolchain (`cmake: command not found`), so the gate could not build without APT installs. Zero results, about $1.70 spent. The instance was deleted. Current-master validation was substituted locally on the RTX 3060 (below). L4 `sm_89` architecture coverage remains `UNRUN`.

### Local re-validation on 2026-10-15 (RTX 3060, ORT CPU)

The build was clean and `ctest -R UnitTests` passed. `--task pose` on `/tmp/da3-small-anyview-pose.onnx` (two square views at 280) produced real poses: view 0 near identity, view 1 rotated and translated, no fallback warning; intrinsics had `fx ≠ fy` from K decomposition. `--task reconstruction --ply-stride 4` wrote `output_cloud.ply`. `--task metric` on the dynamic mono model at 504 produced a finite range and a metric `.npy`. Non-square inputs were correctly rejected with the F-1 shape-lock error.

### Brev L4 gate lessons (2026-10-12)

All were fixed in the repository before the passing run.

- `brev create --type g2-standard-4` was rejected. The full type string is required: `g2-standard-4:nvidia-l4:1`.
- “Ready” is not the same as SSH-ready on GCP; SSH timed out for about 10 minutes after creation. `brev ls` showing `BUILD=COMPLETED SHELL=READY` is the true signal.
- The first V-4 run failed because `libnvinfer_builder_resource_sm89.so` was not found. `libnvinfer` loads the `sm_XX` resource at build time, and RPATH does not cover it. Fix: put `TENSORRT_ROOTDIR/lib` on `LD_LIBRARY_PATH` in the `setup_colab.sh` env file, with a redundant export in `run_gpu_gate.sh`.
- `setup_colab.sh` hardcodes `/content`, which is Colab-only. Brev needed `sudo mkdir -p /content` as a shim. Follow-up: parameterize `STAGE_DIR`.
- The gate image must match the fixed engine shape. The first launch used non-square `test.jpg` for ORT V-2 against the square fixed ONNX; the rerun used `test504.png` for both.

### First Brev run, 2026-10-11 (closed without a pass)

The run failed fast in the pip chain and wasted about one hour on an L4 at $0.85/h. Lessons to fix before the next paid run:

- The box had Python 3.10 with an old pip (no `--break-system-packages`) and no `pip3` until `apt install python3-pip`.
- `pip install -e Depth-Anything-3` pulls `moviepy`, whose `egg_info` fails on Python 3.10 with new setuptools. Fix: `--no-deps` plus pre-installing only the export libraries (torch, einops, huggingface_hub, numpy<2, opencv, pillow, omegaconf, safetensors).
- Editable installs with hatchling need `hatchling` and `editables` (or an isolated build). Bake this into the setup script and test it for free first.

### Strategy change

The user decided to perfect the setup script on the free LAN GPU at `192.168.0.212`, then re-rent Brev.

### Local export findings, 2026-10-12

Using pip torch 2.12.0+cu130, no NGC container:

- `export_onnx.py` fixes: the wrapper’s fourth positional argument must be `[]`, not `None`, because the upstream forward signature is now `(x, extrinsics, intrinsics, export_feat_layers=[], ...)`. `dynamo=False`, because dynamo export fails with a SymInt error. A `cartesian_prod` compatibility patch is needed for the RoPE grid.
- Environment shims are import-time only and never called in the mono path: a moviepy 2.x + `editor.py` shim and a `/tmp/da3shim/pycolmap.py` stub. Pip installs: einops, omegaconf, moviepy, addict, plyfile, evo.
- **Fixed-shape limitation:** the traced ONNX bakes RoPE tables for the 504×504 trace grid. A non-square input (280×504) fails in ORT (`Add` broadcast 721 vs 1297). Dynamic axes are declared but not honored. Multi-shape export needs dynamo plus upstream cache and `int()` rework. The C++ ORT backend now accepts symbolic dims instead of throwing, using per-call shapes.

## Definition of done

- [ ] Every row is executed with evidence, deviations are documented, and the roadmap and changelog are updated.
