# Validation — DA3 GPU Gate

Written 2026-10-11 before implementation. All `[GPU]` rows run on rented/Colab card; local host has no GPU.

## Automated Checks (Brev full gate)

```text
# on gate box, in tmux:
DEADLINE_HOURS=2 CUDA_ARCH=86 MODEL=~/da3-small.onnx IMAGE=~/test.jpg \
  ./scripts/run_gpu_gate.sh 2>&1 | tee ~/gate.log
```

- [ ] [V-0] Env identity: `nvidia-smi`, driver/CUDA, ORT 1.21.0 / TRT 11.2.1.2 digests, commit — expect recorded in `environment.txt`
- [ ] [V-1] ORT CPU build: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest -R UnitTests` — exit 0
- [ ] [V-2] ORT CUDA infer: `inference_app model.onnx image.jpg --conf` — `output_depth.jpg` + `output_conf.jpg` exist, depth range finite
- [ ] [V-3] TRT build: `cmake -S . -B build-trt -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON` + build — exit 0
- [ ] [V-4] TRT infer: `inference_app model.onnx image.jpg` then `inference_app model.engine image.jpg` — both produce depth, second reuses engine
- [ ] [V-5] Parity CPU vs CUDA vs TRT: same image, `max/mean-abs` recorded — gate `mean ≤ 1e-3`, max recorded (decoder-free, so tighter than tritonic D-5)
- [ ] [V-6] Sanitizers CPU-only locally: `cmake -DSANITIZERS=ON` + `ctest` — no new failures (GPU memcheck deferred, no custom kernel)

## Colab partial gate
- [ ] [V-10] T4 smoke: notebook builds ORT-CPU, exports Small, `inference_app` runs — PASS recorded with card sm_75/driver/CUDA
- [ ] [V-11] L4 smoke (optional, same as V-10 on sm_89)
- [ ] [V-12] Colab limits noted: no Docker, no TRT parser pin, `compute-sanitizer` may be UNRUN under forward-compat

## Manual Checks
- [ ] [M-1] View `output_depth.jpg` next to Python reference colormap — visually plausible (near/far correct)
- [ ] [M-2] `brev ls` shows instance deleted/stopped after run — no lingering billing

## Evidence Log
| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | Local ORT-CPU build+UnitTests | pass | 2026-10-11 | user run: `scripts/run_gpu_gate.sh` → PASS V-1, V-2/V-5 UNRUN (no MODEL/IMAGE) |
| V-2/V-5 | Local infer+parity | pass (square-504) | 2026-10-12 | RTX 3060: inference_app on /tmp/da3-small.onnx + /tmp/test504.png → depth [0.546271,1.50737] + conf; eager-vs-ONNX max-abs 5.662e-06 mean-abs 4.057e-07 (gate mean<=1e-3 PASS). Evidence: evidence/2026-10-12-local-3060/. Fixed-shape caveat below. |
| V-3/V-4 | Brev L4 full gate (TRT 11.2.1.2) | pass | 2026-10-12 | `g2-standard-4` L4 sm_89 $0.85/h, driver 595.91, CUDA 13.3 toolkit + pre-staged TRT via `scripts/colab/setup_colab.sh`. `run_gpu_gate.sh` STATUS: PASS env, V-1, V-2, V-3, V-4 (onnx→engine), V-4 (engine-reuse). Ranges: ORT [0.417894,1.80673], TRT [0.417865,1.80692] (min Δ 2.9e-05, max Δ 1.9e-04). Cross-arch vs local 3060 TRT [0.417915,1.80673]: min Δ 5e-05. Evidence: evidence/2026-10-12-brev-l4/. Instance deleted after collect. |
| V-3/V-4 | Local TRT 11.2.1.2 build+infer (same pin as rfdetr) | pass | 2026-10-12 | RTX 3060 sm_86, driver 610.43 (CUDA 13.3 UMD), TENSORRT_ROOTDIR=/home/oli/dependencies/TensorRT-11.2.1.2, cudart/cublas13+cudnn9 from pip. `cmake -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON` + build 17/17 clean; fixed-size export (`--fixed-size`, 100MB, validated) → engine built+saved; infer depth [0.417915,1.80673]; `.engine` rerun identical (deserialize path). Negatives: non-square image → clear size-mismatch error; dynamic-axes ONNX → parser rejects (squeeze needs static shape), no engine written. Evidence: evidence/2026-10-12-local-3060-trt11/. |
| V-5 | TRT-vs-ORT parity (local) | pass (range-level) | 2026-10-12 | Same fixed ONNX + test504.png: ORT CPU [0.417894,1.80673] vs TRT [0.417915,1.80673] (min Δ 2.1e-05, max identical). Raw-tensor max/mean-abs still UNRUN (binaries save colorized JPG only) — Brev T-6. |
| V-5 | Numeric parity raw-vs-raw | pass | 2026-10-12 | `--dump-raw` .npy dumps + `scripts/compare_depth.py`: ORT vs TRT on test504.png → shapes (504,504), max-abs 1.701e-03, mean-abs 4.087e-05 ≤ 1e-3 PASS. Gate V-5 hook wired in `run_gpu_gate.sh` (compares only when IMAGE == TRT_IMAGE). |
| V-10 | Colab T4 | pending | — | notebook skeleton T-2 first |
| M-1/M-2 | manual | pending | — | — |
| — | Local pre-check | pass | 2026-10-11 | `nvidia-smi` absent → confirms need for rented GPU; `brev search` works logged-out (A10G g5.xlarge $1.21/h, L4 g2-standard-4 $0.85/h) |
| C-1 | Gate skeleton | pass | 2026-10-11 | `scripts/run_gpu_gate.sh` + `setup_colab.sh` bash -n + shellcheck clean; `gpu_gate.ipynb` JSON valid |
| C-2 | Skeleton re-audit (orchestrated) | pass w/ note | 2026-10-12 | bash -n clean on run_gpu_gate.sh/gpu_host_setup.sh/setup_colab.sh, ipynb JSON valid; shellcheck NOT installed locally (UNRUN); local GPU now present (see deviations) |

## Deviations
- Brev L4 gate 2026-10-12 lessons (all fixed in-repo before the passing run):
  - `brev create --type g2-standard-4` rejected — full TYPE string required (`g2-standard-4:nvidia-l4:1`).
  - "Ready" ≠ SSH-ready on GCP (SSH timeouts ~10 min post-create); `brev ls` BUILD=COMPLETED SHELL=READY is the true signal.
  - V-4 first run FAILED: `libnvinfer_builder_resource_sm89.so` not found — libnvinfer dlopens sm_XX resources at build time and RPATH doesn't cover it. Fix: `TENSORRT_ROOTDIR/lib` on LD_LIBRARY_PATH in `setup_colab.sh` env file + belt-and-braces export in `run_gpu_gate.sh`.
  - `setup_colab.sh` hardcodes `/content` (Colab-only); Brev needed `sudo mkdir -p /content` shim. Follow-up: parameterize `STAGE_DIR`.
  - Gate IMAGE must match the fixed engine shape: first launch used non-square `test.jpg` for ORT V-2 against the square fixed ONNX — relaunched with `test504.png` for both.
- Brev run 2026-10-11 CLOSED without pass (fail-fast pip chain, ~1h L4 $0.85/h wasted). Lessons, fix before next paid run:
  - Box python3.10: old pip (no `--break-system-packages`), no `pip3` until `apt install python3-pip`.
  - `pip install -e Depth-Anything-3` pulls `moviepy` → egg_info fails on py3.10/new setuptools. Fix: `--no-deps` + pre-install only export libs (torch, einops, huggingface_hub, numpy<2, opencv, pillow, omegaconf, safetensors).
  - Editable+hatchling needs `hatchling`+`editables` (or isolated build). Fix: bake into setup script, test free first.
- Strategy change (user): perfect setup script on free LAN GPU `192.168.0.212`, only then re-rent Brev.
- 2026-10-12 local export findings (pip torch 2.12.0+cu130, no NGC needed):
  - export_onnx.py fixes: wrapper 4th positional must be `[]` not None (upstream forward
    signature now `(x, extrinsics, intrinsics, export_feat_layers=[], ...)`); `dynamo=False`
    (dynamo export fails with SymInt error); cartesian_prod compat patch for RoPE grid.
  - env shims (import-time only, never called in mono path): moviepy 2.x + editor.py shim,
    /tmp/da3shim/pycolmap.py stub; pip installs: einops, omegaconf, moviepy, addict, plyfile, evo.
  - FIXED-SHAPE LIMITATION: traced ONNX bakes RoPE tables for the 504x504 trace grid;
    non-square input (280x504) fails in ORT (`Add` broadcast 721 vs 1297). Dynamic axes are
    declared but not honored — multi-shape export needs dynamo + upstream cache/int() rework.
    C++ ORT backend now accepts symbolic dims instead of throwing (per-call shapes).

## Definition of Done
- [ ] Every row executed with evidence; deviations documented; roadmap/changelog updated.
