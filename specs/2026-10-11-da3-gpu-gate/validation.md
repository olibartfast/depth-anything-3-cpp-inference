# Validation — DA3 GPU Gate

Written 2026-10-11 before implementation. All `[GPU]` rows run on rented/Colab card; local host has no GPU.

## Automated Checks (Brev full gate)

```text
# on gate box, in tmux:
DEADLINE_HOURS=2 CUDA_ARCH=86 MODEL=~/da3-small.onnx IMAGE=~/test.jpg \
  ./scripts/run_gpu_gate.sh 2>&1 | tee ~/gate.log
```

- [ ] [V-0] Env identity: `nvidia-smi`, driver/CUDA, ORT 1.21.0 / TRT 10.x digests, commit — expect recorded in `environment.txt`
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
| V-2/V-5 | Local infer+parity | pending | — | needs MODEL+IMAGE; no torch locally → export on Colab per below |
| V-10 | Colab T4 | pending | — | notebook skeleton T-2 first |
| M-1/M-2 | manual | pending | — | — |
| — | Local pre-check | pass | 2026-10-11 | `nvidia-smi` absent → confirms need for rented GPU; `brev search` works logged-out (A10G g5.xlarge $1.21/h, L4 g2-standard-4 $0.85/h) |
| C-1 | Gate skeleton | pass | 2026-10-11 | `scripts/run_gpu_gate.sh` + `setup_colab.sh` bash -n + shellcheck clean; `gpu_gate.ipynb` JSON valid |

## Deviations
- Brev run 2026-10-11 CLOSED without pass (fail-fast pip chain, ~1h L4 $0.85/h wasted). Lessons, fix before next paid run:
  - Box python3.10: old pip (no `--break-system-packages`), no `pip3` until `apt install python3-pip`.
  - `pip install -e Depth-Anything-3` pulls `moviepy` → egg_info fails on py3.10/new setuptools. Fix: `--no-deps` + pre-install only export libs (torch, einops, huggingface_hub, numpy<2, opencv, pillow, omegaconf, safetensors).
  - Editable+hatchling needs `hatchling`+`editables` (or isolated build). Fix: bake into setup script, test free first.
- Strategy change (user): perfect setup script on free LAN GPU `192.168.0.212`, only then re-rent Brev.

## Definition of Done
- [ ] Every row executed with evidence; deviations documented; roadmap/changelog updated.
