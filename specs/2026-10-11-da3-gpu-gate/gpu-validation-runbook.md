# GPU Validation Runbook — DA3 C++ Inference

Pattern from tritonic `specs/2026-10-01-gpu-preprocess-alternatives/gpu-validation-runbook.md` + rf-detr `specs/rented-gpu-runbook.md` + `2026-10-08-colab-gpu-gate/validation.md`. DA3 is simpler: no Triton/DALI/custom kernels — just ORT (CPU/CUDA) + TensorRT parser + `inference_app`.

| Path | Needs | Covers | Cost |
|---|---|---|---|
| A. Colab (partial) | T4/L4 runtime, no Docker | V-10..V-12, UnitTests + ORT-CPU smoke | free tier, ~20 min (High-RAM on) |
| B. Brev Docker/GPU host (full) | driver ≥R580, Docker+NVIDIA toolkit, ≥100GB disk | V-0..V-6 everything | A10G `g5.xlarge` $1.21/h or L4 `g2-standard-4` $0.85/h, ~1h |

CUDA_ARCH cheat-sheet (property of box): T4 `75`, A10G/A4000 `86`, L4/L40S `89`, A100 `80`.

## Path A — Colab (do first, cheap arch cover)

1. Export Small at home (free): `python export_onnx.py --model-name da3-small --output da3-small.onnx` (ORT smoke) + `python export_onnx.py --model-name da3-small --output da3-small-fixed.onnx --fixed-size` (TRT) + square `test504.png` (TRT engine is fixed-shape).
2. Open `scripts/colab/gpu_gate.ipynb` in Colab → GPU T4 → Run all. It installs CUDA 13.3 + pre-stages TRT 11.2.1.2, builds ORT-CPU, runs `ctest -R UnitTests`, runs `inference_app` (ORT on test.jpg, TRT build+infer on test504.png).
3. Copy `output_depth.jpg` + `output_depth_trt.jpg` + console depth-ranges + `nvidia-smi` header into `evidence/<date>-colab-t4/`. Disconnect runtime when done.
4. Limits (record as UNRUN not FAIL): `compute-sanitizer` under forward-compat may be UNRUN, pip OpenCV not NGC build.

## Path B — Brev full gate

### 1. At home (free)
```bash
ctest --test-dir build --output-on-failure -R UnitTests  # V-1 CPU baseline
python export_onnx.py --model-name da3-small --output /tmp/da3-small.onnx  # or reuse HF ONNX
ffmpeg -loop 1 -i test.jpg -t 4 -r 25 -pix_fmt yuv420p long.mp4  # optional soak asset
```

### 2. Provision (needs confirm)
```bash
brev login
brev search --gpu-name L4 --sort price     # cheapest today: g2-standard-4 $0.85/h
brev search --gpu-name A10G --sort price   # alt: g5.xlarge $1.21/h, 4 vCPU
# CONFIRM type + $/h with user first (safety rule), then:
brev create da3-gate --type g5.xlarge --min-disk 100
brev ls  # wait BUILD=COMPLETED SHELL=READY (~7-10 min)
```
Pick on cores not VRAM (build-bound). Budget ≥2GB RAM/core or parallel `ninja` OOMs. Default 10GB disk is too small — use `--min-disk 100`.

### 3. Copy + run (detached, tmux)
```bash
git ls-files --cached --others --exclude-standard | zip -q da3-gate.zip -@
brev copy da3-gate.zip da3-gate:~/
brev copy /tmp/da3-small.onnx da3-gate:~/
brev copy /tmp/da3-small-fixed.onnx da3-gate:~/
brev exec da3-gate "rm -rf ~/da3 && mkdir ~/da3 && cd ~/da3 && unzip -q ~/da3-gate.zip && cp ~/da3-small.onnx . && cp ~/da3-small-fixed.onnx ."
brev exec da3-gate "cd ~/da3 && tmux new -d -s gate 'MODEL=~/da3/da3-small.onnx IMAGE=~/da3/test.jpg TRT_MODEL=~/da3/da3-small-fixed.onnx TRT_IMAGE=~/da3/test504.png ./scripts/run_gpu_gate.sh ~/gate-results > ~/gate.log 2>&1'"
brev exec da3-gate "cat ~/gate-results/STATUS; tail -5 ~/gate.log"  # poll
```
`run_gpu_gate.sh` steps: `env` (V-0) → `ort-cpu` (V-1) → `ort-cuda` (V-2) → `trt-build` (V-3) → `trt-infer` (V-4) → `parity` (V-5) → `report`. Failing step logs FAIL but continues.

### 4. Collect then kill (confirm delete)
```bash
brev copy da3-gate:~/gate-results.tar.gz ./
tar xzf gate-results.tar.gz -C specs/2026-10-11-da3-gpu-gate/evidence/
# CONFIRM first:
brev delete da3-gate
```
Then fill `validation.md` evidence log + `evidence/<date>-brev-<gpu>/comparison.md`, update roadmap/CHANGELOG stating UNRUNs plainly.

## Pitfalls (from refs)
- `brev create` may print Ready while console still BUILDING — SSH only on SHELL READY.
- Plain background `brev exec … &` stays attached — use `tmux new -d` or `nohup … timeout`.
- `brev ls` vs `brev ls nodes` are separate namespaces; check both before assuming missing.
- Colab exports own `CUDA_VERSION` — unset before sourcing version pins; prepend `cuda-compat` only on pre-R580 drivers.
- TRT parser needs CUDA toolkit at configure time (auto-download); host needs only Docker + driver + `curl/python3`.
- Stopped ≠ deleted — storage still bills; confirm delete on console.
