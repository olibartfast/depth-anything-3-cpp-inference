# GPU Validation Runbook — DA3 C++ Inference

This runbook follows the pattern from tritonic’s `specs/2026-10-01-gpu-preprocess-alternatives/gpu-validation-runbook.md`, rf-detr’s `specs/rented-gpu-runbook.md`, and `2026-10-08-colab-gpu-gate/validation.md`. DA3 is simpler: there is no Triton, DALI, or custom kernel — only ORT (CPU or CUDA), the TensorRT parser, and `inference_app`.

| Path | Requirements | Covers | Cost |
|---|---|---|---|
| A. Colab (partial) | T4 or L4 runtime, no Docker | V-10 through V-12, UnitTests, ORT-CPU smoke | Free tier, about 20 minutes with High-RAM enabled |
| B. Brev Docker/GPU host (full) | Driver ≥ R580, Docker and NVIDIA toolkit, ≥ 100 GB disk | V-0 through V-6 | A10G `g5.xlarge` at $1.21/h or L4 `g2-standard-4` at $0.85/h, about 1 hour |

CUDA architecture cheat sheet: T4 `75`, A10G/A4000 `86`, L4/L40S `89`, A100 `80`.

## Path A — Colab (run first; cheap architecture coverage)

1. Export Small at home, for free:
   ```bash
   python export_onnx.py --model-name da3-small --output da3-small.onnx
   python export_onnx.py --model-name da3-small --output da3-small-fixed.onnx --fixed-size
   ```
   Also prepare a square `test504.png`; the TensorRT engine is fixed-shape.
2. Open `docs/colab-inference.ipynb` in Colab with a GPU T4 runtime and choose **Run all**. The current notebook builds this repository, exports DA3-SMALL mono and any-view pose ONNX, runs `--task mono` and `--task pose`, and prints poses and intrinsics. The legacy alternative is `scripts/colab/gpu_gate.ipynb`, which pre-stages TensorRT 11.2.1.2 and runs the ORT smoke test.
3. Copy `output_depth.jpg`, `output_depth_trt.jpg`, the console depth ranges, and the `nvidia-smi` header into `evidence/<date>-colab-t4/`. For the current notebook, also paste back the `ctest` result, both export logs with output names, both inference logs, and `poses.txt` and `intrinsics.txt`. Disconnect the runtime when done.
4. Record these limits as `UNRUN`, not `FAIL`: `compute-sanitizer` under forward compatibility may be `UNRUN`, and pip OpenCV is not an NGC build.

## Path B — Brev full gate

### 1. At home (free)

```bash
ctest --test-dir build --output-on-failure -R UnitTests  # V-1 CPU baseline
python export_onnx.py --model-name da3-small --output /tmp/da3-small.onnx  # or reuse a Hugging Face ONNX
ffmpeg -loop 1 -i test.jpg -t 4 -r 25 -pix_fmt yuv420p long.mp4  # optional soak asset
```

### 2. Provision (confirm first)

```bash
brev login
brev search --gpu-name L4 --sort price     # cheapest today: g2-standard-4 at $0.85/h
brev search --gpu-name A10G --sort price   # alternative: g5.xlarge at $1.21/h, 4 vCPU

# Confirm the type and $/h with the user before creating anything.
brev create da3-gate --type g5.xlarge --min-disk 100
# Or, for L4 (the full type string is required):
# brev create da3-gate --type g2-standard-4:nvidia-l4:1 --min-disk 100

brev ls  # wait for BUILD=COMPLETED and SHELL=READY (about 7-10 minutes)
```

Choose on cores, not VRAM; this gate is build-bound. Budget at least 2 GB of RAM per core, or parallel `ninja` runs out of memory. The default 10 GB disk is too small, so use `--min-disk 100`.

### 3. Copy and run (detached, in tmux)

```bash
git ls-files --cached --others --exclude-standard | zip -q da3-gate.zip -@
brev copy da3-gate.zip da3-gate:~/
brev copy /tmp/da3-small.onnx da3-gate:~/
brev copy /tmp/da3-small-fixed.onnx da3-gate:~/
brev exec da3-gate "rm -rf ~/da3 && mkdir ~/da3 && cd ~/da3 && unzip -q ~/da3-gate.zip && cp ~/da3-small.onnx . && cp ~/da3-small-fixed.onnx ."
brev exec da3-gate "cd ~/da3 && tmux new -d -s gate 'MODEL=~/da3/da3-small.onnx IMAGE=~/da3/test.jpg TRT_MODEL=~/da3/da3-small-fixed.onnx TRT_IMAGE=~/da3/test504.png ./scripts/run_gpu_gate.sh ~/gate-results > ~/gate.log 2>&1'"
brev exec da3-gate "cat ~/gate-results/STATUS; tail -5 ~/gate.log"  # poll
```

`run_gpu_gate.sh` runs `env` (V-0) → `ort-cpu` (V-1) → `ort-cuda` (V-2) → `trt-build` (V-3) → `trt-infer` (V-4) → `parity` (V-5) → `report`. A failing step logs `FAIL` but continues.

### 4. Collect, then delete (confirm the delete)

```bash
brev copy da3-gate:~/gate-results.tar.gz ./
tar xzf gate-results.tar.gz -C specs/2026-10-11-da3-gpu-gate/evidence/

# Confirm with the user before running this:
brev delete da3-gate
```

Then fill the evidence log in [validation.md](validation.md) and `evidence/<date>-brev-<gpu>/comparison.md`, and update the roadmap and changelog, stating any `UNRUN` items plainly.

## Pitfalls

- `brev create` may print Ready while the console is still building. SSH works only when `SHELL=READY`.
- A plain background `brev exec … &` stays attached. Use `tmux new -d` or `nohup … timeout`.
- `brev ls` and `brev ls nodes` are separate namespaces. Check both before assuming a machine is missing.
- Colab exports its own `CUDA_VERSION`; unset it before sourcing version pins. Prepend `cuda-compat` only on pre-R580 drivers.
- The TensorRT parser needs the CUDA toolkit at configure time, and CMake locates it with `find_package(CUDAToolkit)`. The host needs Docker, the driver, the CUDA toolkit, and `curl`/`python3`.
- Stopped is not deleted; storage still bills. Confirm the delete on the console.
