# Plan — DA3 GPU Gate

Requirements: `requirements.md` · Proof: `validation.md` · Runbook: `gpu-validation-runbook.md`

## Group 1 — Runbook + harness (no GPU needed)
- [T-1] Add `gpu-validation-runbook.md` (Brev full + Colab partial, adapted from tritonic runbook + rf-detr rented-gpu-runbook) — Deliverable: this packet's runbook
  - Checks: `ls specs/2026-10-11-da3-gpu-gate/`
- [T-2] Add `scripts/run_gpu_gate.sh` + `scripts/colab/gpu_gate.ipynb` skeleton: env probe (`nvidia-smi`, driver/CUDA, ORT/TRT digests, commit), ORT-CPU build+test, ORT-CUDA build+infer, TRT build+infer, parity compare — Deliverable: scripts executable, `--help` works locally
  - Checks: `shellcheck scripts/run_gpu_gate.sh`

## Group 2 — Brev full gate [GPU]
- [T-3] Provision: `brev login`, `brev search --gpu-name L4/A10G --sort price`, `brev create da3-gate --type g5.xlarge --min-disk 100` (or L4 equivalent), wait `SHELL READY` — Deliverable: `evidence/<date>-brev-<gpu>/environment.txt`
- [T-4] ORT path: build default, `ctest -R UnitTests`, export `DA3-Small` ONNX (or copy pre-exported), `inference_app model.onnx image.jpg --conf` — Deliverable: `output_depth.jpg`, depth range log
- [T-5] TRT path: `cmake -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON`, build, `inference_app model.onnx` (parser builds engine) + `inference_app model.engine` rerun — Deliverable: engine + depth outputs
- [T-6] Parity + collect: CPU vs CUDA vs TRT depth `max/mean-abs`, `brev copy` results, `brev delete` (confirm) — Deliverable: `evidence/.../STATUS` PASS/FAIL per V-row

## Group 3 — Colab partial gate [GPU-cheap]
- [T-7] Notebook cells: setup (CUDA pin, OpenCV, CMake), build ORT-CPU, export Small, smoke infer, parity snippet — Deliverable: Drive dir per card (T4/L4) + copied logs
  - Checks: V-10..V-12 in validation.md

## Group 4 — Report
- [T-8] Write `comparison.md` snippet + update `validation.md` evidence log, roadmap status, CHANGELOG — Deliverable: packet closed, deviations noted

## Notes
- Additive only: no `src/` behavior change to pass gate; if code must change, replan in same branch.
- Orchestration: T-3..T-6 are maintainer-run worker packets, writable `~/da3-gate`, `evidence/**` only; acceptance `scripts/run_gpu_gate.sh` runs once; `brev delete/stop` needs explicit user confirm per brev-cli safety rules.
