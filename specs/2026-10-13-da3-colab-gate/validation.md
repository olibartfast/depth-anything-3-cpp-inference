# Validation — DA3 Colab GPU Gate (Phase C)

| ID | Requirement | Outcome |
|----|-------------|---------|
| R-1 / V-1 | Notebook authored | GREEN (static): `docs/colab-inference.ipynb`, 9 cells, `nbformat` 4.5, JSON-validated 2026-10-13. Covers env check → apt/pip → clone + build + `ctest -R UnitTests` → DA3-SMALL weights → mono (`--fixed-size`) + pose (`--dynamo`) exports → 2-view 280² prep → `--task mono` + `--task pose` + poses/intrinsics print. No credentials embedded. Execution PENDING-USER-RUN (Colab cannot run here). |
| R-2 / V-2 | Runbook + evidence path | GREEN: runbook Path A now references `docs/colab-inference.ipynb` (legacy `scripts/colab/gpu_gate.ipynb` kept as alternative). Evidence contract: user pastes back `nvidia-smi` header, `ctest` line, both export logs, both inference logs, `poses.txt`/`intrinsics.txt` into `specs/2026-10-11-da3-gpu-gate/evidence/<date>-colab-t4/`. |

## Status
PENDING-USER-RUN. The notebook pins `master` post-merge (see cells); run only after this stack lands or point the clone cell at the phase branch.
