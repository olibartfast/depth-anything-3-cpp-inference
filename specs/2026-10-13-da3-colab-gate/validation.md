# Validation — DA3 Colab GPU Gate

The outcome is recorded against the [requirements](requirements.md).

## Results

| ID | Requirement | Outcome |
|---|---|---|
| R-1 / V-1 | Notebook authored | **GREEN (static).** `docs/colab-inference.ipynb` has 9 cells and uses `nbformat` 4.5. It was JSON-validated on 2026-10-13. The notebook covers the environment check, APT and pip setup, clone and build with `ctest -R UnitTests`, DA3-SMALL weight download, mono (`--fixed-size`) and pose (`--dynamo`) exports, two-view 280² preparation, `--task mono` and `--task pose`, and printing `poses.txt` and `intrinsics.txt`. No credentials are embedded. Execution is `PENDING-USER-RUN` because Colab cannot run in this environment. |
| R-2 / V-2 | Runbook and evidence path | **GREEN.** The runbook’s Path A now references `docs/colab-inference.ipynb`; the legacy `scripts/colab/gpu_gate.ipynb` remains as an alternative. The evidence contract asks the user to paste back the `nvidia-smi` header, the `ctest` line, both export logs, both inference logs, and `poses.txt` and `intrinsics.txt` into `specs/2026-10-11-da3-gpu-gate/evidence/<date>-colab-t4/`. |

## Status

`PENDING-USER-RUN`. The notebook pins `master` after the merge. Run it only after this stack lands, or point the clone cell at the phase branch.
