# Plan — DA3 Upstream Alignment

This plan delivers the [requirements](requirements.md) in four independently reviewable groups. The validation steps are in [validation.md](validation.md).

## Group 1 — Constitution

The constitution was completed on this branch.

| ID | Task | Deliverable | Check |
|---|---|---|---|
| T-1 | Write the mission, tech stack, and roadmap documents. | Three short files matching the project templates. | `ls specs && ls specs/2026-10-11-da3-upstream-alignment` |

## Group 2 — Documentation traceability

This group changes documentation only. It must not touch `src/`.

| ID | Task | Deliverable | Check |
|---|---|---|---|
| T-2 | Synchronize the README: official site, GitHub, citation, both Hugging Face collections, the model table with `-1.1` and licenses, and the metric note. | A README-only diff. | `git diff --stat`; manual link click-through for V-2 through V-4. |

## Group 3 — Parity verification

| ID | Task | Deliverable | Check |
|---|---|---|---|
| T-3 | Verify the parity notes between `export_onnx.py` and `src/*`, and record the evidence against R-4. | Requirements evidence; no behavior change. | `ctest --test-dir build --output-on-failure -R UnitTests` |
| T-4 | Add a metric-scaling unit test as the Phase 2 thin slice. | Coverage for `focal * x / 300`, including the Nested passthrough note. | Focused GoogleTest run. |

## Group 4 — Verification and close-out

| ID | Task |
|---|---|
| T-5 | Execute the automated and manual checks in [validation.md](validation.md) and record dated evidence. |
| T-6 | Update the roadmap status and changelog, and report any deviations. |

## Notes

- Keep the groups independently reviewable. T-2 must not touch `src/`, and T-4 must not change the backend ABI.
- Orchestration: T-2 and T-3 are small, local worker packets with a read-only acceptance suite, one acceptance run, and deny-by-default write access. The handoff details are in [validation.md](validation.md).
