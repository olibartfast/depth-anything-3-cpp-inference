# Validation — DA3 Upstream Alignment

## Automated Checks
Run from repository root:

```text
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R UnitTests
```

- [ ] [V-1] Focused unit tests pass (normalize, compute_output_size, preprocess, colormap, mock-backend inference) — command above, expect exit 0
- [ ] [V-2] Full regression `ctest --test-dir build --output-on-failure` passes
- [ ] [V-3] Build check passes for default ORT backend; TRT path untouched
- [ ] [V-4] Format/static: `clang-format-18 --dry-run` on touched files if any (docs-only T-2 should show no src diff)

## Manual Checks
- [ ] [M-1] Official site walkthrough: `https://depth-anything-3.github.io/` loads, abilities + citation visible, matches packet claims
- [ ] [M-2] Upstream code walkthrough: `ByteDance-Seed/Depth-Anything-3` Model Zoo + API (`depth/conf [N,H,W]`, `extrinsics [N,3,4]`, `intrinsics [N,3,3]`, `use_ray_pose`) matches R-6 deferral note
- [ ] [M-3] HF collections resolve: `depth-anything/depth-anything-3` (11 items incl. `-1.1`, space, DA3-BENCH) and `depth-anything/camera-depth-models` (7 items) match R-3 catalog
- [ ] [M-4] README links + license split (Apache-2.0 vs CC BY-NC 4.0) + metric formula `focal * net / 300` readable, no dead links

## Evidence Log
| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `ctest -R UnitTests` | pending | — | run after T-3 |
| M-1 | site walkthrough | pass | 2026-10-11 | fetched markdown via WebFetch, teaser + abilities verified |
| M-2 | GitHub walkthrough | pass | 2026-10-11 | Model Zoo/API/FAQ verified via WebFetch |
| M-3 | HF collections | pass | 2026-10-11 | DA3 (11) + camera-depth (7) verified via WebFetch |
| M-4 | README sync | pending | — | after T-2 diff |

## Deviations
- None yet; any unmet criterion must note why + tracking location here.

## Definition of Done (integration)
- [ ] Every automated and manual check executed, with evidence recorded
- [ ] Evidence traced to [R-n]; deviations documented
- [ ] Spec, code, roadmap, and changelog agree — merge as one coherent change
- [ ] Durable discoveries propagated to `tech-stack.md` / `mission.md`

## Orchestration handoff (per orchestrate-ai-coding-workflows)
- Worker writable: `README.md`, `specs/**` only; read-only: `src/**`, `tests/**`, acceptance suite.
- Permitted: `git diff --stat`, `cmake --build`, `ctest -R UnitTests` (once as final acceptance), `clang-format --dry-run`.
- Stop: one acceptance run, report pass/fail, no silent self-repair; planner decides next.
