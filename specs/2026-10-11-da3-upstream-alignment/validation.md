# Validation — DA3 Upstream Alignment

This document defines the checks for the [requirements](requirements.md) and records the evidence collected when they were run.

## Automated checks

Run these commands from the repository root:

```text
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R UnitTests
```

| ID | Check | Expected |
|---|---|---|
| V-1 | Run the focused unit tests for normalization, output sizing, preprocessing, colormaps, and mock-backend inference. | Exit code 0. |
| V-2 | Run the full regression suite with `ctest --test-dir build --output-on-failure`. | Exit code 0. |
| V-3 | Build the default ORT backend and confirm the TensorRT path is untouched. | Build succeeds. |
| V-4 | Run `clang-format-18 --dry-run` on touched files. | A documentation-only change should show no source diff. |

## Manual checks

| ID | Check | Expected |
|---|---|---|
| M-1 | Walk through <https://depth-anything-3.github.io/>. | The site loads, and the abilities and citation match the packet’s claims. |
| M-2 | Walk through `ByteDance-Seed/Depth-Anything-3`, including the Model Zoo and API. | The documented `depth/conf [N,H,W]`, `extrinsics [N,3,4]`, `intrinsics [N,3,3]`, and `use_ray_pose` match the R-6 deferral note. |
| M-3 | Open both Hugging Face collections. | `depth-anything/depth-anything-3` resolves with 12 items: 10 model checkpoints plus the demo space and the `DA3-BENCH` dataset. `depth-anything/camera-depth-models` resolves with 7 items. |
| M-4 | Review the README links, the Apache-2.0 versus CC BY-NC 4.0 split, and the metric formula. | Links resolve, the license split is clear, and the formula reads `focal * net / 300`. |

## Evidence log

| ID | Command or check | Result | Date | Notes |
|---|---|---|---|---|
| V-1 | `ctest -R UnitTests` | Pass | 2026-10-12 | Orchestrated run: 27/27 passed, including the new MetricScaling formula, zero-input, and large-focal tests. One repair cycle corrected the formula expectation to 100/300. |
| V-2 | Full `ctest` regression | Pass | 2026-10-12 | The single UnitTests suite is the full suite; exit code 0. |
| V-4 | `clang-format` on touched files | Pass with note | 2026-10-12 | The new helper lines were clean. The only violation was pre-existing at `processing_utils.cpp:82`. |
| M-1 | Site walkthrough | Pass | 2026-10-11 | Fetched the page with WebFetch and verified the teaser and abilities. |
| M-2 | GitHub walkthrough | Pass | 2026-10-11 | Verified the Model Zoo, API, and FAQ with WebFetch. |
| M-3 | Hugging Face collections | Pass | 2026-10-11; count re-verified 2026-10-19 | The 2026-10-11 WebFetch recorded 11 items. The Hugging Face API now returns 12 (10 model checkpoints, the demo space, and the `DA3-BENCH` dataset) (R-3). |
| M-4/T-2 | README synchronization | Pass | 2026-10-12 | Documentation-only diff. Added the project page, GitHub, `arXiv:2511.10647`, both Hugging Face collections, the Apache-2.0 versus CC BY-NC 4.0 split, the metric formula `focal * net / 300.`, the parity statement, and the any-view deferral note. All links resolved. |

## Deviations

- **Local GPU availability changed.** During the 2026-10-12 orchestration, the local host had an RTX 3060 Laptop with 6 GB, compute capability 8.6, driver 610.43, and CUDA user-mode driver 13.3. This contradicts the 2026-10-11 assumption in the GPU-gate spec that no local GPU was available. The ORT-CUDA smoke test remains unrun because no model or image weights were present locally. The Brev and Colab gates remain maintainer-run. The GPU-gate validation document was updated separately.
- **Reviewer notes accepted as follow-ups.** The README table of contents order does not match section placement, the “11 items” phrasing could be clearer, the Nested row sits under the monocular heading, and the demo-space and model-card links are named but not linked. These are not merge blockers.

## Definition of done

- Every automated and manual check has been executed and its evidence recorded.
- Evidence is traceable to the requirements, and deviations are documented.
- The spec, code, roadmap, and changelog agree, and the change merges as one coherent unit.
- Durable discoveries are propagated to the [tech stack](../tech-stack.md) and [mission](../mission.md).

## Orchestration handoff

This handoff follows the orchestrate-ai-coding-workflows convention.

- **Writable paths:** `README.md` and `specs/**` only.
- **Read-only paths:** `src/**`, `tests/**`, and the acceptance suite.
- **Permitted commands:** `git diff --stat`, `cmake --build`, one final `ctest -R UnitTests` acceptance run, and `clang-format --dry-run`.
- **Stop condition:** one acceptance run, report pass or fail, and do not silently self-repair. The planner decides the next step.
