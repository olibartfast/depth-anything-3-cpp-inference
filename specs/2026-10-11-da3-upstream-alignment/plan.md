# Plan — DA3 Upstream Alignment

## Group 1 — Constitution (done this branch)
- [T-1] Write `specs/mission.md`, `tech-stack.md`, `roadmap.md` — Deliverable: three short files matching templates
  - Checks: `ls specs && ls specs/2026-10-11-da3-upstream-alignment`

## Group 2 — Docs traceability (next, isolated)
- [T-2] README sync: official site link, GitHub link, citation, HF both collections, model table with `-1.1` + licenses + metric note — Deliverable: README diff only, no code change
  - Checks: `git diff --stat`, manual link click-through (V-2..V-4)

## Group 3 — Parity verification (read-only + tests)
- [T-3] Verify `export_onnx.py` vs `src/*` parity notes in packet — Deliverable: requirements R-4 evidence, no behavior change
  - Checks: `ctest --test-dir build --output-on-failure -R UnitTests`
- [T-4] Add metric-scaling unit test helper (Phase 2 thin slice) — Deliverable: `focal * x / 300` covered, Nested passthrough noted
  - Checks: focused gtest run

## Group 4 — Verification
- [T-5] Execute `validation.md` automated + manual checks, record evidence with dates
- [T-6] Update roadmap status + changelog/docs, report deviations

## Notes
- Keep groups independently reviewable; T-2 must not touch `src/`; T-4 must not change backend ABI.
- Orchestration: T-2/T-3 are cheap/local-tier worker packets with read-only acceptance suite, one acceptance run, deny-by-default writes (see handoff in validation notes).
