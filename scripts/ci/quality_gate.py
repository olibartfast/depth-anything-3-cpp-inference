#!/usr/bin/env python3
"""Evaluate the CI quality gate from the workflow's `needs` context.

Every dependency must succeed. Skipped dependencies fail closed so a disabled
or accidentally-removed job cannot silently weaken the gate.
"""

from __future__ import annotations

import json
import os
import sys

ORDER = ["format", "build-test", "sanitizers", "clang-tidy", "cppcheck", "coverage", "benchmark"]


def main() -> int:
    raw = os.environ.get("NEEDS_JSON", "")
    try:
        needs = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"::error::invalid NEEDS_JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(needs, dict) or not needs:
        print("::error::NEEDS_JSON must be a non-empty object", file=sys.stderr)
        return 2

    jobs = sorted(needs, key=lambda name: (ORDER.index(name) if name in ORDER else len(ORDER), name))
    failed: list[str] = []
    rows: list[str] = []

    for job in jobs:
        result = str(needs[job].get("result", "unknown"))
        ok = result == "success"
        if not ok:
            failed.append(job)
        rows.append(f"| `{job}` | {'PASS' if ok else 'FAIL'} | `{result}` |")

    lines = [
        "## Code quality gate",
        "",
        "| Job | Gate | Result |",
        "| --- | --- | --- |",
        *rows,
        "",
    ]

    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if repo and run_id:
        lines.append(f"[Workflow run]({server}/{repo}/actions/runs/{run_id})")
        lines.append("")

    if failed:
        lines.append(f"**Gate failed:** {', '.join(failed)}")
    else:
        lines.append("**Gate passed:** all blocking jobs succeeded.")

    report = "\n".join(lines) + "\n"
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(report)

    if failed:
        print(f"::error::quality gate failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
