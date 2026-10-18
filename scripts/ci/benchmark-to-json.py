#!/usr/bin/env python3
"""Convert Google Benchmark JSON aggregates to github-action-benchmark input.

Google Benchmark emits aggregate rows named like `BM_Foo_median` when run with
`--benchmark_repetitions` and `--benchmark_report_aggregates_only`. The
benchmark action expects a flat customSmallerIsBetter array, so keep only the
median row per benchmark and drop the mean/stddev/cv series.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <google-benchmark.json> <action.json>", file=sys.stderr)
        return 2

    with open(sys.argv[1], encoding="utf-8") as source:
        data = json.load(source)

    benchmarks = data.get("benchmarks", [])
    rows = [
        row
        for row in benchmarks
        if row.get("run_type") == "aggregate" and row.get("aggregate_name") == "median"
    ]
    if not rows:
        # Fallback for a single-run file without repetitions.
        rows = [row for row in benchmarks if row.get("run_type") == "iteration"]

    output = []
    for row in rows:
        name = str(row.get("name", ""))
        if name.endswith("_median"):
            name = name[: -len("_median")]
        value = row.get("real_time")
        if not name or value is None or float(value) <= 0.0:
            continue
        output.append(
            {
                "name": name,
                "unit": str(row.get("time_unit", "ns")),
                "value": float(value),
            }
        )

    if not output:
        print("no benchmark results found in input", file=sys.stderr)
        return 2

    with open(sys.argv[2], "w", encoding="utf-8") as target:
        json.dump(output, target, indent=2)
        target.write("\n")

    print(f"wrote {len(output)} benchmark result(s) to {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
