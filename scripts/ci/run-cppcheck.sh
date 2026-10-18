#!/usr/bin/env bash
# Run cppcheck with the project's blocking severity set.
#
# style/information findings are intentionally out of scope: they are noisy on
# this codebase and are covered by clang-tidy and clang-format instead.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BUILD_DIR="${BUILD_DIR:-build}"
LOG="$BUILD_DIR/cppcheck.log"
mkdir -p "$BUILD_DIR"

if ! command -v cppcheck >/dev/null 2>&1; then
    echo "::error::cppcheck not found (install cppcheck)" >&2
    exit 2
fi

echo "Running cppcheck (warning, performance, portability)..."
set +e
cppcheck \
    --enable=warning,performance,portability \
    --std=c++20 \
    --inline-suppr \
    --error-exitcode=1 \
    --suppress=missingIncludeSystem \
    --suppress=unmatchedSuppression \
    --quiet \
    --template='{file}:{line}:{severity}:{id}:{message}' \
    -I src src/ 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

findings=$(grep -cE ':(error|warning|performance|portability):' "$LOG" || true)

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### Cppcheck"
        echo ""
        echo "- Scope: warning, performance, portability (plus cppcheck errors)"
        echo "- Findings: $findings"
        if [[ "$findings" -gt 0 ]]; then
            echo ""
            echo "| Check | Count |"
            echo "| --- | ---: |"
            grep -E ':(error|warning|performance|portability):' "$LOG" \
                | awk -F: '{print $4}' \
                | sort | uniq -c | sort -rn \
                | awk '{printf "| `%s` | %s |\n", $2, $1}'
        fi
    } >> "$GITHUB_STEP_SUMMARY"
fi

if [[ $status -ne 0 ]]; then
    echo "::error::cppcheck found blocking issues. Report: $LOG" >&2
    exit "$status"
fi

echo "cppcheck: clean ($findings finding(s); report: $LOG)."
