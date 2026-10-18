#!/usr/bin/env bash
# Run clang-tidy against the configured check set.
#
# Blocking policy (can be overridden for local experiments):
#   TIDY_WARNINGS_AS_ERRORS="clang-analyzer-*,bugprone-*"  (default)
#   STRICT_TIDY=true                                        (all checks)
#
# The TensorRT backend is excluded because the default CI configuration does
# not install TensorRT (USE_TENSORRT=OFF), so it has no compile command.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BUILD_DIR="${BUILD_DIR:-build}"
LOG="$BUILD_DIR/clang-tidy.log"
mkdir -p "$BUILD_DIR"

TIDY_BIN="${CLANG_TIDY_BINARY:-clang-tidy-18}"
if ! command -v "$TIDY_BIN" >/dev/null 2>&1; then
    echo "::error::clang-tidy binary not found: $TIDY_BIN (install clang-tidy-18)" >&2
    exit 2
fi
if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "::error::$BUILD_DIR/compile_commands.json is missing; configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" >&2
    exit 2
fi

warnings_as_errors="${TIDY_WARNINGS_AS_ERRORS:-clang-analyzer-*,bugprone-*}"
if [[ "${STRICT_TIDY:-false}" == "true" ]]; then
    warnings_as_errors="*"
fi

mapfile -t files < <(find src -name '*.cpp' ! -name 'tensorrt_backend.cpp' | sort)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "::error::no source files found for clang-tidy" >&2
    exit 2
fi

echo "Running clang-tidy on ${#files[@]} files (warnings-as-errors: $warnings_as_errors)..."
set +e
"$TIDY_BIN" -p "$BUILD_DIR" --quiet --warnings-as-errors="$warnings_as_errors" \
    "${files[@]}" > "$LOG" 2>&1
status=$?
set -e

# Keep the CI log readable: print only blocking findings here, and publish the
# complete configured-check report as an artifact.
if [[ "$warnings_as_errors" == "*" ]]; then
    cat "$LOG"
else
    grep -E '\[(clang-analyzer|bugprone)-' "$LOG" || true
fi

warnings=$(grep -cE ' warning: ' "$LOG" || true)
errors=$(grep -cE ' error: ' "$LOG" || true)

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### clang-tidy"
        echo ""
        echo "- Blocking checks: \`$warnings_as_errors\`"
        echo "- Findings: $warnings warning(s), $errors error(s)"
        if [[ "$warnings" -gt 0 ]]; then
            echo ""
            echo "| Check | Count |"
            echo "| --- | ---: |"
            grep -E ' warning: ' "$LOG" \
                | sed -E 's/.*\[([^]]+)\]$/\1/' \
                | sort | uniq -c | sort -rn \
                | awk '{printf "| `%s` | %s |\n", $2, $1}'
        fi
    } >> "$GITHUB_STEP_SUMMARY"
fi

if [[ $status -ne 0 ]]; then
    echo "::error::clang-tidy failed; blocking checks are: $warnings_as_errors. Report: $LOG" >&2
    exit "$status"
fi

echo "clang-tidy: clean for blocking checks ($warnings non-blocking warning(s); report: $LOG)."
