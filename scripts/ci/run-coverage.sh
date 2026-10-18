#!/usr/bin/env bash
# Coverage gate: gcov line coverage plus diff coverage on changed lines.
#
# Gated surface: the library code exercised by the model-free unit tests.
# main.cpp and the ONNX Runtime backend are reported but not gated until the
# CLI and a real ONNX fixture are covered (tracked as follow-ups).
#
# Environment:
#   BASE_REF             git revision for diff coverage (default origin/master)
#   COVERAGE_MIN_LINES   minimum line coverage for the gated surface (default 80)
#   DIFF_COVERAGE_MIN    minimum coverage of changed lines (default 70)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build-coverage}"
BASE_REF="${BASE_REF:-origin/master}"
COVERAGE_MIN_LINES="${COVERAGE_MIN_LINES:-80}"
DIFF_COVERAGE_MIN="${DIFF_COVERAGE_MIN:-70}"

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing tool: $1${2:+ ($2)}" >&2
        exit 2
    fi
}

need cmake "install cmake"
need ctest "install cmake"
need g++ "install g++"
need gcovr "pip install gcovr"
need diff-cover "pip install diff-cover"

if ! git rev-parse --verify --quiet "$BASE_REF^{commit}" >/dev/null; then
    if git rev-parse --verify --quiet HEAD^ >/dev/null; then
        BASE_REF="HEAD^"
    else
        BASE_REF="$(git rev-list --max-parents=0 HEAD)"
    fi
fi

echo "==> Configure coverage build ($BUILD_DIR)"
cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_FLAGS="--coverage -O0" \
    -DCMAKE_C_FLAGS="--coverage -O0" \
    -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
    -DBENCHMARKS=OFF \
    -DWERROR=ON

echo "==> Build"
cmake --build "$BUILD_DIR" --parallel

# Avoid stale counters inflating coverage on repeated local runs.
find "$BUILD_DIR" -name '*.gcda' -delete

echo "==> Unit tests"
ctest --test-dir "$BUILD_DIR" --output-on-failure

GATED_LOG="$BUILD_DIR/gcovr-gated.log"
FULL_LOG="$BUILD_DIR/gcovr-full.log"
DIFF_LOG="$BUILD_DIR/diff-cover.log"

echo "==> Line coverage (gated library surface)"
set +e
gcovr -r . \
    --filter 'src/depth_anything3_inference' \
    --filter 'src/processing_utils' \
    --filter 'src/backends/inference_backend' \
    --xml "$BUILD_DIR/coverage.xml" \
    --txt "$BUILD_DIR/coverage.txt" \
    --print-summary \
    --fail-under-line "$COVERAGE_MIN_LINES" >"$GATED_LOG" 2>&1
gated_status=$?
set -e
cat "$GATED_LOG"

echo "==> Full source coverage report (visibility only)"
gcovr -r . \
    --filter 'src/' \
    --exclude 'src/backends/tensorrt_backend.cpp' \
    --txt "$BUILD_DIR/coverage-full.txt" \
    --html-details "$BUILD_DIR/coverage-full.html" \
    --print-summary >"$FULL_LOG" 2>&1 || true
cat "$FULL_LOG"

echo "==> Diff coverage ($BASE_REF)"
set +e
diff-cover "$BUILD_DIR/coverage.xml" \
    --compare-branch "$BASE_REF" \
    --fail-under "$DIFF_COVERAGE_MIN" >"$DIFF_LOG" 2>&1
diff_status=$?
set -e
cat "$DIFF_LOG"

gated_summary="$(grep -E '^lines:' "$GATED_LOG" | head -1 || true)"
full_summary="$(grep -E '^lines:' "$FULL_LOG" | head -1 || true)"
diff_summary="$(grep -E '^Coverage:|No lines with coverage' "$DIFF_LOG" | tail -1 || true)"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### Coverage"
        echo ""
        echo "- Gated line coverage: ${gated_summary:-unknown} (minimum ${COVERAGE_MIN_LINES}%)"
        echo "- Full source line coverage: ${full_summary:-unknown}"
        echo "- Diff coverage vs \`$BASE_REF\`: ${diff_summary:-unknown} (minimum ${DIFF_COVERAGE_MIN}%)"
        echo "- Reports: \`coverage.xml\`, \`coverage.txt\`, \`coverage-full.txt\`, \`coverage-full.html\`"
    } >>"$GITHUB_STEP_SUMMARY"
fi

if [[ $gated_status -ne 0 || $diff_status -ne 0 ]]; then
    echo "::error::coverage gate failed (line floor ${COVERAGE_MIN_LINES}%, diff floor ${DIFF_COVERAGE_MIN}%)" >&2
    exit 1
fi

echo "coverage: passed (line floor ${COVERAGE_MIN_LINES}%, diff floor ${DIFF_COVERAGE_MIN}%)."
