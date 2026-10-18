#!/usr/bin/env bash
# Run the same blocking checks as CI, locally.
#
#   scripts/ci/run_quality.sh              changed-lines format gate + build + tests + tidy + cppcheck
#   scripts/ci/run_quality.sh --format-all audit the whole tree with clang-format
#   scripts/ci/run_quality.sh --skip-tests skip ctest (build is still required for clang-tidy)
#
# Format comparison defaults to origin/master. Override with BASE_REF=<rev>.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build-ci}"
BASE_REF="${BASE_REF:-origin/master}"
FORMAT_ALL=false
SKIP_TESTS=false

for arg in "$@"; do
    case "$arg" in
        --format-all) FORMAT_ALL=true ;;
        --skip-tests) SKIP_TESTS=true ;;
        -h|--help)
            sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

stage() {
    printf '\n==> %s\n' "$*"
}

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing tool: $1${2:+ ($2)}" >&2
        exit 2
    fi
}

need cmake "install cmake"
need ctest "install cmake"
need clang-format-18 "install clang-format-18"
need clang-tidy-18 "install clang-tidy-18"
need cppcheck "install cppcheck"
need clang++-18 "install clang-18"

if ! git rev-parse --verify --quiet "$BASE_REF^{commit}" >/dev/null; then
    if git rev-parse --verify --quiet HEAD^ >/dev/null; then
        BASE_REF="HEAD^"
    else
        BASE_REF="$(git hash-object -t tree /dev/null)"
    fi
fi

stage "Configure ($BUILD_DIR)"
cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DWERROR=ON \
    -DBENCHMARKS=ON \
    -DCMAKE_C_COMPILER=clang-18 \
    -DCMAKE_CXX_COMPILER=clang++-18 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

stage "Build"
cmake --build "$BUILD_DIR" --parallel

if [[ "$SKIP_TESTS" == false ]]; then
    stage "Unit tests"
    ctest --test-dir "$BUILD_DIR" --output-on-failure

    stage "Benchmark smoke test"
    "$BUILD_DIR/benchmarks" --benchmark_min_time=0.05s
fi

if [[ "$FORMAT_ALL" == true ]]; then
    stage "Format (whole tree)"
    ./scripts/ci/check-format.sh --all
else
    stage "Format (changed lines vs $BASE_REF)"
    ./scripts/ci/check-format.sh "$BASE_REF"
fi

stage "Clang-Tidy"
BUILD_DIR="$BUILD_DIR" ./scripts/ci/run-clang-tidy.sh

stage "Cppcheck"
BUILD_DIR="$BUILD_DIR" ./scripts/ci/run-cppcheck.sh

printf '\nQuality checks completed successfully.\n'
