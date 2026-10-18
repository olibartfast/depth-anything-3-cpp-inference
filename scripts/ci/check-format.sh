#!/usr/bin/env bash
# clang-format gate.
#
#   check-format.sh <base> [<head>]   changed lines between two commits
#   check-format.sh <base>            working tree vs <base>
#   check-format.sh --all             every tracked C++ source in src/ and tests/
#
# The tree is formatted, so CI checks the whole tree with --all. The
# changed-lines mode remains available for backports to branches with
# historical formatting drift.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

FORMAT_BIN="${CLANG_FORMAT_BINARY:-clang-format-18}"
GIT_FORMAT_BIN="${GIT_CLANG_FORMAT_BINARY:-git-clang-format-18}"
if ! command -v "$FORMAT_BIN" >/dev/null 2>&1; then
    echo "::error::clang-format binary not found: $FORMAT_BIN (install clang-format-18)" >&2
    exit 2
fi

if [[ "${1:-}" == "--all" ]]; then
    mapfile -t files < <(find src tests -type f \( -name '*.cpp' -o -name '*.hpp' \) | sort)
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No C++ sources found." >&2
        exit 2
    fi
    echo "Checking ${#files[@]} files against .clang-format..."
    if "$FORMAT_BIN" --dry-run --Werror "${files[@]}"; then
        echo "clang-format: clean (whole tree)."
        exit 0
    fi
    echo "::error::clang-format violations found. Run: $FORMAT_BIN -i <files>" >&2
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <base> [<head>] | --all" >&2
    exit 2
fi

base="$1"
head="${2:-}"
if ! git rev-parse --verify --quiet "$base" >/dev/null; then
    echo "::error::base revision not found: $base" >&2
    exit 2
fi

if ! command -v "$GIT_FORMAT_BIN" >/dev/null 2>&1; then
    echo "::error::$GIT_FORMAT_BIN not found (install clang-format-18)" >&2
    exit 2
fi

# Use the merge base so a PR is checked against three-dot semantics: only
# lines introduced by the PR, not reverse diffs from newer base-branch commits.
target="${head:-HEAD}"
if merge_base="$(git merge-base "$base" "$target" 2>/dev/null)"; then
    base="$merge_base"
fi

if [[ -n "$head" ]]; then
    if ! git rev-parse --verify --quiet "$head^{commit}" >/dev/null; then
        echo "::error::head revision not found: $head" >&2
        exit 2
    fi
    echo "Checking lines changed between $base and $head..."
    set +e
    output="$("$GIT_FORMAT_BIN" --diff --binary "$FORMAT_BIN" --style=file \
        --extensions cpp,hpp "$base" "$head" -- src tests 2>&1)"
    status=$?
    set -e
else
    echo "Checking lines changed between $base and the working tree..."
    set +e
    output="$("$GIT_FORMAT_BIN" --diff --binary "$FORMAT_BIN" --style=file \
        --extensions cpp,hpp "$base" -- src tests 2>&1)"
    status=$?
    set -e
fi

if [[ $status -ne 0 ]]; then
    printf '%s\n' "$output"
    echo "::error::clang-format violations in changed lines. Run 'git clang-format-18 $base' locally, or set format_scope=all to audit the tree." >&2
    exit 1
fi

echo "clang-format: clean (changed lines)."
