#!/usr/bin/env bash
# DA3 GPU gate driver (Colab-first, Brev-compatible).
# Usage: ./scripts/run_gpu_gate.sh [RESULTS_DIR]
# Env: MODEL (path to .onnx), IMAGE (path to .jpg/.png), PROCESS_RES (default 504)
set -euo pipefail

RESULTS="${1:-${RESULTS:-$HOME/gate-results}}"
MODEL="${MODEL:-}"
IMAGE="${IMAGE:-}"
PROCESS_RES="${PROCESS_RES:-504}"
mkdir -p "$RESULTS/logs"
STATUS="$RESULTS/STATUS"
: > "$STATUS"

log() { echo "[gate] $*" | tee -a "$RESULTS/gate.log"; }
pass() { echo "PASS $1" >> "$STATUS"; log "PASS $1"; }
fail() { echo "FAIL $1" >> "$STATUS"; log "FAIL $1"; }
unrun() { echo "UNRUN $1" >> "$STATUS"; log "UNRUN $1"; }

# V-0 env
{
  echo "== nvidia-smi =="; nvidia-smi 2>&1 || echo "no nvidia-smi"
  echo "== git =="; git rev-parse --short HEAD 2>&1 || echo "no git"
  echo "== cmake =="; cmake --version 2>&1 | head -1
  echo "== ORT/TRT pins =="; grep -E "ORT_VERSION|TRT_MAJOR|TRT_VERSION" CMakeLists.txt || true
} > "$RESULTS/environment.txt" 2>&1
log "env written to $RESULTS/environment.txt"
echo "PASS env" >> "$STATUS"

# V-1 ORT CPU build + unit tests
if cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release > "$RESULTS/logs/build.log" 2>&1 \
  && cmake --build build --parallel >> "$RESULTS/logs/build.log" 2>&1 \
  && ctest --test-dir build --output-on-failure -R UnitTests > "$RESULTS/logs/ctest.log" 2>&1; then
  pass "V-1 ort-cpu-build+UnitTests"
else
  fail "V-1 ort-cpu-build+UnitTests (see logs/build.log, logs/ctest.log)"
  unrun "V-2 infer (no binary)"
  unrun "V-5 parity (no outputs)"
  exit 0
fi

# V-2 smoke infer (needs pre-exported ONNX; export at home per runbook)
if [[ -z "$MODEL" || -z "$IMAGE" ]]; then
  unrun "V-2 infer (MODEL or IMAGE unset; export at home: python export_onnx.py --model-name da3-small --output da3-small.onnx)"
  unrun "V-5 parity (no outputs)"
  exit 0
fi

if ./build/inference_app "$MODEL" "$IMAGE" --conf --process-res "$PROCESS_RES" > "$RESULTS/logs/infer.log" 2>&1; then
  cp -f output_depth.jpg "$RESULTS/" 2>/dev/null || true
  cp -f output_conf.jpg "$RESULTS/" 2>/dev/null || true
  if [[ -f "$RESULTS/output_depth.jpg" ]]; then
    pass "V-2 infer depth+conf"
  else
    fail "V-2 infer (no output_depth.jpg)"
  fi
else
  fail "V-2 infer (see logs/infer.log)"
fi

# V-5 parity stub: single-backend run records depth range; cross-backend compare is Brev-full (T-6)
if grep -E "Depth range" "$RESULTS/logs/infer.log" >> "$RESULTS/gate.log" 2>&1; then
  unrun "V-5 parity (single run only; needs CPU vs CUDA vs TRT on Brev)"
else
  unrun "V-5 parity (no depth range logged)"
fi

log "done. STATUS:"; cat "$STATUS"
