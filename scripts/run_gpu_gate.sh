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
  echo "== ORT/TRT pins =="; grep -E "ORT_VERSION|TRT_VERSION|TRT_CUDA_VERSION" CMakeLists.txt || true
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

# V-3/V-4 TensorRT build + infer (needs CUDA 13.3 toolkit + TRT_MODEL: fixed-size
# ONNX, e.g. exported with `python export_onnx.py --fixed-size --process-res 504`,
# plus a square test image at that resolution — the engine is fixed-shape).
# TENSORRT_ROOTDIR passthrough supported for pre-staged prefixes (Colab gate).
TRT_MODEL="${TRT_MODEL:-}"
# libnvinfer dlopens its sm_XX builder-resource libs at engine-build time;
# RPATH does not cover that lookup, so the prefix lib dir must be on
# LD_LIBRARY_PATH (setup_colab.sh already does this; belt and braces here).
if [[ -n "${TENSORRT_ROOTDIR:-}" && -d "$TENSORRT_ROOTDIR/lib" ]]; then
  export LD_LIBRARY_PATH="$TENSORRT_ROOTDIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Square image at the fixed-export resolution (aspect-preserving preprocess maps
# a square image to exactly the square engine shape). Defaults to IMAGE.
TRT_IMAGE="${TRT_IMAGE:-$IMAGE}"
if [[ -z "$TRT_MODEL" ]]; then
  unrun "V-3 trt-build (TRT_MODEL unset; export fixed-size: python export_onnx.py --model-name da3-small --output model_fixed.onnx --fixed-size)"
  unrun "V-4 trt-infer (no TRT model)"
else
  if cmake -S . -B build-trt -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON > "$RESULTS/logs/build-trt.log" 2>&1 \
    && cmake --build build-trt --parallel >> "$RESULTS/logs/build-trt.log" 2>&1; then
    pass "V-3 trt-build"
    if ./build-trt/inference_app "$TRT_MODEL" "$TRT_IMAGE" --process-res "$PROCESS_RES" > "$RESULTS/logs/infer-trt.log" 2>&1; then
      cp -f output_depth.jpg "$RESULTS/output_depth_trt.jpg" 2>/dev/null || true
      if [[ -f "$RESULTS/output_depth_trt.jpg" ]]; then
        pass "V-4 trt-infer (onnx->engine)"
      else
        fail "V-4 trt-infer (no output_depth_trt.jpg)"
      fi
      # Engine reuse: second run must deserialize the cached .engine
      TRT_ENGINE="${TRT_MODEL%.onnx}.engine"
      if [[ -f "$TRT_ENGINE" ]] && ./build-trt/inference_app "$TRT_ENGINE" "$TRT_IMAGE" --process-res "$PROCESS_RES" > "$RESULTS/logs/infer-trt-engine.log" 2>&1; then
        pass "V-4 trt-engine-reuse"
      else
        fail "V-4 trt-engine-reuse (see logs/infer-trt-engine.log)"
      fi
    else
      fail "V-4 trt-infer (see logs/infer-trt.log)"
    fi
  else
    fail "V-3 trt-build (see logs/build-trt.log)"
    unrun "V-4 trt-infer (no trt binary)"
  fi
fi

# V-5 parity stub: single-backend run records depth range; cross-backend compare is Brev-full (T-6)
if grep -E "Depth range" "$RESULTS/logs/infer.log" >> "$RESULTS/gate.log" 2>&1; then
  unrun "V-5 parity (single run only; needs CPU vs CUDA vs TRT on Brev)"
else
  unrun "V-5 parity (no depth range logged)"
fi

log "done. STATUS:"; cat "$STATUS"
