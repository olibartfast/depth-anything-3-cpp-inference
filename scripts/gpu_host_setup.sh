#!/usr/bin/env bash
# First-try-clean GPU host setup for the DA3 gate (LAN or Brev).
# Lessons from Brev run 2026-10-11 baked in; see specs/2026-10-11-da3-gpu-gate/validation.md Deviations.
# Usage: ./scripts/gpu_host_setup.sh [RESULTS_DIR]   (idempotent, safe to re-run)
set -u
RESULTS="${1:-$HOME/gate-results}"
LOG="$HOME/setup-gate.log"
mkdir -p "$RESULTS"
: > "$LOG"

step() { echo "=== $* ===" | tee -a "$LOG"; }
fail() { echo "SETUP_FAIL: $*" | tee -a "$LOG"; exit 1; }

step "0. env"
nvidia-smi >> "$LOG" 2>&1 || fail "no NVIDIA driver (nvidia-smi failed)"
python3 --version >> "$LOG" 2>&1 || fail "no python3"

step "1. apt deps (needs sudo, no prompt if cached)"
sudo apt-get update -qq >> "$LOG" 2>&1
sudo apt-get install -y -qq cmake ninja-build clang libopencv-dev python3-pip git wget >> "$LOG" 2>&1 \
  || fail "apt install failed (see $LOG)"

step "2. torch stack (large download, minutes)"
python3 -m pip install --quiet torch torchvision xformers >> "$LOG" 2>&1 \
  || fail "torch install failed"

step "3. export-only python deps (NOT full requirements: moviepy breaks on py3.10, see Deviations)"
python3 -m pip install --quiet einops huggingface_hub "numpy<2" opencv-python pillow omegaconf safetensors >> "$LOG" 2>&1 \
  || fail "export deps failed"

step "4. Depth-Anything-3 source, no-deps install (hatchling auto via isolated build)"
[ -d ~/Depth-Anything-3 ] || git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git ~/Depth-Anything-3 >> "$LOG" 2>&1 \
  || fail "DA3 clone failed"
python3 -m pip install --quiet --no-deps -e ~/Depth-Anything-3 >> "$LOG" 2>&1 \
  || fail "DA3 install failed"
python3 -c "import depth_anything_3; print('da3 import ok')" >> "$LOG" 2>&1 \
  || fail "import depth_anything_3 failed"

step "5. export da3-small ONNX"
[ -f ~/da3-small.onnx ] || python3 ~/da3/export_onnx.py --model-name da3-small --output ~/da3-small.onnx --process-res 504 >> "$LOG" 2>&1 \
  || fail "export failed"
ls -la ~/da3-small.onnx | tee -a "$LOG"

step "6. test image"
[ -f ~/test.jpg ] || wget -q https://raw.githubusercontent.com/ByteDance-Seed/manip-as-in-sim-suite/main/cdm/example_data/color_12.png -O ~/test.jpg >> "$LOG" 2>&1 \
  || fail "test image download failed"

echo "SETUP_DONE: MODEL=$HOME/da3-small.onnx IMAGE=$HOME/test.jpg" | tee -a "$LOG"
echo "Next: MODEL=~/da3-small.onnx IMAGE=~/test.jpg ./scripts/run_gpu_gate.sh ~/gate-results"
