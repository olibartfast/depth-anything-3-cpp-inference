#!/usr/bin/env bash
# Colab setup: pinned toolchain first on PATH (Colab preinstalled CUDA is not the pin).
# Mirrors rf-detr setup_colab.sh handling: unset Colab CUDA_VERSION, install toolkit if needed.
set -euxo pipefail
unset CUDA_VERSION || true
sudo apt-get update
sudo apt-get install -y cmake ninja-build clang libopencv-dev tmux rsync
# Python export deps are home-side; Colab only needs the pre-exported ONNX (see notebook cell 1).
touch /content/SETUP_DONE
