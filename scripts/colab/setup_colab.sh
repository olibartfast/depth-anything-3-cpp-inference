#!/usr/bin/env bash
# setup_colab.sh — prepare a Google Colab GPU runtime for scripts/run_gpu_gate.sh.
#
# Mirrors rf-detr-cpp-inference scripts/colab/setup_colab.sh: Colab has no Docker,
# a preinstalled CUDA that is not the pinned one (13.3), and a runtime recycled
# without warning. This installs the pinned CUDA toolkit + TRT 11 prefix (with
# wget retries: the in-configure download is single-attempt) and writes the
# environment the gate must run with to ${ENV_FILE}, sourced before every gate
# command:
#
#   ./scripts/colab/setup_colab.sh
#   source /content/da3_gate_env.sh && ./scripts/run_gpu_gate.sh
#
# Driven by scripts/colab/gpu_gate.ipynb.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Colab's image exports its own CUDA_VERSION (e.g. 13.0.x). The pins below must
# win, so unset it first (same handling as rf-detr setup_colab.sh).
unset CUDA_VERSION || true

# Pins — same as rf-detr-cpp-inference versions.env.
TRT_VERSION="${TRT_VERSION:-11.2.1.2}"
TRT_SHORT_VERSION="${TRT_SHORT_VERSION:-11.2.1}"
CUDA_VERSION_PIN="${CUDA_VERSION_PIN:-13.3}"

ENV_FILE="${ENV_FILE:-/content/da3_gate_env.sh}"
STAGE_DIR="${STAGE_DIR:-/content}"

# CUDA_VERSION_PIN 13.3 -> apt suffix 13-3, install prefix /usr/local/cuda-13.3.
CUDA_DASHED="${CUDA_VERSION_PIN//./-}"
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION_PIN}"
# CUDA 13.x needs an R580+ driver. Older drivers can still run it on data-centre
# cards (T4, L4, A100, ...) through the forward-compatibility package.
CUDA_MIN_DRIVER=580

SUDO=""
[[ "$(id -u)" -ne 0 ]] && SUDO="sudo"

step() { echo; echo "=== $* ==="; }

step "GPU"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
gpu_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
echo "CUDA_ARCH=${gpu_arch}  driver major=${driver_major}  nproc=$(nproc)  RAM=$(free -g | awk '/Mem:/{print $2}')G"

step "CUDA ${CUDA_VERSION_PIN} toolkit from the NVIDIA apt repository"
# shellcheck source=/dev/null
. /etc/os-release
distro="ubuntu${VERSION_ID//./}"
keyring="$(mktemp -d)/cuda-keyring.deb"
wget -q -O "$keyring" \
    "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
$SUDO dpkg -i "$keyring"
$SUDO apt-get update -qq
packages=(
    "cuda-toolkit-${CUDA_DASHED}"
    cmake ninja-build clang libopencv-dev
    pkg-config zstd
)
compat_dir=""
if [[ "$driver_major" -lt "$CUDA_MIN_DRIVER" ]]; then
    echo "driver ${driver_major} < ${CUDA_MIN_DRIVER}: adding cuda-compat-${CUDA_DASHED}"
    packages+=("cuda-compat-${CUDA_DASHED}")
    compat_dir="${CUDA_HOME}/compat"
fi
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "${packages[@]}"

step "TensorRT ${TRT_VERSION}"
# cmake downloads this tarball at configure time, but its single attempt can
# time out on Colab. wget retries and resumes; the gate then uses the prefix
# through TENSORRT_ROOTDIR.
trt_tarball="TensorRT-Enterprise-${TRT_VERSION}-Linux-x86_64-cuda-${CUDA_VERSION_PIN}-Release-external.tar.zst"
TENSORRT_ROOTDIR="${STAGE_DIR}/TensorRT-${TRT_VERSION}"
if [[ ! -f "${TENSORRT_ROOTDIR}/include/NvInfer.h" ]]; then
    wget -c --tries=20 --timeout=60 -nv -O "${STAGE_DIR}/${trt_tarball}" \
        "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${TRT_SHORT_VERSION}/tars/${trt_tarball}"
    tar --zstd -xf "${STAGE_DIR}/${trt_tarball}" -C "${STAGE_DIR}"
    rm -f "${STAGE_DIR}/${trt_tarball}"
fi

step "environment -> ${ENV_FILE}"
{
    echo "# written by scripts/colab/setup_colab.sh on $(date -Is)"
    echo "export PATH=\"${CUDA_HOME}/bin:\$PATH\""
    echo "export CUDACXX=\"${CUDA_HOME}/bin/nvcc\""
    # Colab's own CUDA (/usr/local/cuda -> 12.x) sits on LD_LIBRARY_PATH; the
    # pinned toolkit, and compat when needed, must come first. The TensorRT lib
    # dir must also be present: libnvinfer dlopens its sm_XX builder-resource
    # libs at engine-build time, and RPATH alone does not cover that lookup.
    echo "export LD_LIBRARY_PATH=\"${TENSORRT_ROOTDIR}/lib${compat_dir:+:${compat_dir}}:${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}\""
    echo "export CUDA_ARCH=\"${gpu_arch}\""
    echo "export TENSORRT_ROOTDIR=\"${TENSORRT_ROOTDIR}\""
} > "$ENV_FILE"
cat "$ENV_FILE"

step "sanity check"
# shellcheck source=/dev/null
source "$ENV_FILE"
nvcc --version | tail -2
# A driver too old even for compat fails here, in seconds, rather than as a
# TensorRT engine-build error twenty minutes into the gate.
cat > /tmp/cuda_probe.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    cudaError_t err = cudaFree(nullptr);
    std::printf("driver API %d, runtime %d, context: %s\n", driver, runtime, cudaGetErrorString(err));
    return err == cudaSuccess ? 0 : 1;
}
EOF
nvcc -o /tmp/cuda_probe /tmp/cuda_probe.cu && /tmp/cuda_probe
echo "setup done"
