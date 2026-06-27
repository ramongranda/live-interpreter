#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/cuda-env.sh"

CUDA_ROOT="${LI_CUDA_TOOLKIT_ROOT:-${ROOT}/.venv-cuda-libs/lib/python3.12/site-packages/nvidia/cu13}"
export CUDAToolkit_ROOT="${CUDA_ROOT}"
export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${CUDA_ROOT}/nvvm/bin:${PATH}"
export CMAKE_CUDA_ARCHITECTURES="${LI_CMAKE_CUDA_ARCHITECTURES:-120}"
export BINDGEN_EXTRA_CLANG_ARGS="${BINDGEN_EXTRA_CLANG_ARGS:--I$(gcc -print-file-name=include)}"
export RUSTFLAGS="${RUSTFLAGS:--L native=${CUDA_ROOT}/lib -L native=/usr/lib/x86_64-linux-gnu}"

cargo build --release --features cuda
