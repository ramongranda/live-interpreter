#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_INSTALL_DIR="${OVT_QWEN_INSTALL_DIR:-${ROOT}/vendor/qwen3_tts_rs}"
CUDA_WHEEL_VENV="${OVT_CUDA_WHEEL_VENV:-${ROOT}/.venv-cuda-libs}"

python_torch_lib_path() {
  local torch_lib="${CUDA_WHEEL_VENV}/lib/python3.12/site-packages/torch/lib"
  if [ -d "${torch_lib}" ]; then
    printf "%s" "${torch_lib}"
  fi
}

cuda_wheel_lib_path() {
  if [ -d "${CUDA_WHEEL_VENV}" ]; then
    find "${CUDA_WHEEL_VENV}" -type d -path "*/nvidia/*/lib" | sort | paste -sd: -
  fi
}

PYTORCH_LIB_PATH="${OVT_PYTORCH_LIB_PATH:-$(python_torch_lib_path)}"
CUDA_LIB_PATH="${OVT_CUDA_LIB_PATH:-$(cuda_wheel_lib_path)}"
EXTRA_CUDA_LIB_PATH="${OVT_EXTRA_CUDA_LIB_PATH:-/home/rgranda/.local/ollama-v0.30.6/lib/ollama/cuda_v12:/home/rgranda/.cache/uv/archive-v0/7fYrxrEsT4mtow-nv-N7X/triton/backends/nvidia/lib/cupti}"

export LD_LIBRARY_PATH="${PYTORCH_LIB_PATH}:${QWEN_INSTALL_DIR}/libtorch/lib:${CUDA_LIB_PATH}:${EXTRA_CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
