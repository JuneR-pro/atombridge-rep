#!/usr/bin/env bash
set -euo pipefail
atombridge_train_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,5,6,7}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
IFS=',' read -r -a atombridge_train_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#atombridge_train_gpus[@]} -ne 4 ]]; then
    echo "This launcher requires four GPUs in CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi
for atombridge_train_gpu in "${atombridge_train_gpus[@]}"; do
    atombridge_train_used="$(nvidia-smi -i "$atombridge_train_gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    if (( atombridge_train_used > 512 )); then
        echo "GPU $atombridge_train_gpu already uses $atombridge_train_used MiB; choose four idle GPUs." >&2
        exit 1
    fi
done
exec bash "$atombridge_train_root/scripts/atombridge_python.sh" -u \
    "$atombridge_train_root/scripts/formal_training.py" train "$@"
