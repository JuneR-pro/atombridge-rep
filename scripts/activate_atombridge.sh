# Source this file to use the existing interpreter with AtomBridge dependencies.
atombridge_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
atombridge_python="${ATOMBRIDGE_BASE_PYTHON:-/home/yanfeng/code/EmA/EmbodiedAgentLab/.venv/bin/python}"

if [[ ! -x "$atombridge_python" || ! -d "$atombridge_root/.deps/mujoco" ]]; then
    echo "AtomBridge interpreter or project dependencies are missing." >&2
    return 1
fi

export ATOMBRIDGE_ROOT="$atombridge_root"
export ATOMBRIDGE_BASE_PYTHON="$atombridge_python"
export PATH="$(dirname -- "$atombridge_python"):$PATH"
export PYTHONPATH="$atombridge_root/.deps:$atombridge_root/third_party/openpi/src:$atombridge_root/third_party/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$atombridge_root/artifacts/runtime/matplotlib}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$atombridge_root/artifacts/runtime/hf_datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$atombridge_root/data/lerobot}"
cd -- "$atombridge_root"
