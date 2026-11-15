#!/bin/bash
# ----------------------------------------------------------------------------
# bash policy/MyPolicy/eval.sh <task_name> <setting> [checkpoint_path|auto] [gpu_id] [extra_opts...]
# bash policy/MyPolicy/eval.sh click_bell random auto 0 --max_episodes 5
#
# Each task runs the backend and checkpoint configured for it in
# deploy_policy.yml (task_policies). The third argument is kept for
# compatibility with the official eval.sh and is not used.
# Every action goes through action_post.py (real-robot joint limits, gripper 0-1).
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
POLICY_NAME=MyPolicy

# Simulator / adapter Python: PYTHON_BIN, else the official ACT environment
# (`uv sync` in policy/act, README step 3), else `python`.
ACT_VENV_PYTHON="$REPO_ROOT/policy/act/.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "$ACT_VENV_PYTHON" ]]; then
    PYTHON_BIN="$ACT_VENV_PYTHON"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi
# OfI worker environment and sources (README step 4); unset values fall back
# to directories next to the RoboSynChallenge checkout.
if [[ -z "${OFI_PYTHON:-}" && -x "$WORKSPACE_ROOT/ofi_env/bin/python" ]]; then
    export OFI_PYTHON="$WORKSPACE_ROOT/ofi_env/bin/python"
fi
if [[ -z "${OFI_SRC_ROOT:-}" && -d "$WORKSPACE_ROOT/ofi_src" ]]; then
    export OFI_SRC_ROOT="$WORKSPACE_ROOT/ofi_src"
fi
# Checkpoint root (README step 5): deploy_policy.yml paths use ${MYPOLICY_CKPT_ROOT}.
export MYPOLICY_CKPT_ROOT="${MYPOLICY_CKPT_ROOT:-$WORKSPACE_ROOT/mypolicy_ckpt}"

USAGE="Usage: bash policy/$POLICY_NAME/eval.sh <task_name> <setting> [checkpoint_path|auto] [gpu_id] [extra_opts...]"
TASK_NAME="${1:?$USAGE}"
SETTING="${2:?$USAGE}"
CHECKPOINT_PATH="${3:-auto}"
GPU_ID="${4:-0}"
shift 4 2>/dev/null || shift $#
EXTRA_ARGS=("$@")
for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        true) EXTRA_ARGS[$i]=True ;;
        false) EXTRA_ARGS[$i]=False ;;
        none|null) EXTRA_ARGS[$i]=None ;;
    esac
done
MODEL_NAME="${POLICY_NAME}_${TASK_NAME}"

if [[ ! -f "$REPO_ROOT/scripts/eval_policy.py" ]]; then
    echo "Error: $SCRIPT_DIR must be placed at <RoboSynChallenge>/policy/$POLICY_NAME" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    exit 1
fi

# DexSim picks the physical GPU from --gpu_id, so the simulator sees all GPUs;
# the OfI worker is pinned to that card by the backend.
unset CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
    export PYTHONPATH="$EMBODICHAIN_ROOT:$PYTHONPATH"
fi
export EMBODICHAIN_SIM_EXIT_PROCESS=0
# The evaluation downloads nothing.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================="
echo "  MyPolicy Evaluation (finals)"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: per task, deploy_policy.yml (MYPOLICY_CKPT_ROOT=$MYPOLICY_CKPT_ROOT)"
echo "  GPU:        $GPU_ID"
echo "========================================="

cd "$REPO_ROOT"
PYTHONWARNINGS=ignore::UserWarning "$PYTHON_BIN" scripts/eval_policy.py \
    --config "policy/$POLICY_NAME/deploy_policy.yml" \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --model_name "$MODEL_NAME" \
    --gpu_id "$GPU_ID" \
    "${EXTRA_ARGS[@]}"
