#!/usr/bin/env bash
set -euo pipefail

################################################################################
# USER CONFIGURATION SECTION
# Please modify all settings below according to your environment.
################################################################################

############################
# 1. Path Configuration
############################
# Conda installation root directory, e.g. /opt/miniconda3.
CONDA_ROOT="PUT_YOUR_CONDA_ROOT_PATH_HERE"
# Conda environment name that contains verl dependencies.
ENV_NAME="PUT_YOUR_CONDA_ENV_NAME_HERE"
# Project repository path (directory that contains this script).
REPO_DIR="PUT_YOUR_REPO_PATH_HERE"
# Base working directory for cache/outputs/checkpoints.
BASE="PUT_YOUR_BASE_PATH_HERE"

############################
# 2. Model and Data Paths
############################
# Pre-trained model name/path.
MODEL="PUT_YOUR_MODEL_PATH_HERE"
# Data directory that stores parquet files.
DATA_DIR="PUT_YOUR_DATA_DIR_HERE"
# Training data file name (joined with DATA_DIR).
TRAIN_FILE="PUT_YOUR_TRAINING_FILE_HERE"
# Validation data file name (joined with DATA_DIR).
VAL_FILE="PUT_YOUR_VALIDATION_FILE_HERE"


############################
# 3. GPU Configuration
############################
# Number of GPUs to use per node.
GPUS=4
# Visible GPU IDs. Override this env var if needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# GPU memory utilization ratio for vLLM rollout.
GPU_UTIL=0.8
NNODES=1
############################
# 4. Training Configuration
############################
# Python executable inside the activated conda environment.
PYTHON_BIN="python"
# GRPO group size (positive integer), e.g. 8.
GROUP_SIZE="PUT_YOUR_GROUP_SIZE_HERE"
# Run tag used in checkpoint/output paths, e.g. gsm8k_g8.
RUN_TAG="PUT_YOUR_RUN_TAG_HERE"
# Experiment name shown in logs, e.g. gsm8k_g8.
EXPERIMENT_NAME="PUT_YOUR_EXPERIMENT_NAME_HERE"
# Project name namespace for wandb logs.
PROJECT_NAME="PUT_YOUR_PROJECT_NAME_HERE"

FIXED_TRAIN_BATCH_SIZE=1024
FIXED_PPO_MINI_BATCH_SIZE=256
FIXED_EPOCH=128


################################################################################
# Validation and Environment Setup
################################################################################

require_config() {
  local var_name="$1"
  local var_value="$2"
  if [[ -z "$var_value" ]] || [[ "$var_value" == PUT_YOUR_*_HERE ]]; then
    echo "[ERROR] Please set $var_name in the USER CONFIGURATION SECTION."
    exit 1
  fi
}

require_config "CONDA_ROOT" "$CONDA_ROOT"
require_config "ENV_NAME" "$ENV_NAME"
require_config "REPO_DIR" "$REPO_DIR"
require_config "BASE" "$BASE"
require_config "MODEL" "$MODEL"
require_config "DATA_DIR" "$DATA_DIR"
require_config "TRAIN_FILE" "$TRAIN_FILE"
require_config "VAL_FILE" "$VAL_FILE"
require_config "GROUP_SIZE" "$GROUP_SIZE"
require_config "RUN_TAG" "$RUN_TAG"
require_config "EXPERIMENT_NAME" "$EXPERIMENT_NAME"
require_config "PROJECT_NAME" "$PROJECT_NAME"

TRAIN_PATH="${DATA_DIR%/}/${TRAIN_FILE}"
VAL_PATH="${DATA_DIR%/}/${VAL_FILE}"

if ! [[ "$GROUP_SIZE" =~ ^[0-9]+$ ]] || (( GROUP_SIZE <= 0 )); then
  echo "[ERROR] GROUP_SIZE must be a positive integer, got: $GROUP_SIZE"
  exit 1
fi

if (( FIXED_TRAIN_BATCH_SIZE % GROUP_SIZE != 0 )); then
  echo "[ERROR] FIXED_TRAIN_BATCH_SIZE ($FIXED_TRAIN_BATCH_SIZE) must be divisible by GROUP_SIZE ($GROUP_SIZE)"
  exit 1
fi

if (( FIXED_PPO_MINI_BATCH_SIZE % GROUP_SIZE != 0 )); then
  echo "[ERROR] FIXED_PPO_MINI_BATCH_SIZE ($FIXED_PPO_MINI_BATCH_SIZE) must be divisible by GROUP_SIZE ($GROUP_SIZE)"
  exit 1
fi

if (( FIXED_EPOCH % GROUP_SIZE != 0 )); then
  echo "[ERROR] FIXED_EPOCH ($FIXED_EPOCH) must be divisible by GROUP_SIZE ($GROUP_SIZE)"
  exit 1
fi

TRAIN_BATCH_SIZE=$(( FIXED_TRAIN_BATCH_SIZE / GROUP_SIZE ))
PPO_MINI_BATCH_SIZE=$(( FIXED_PPO_MINI_BATCH_SIZE / GROUP_SIZE ))
EPOCH=$(( FIXED_EPOCH / GROUP_SIZE ))

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "[ERROR] conda.sh not found under CONDA_ROOT: $CONDA_ROOT"
  exit 1
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "[ERROR] REPO_DIR not found: $REPO_DIR"
  exit 1
fi

if [[ ! -f "$TRAIN_PATH" ]]; then
  echo "[ERROR] Training parquet not found: $TRAIN_PATH"
  exit 1
fi

if [[ ! -f "$VAL_PATH" ]]; then
  echo "[ERROR] Validation parquet not found: $VAL_PATH"
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found in current environment: $PYTHON_BIN"
  exit 1
fi

cd "$REPO_DIR"

CACHE_ROOT="$BASE/hf_cache"
OUTPUT_ROOT="$BASE/outputs"
CHECKPOINT_ROOT="$BASE/checkpoints"
HYDRA_OUTPUT_DIR="$BASE/hydra_outputs"
TMPDIR="$BASE/tmp"
LOCALDIR="$TMPDIR"
WANDB_DIR="$OUTPUT_ROOT"

mkdir -p \
  "$CACHE_ROOT" \
  "$OUTPUT_ROOT" \
  "$CHECKPOINT_ROOT" \
  "$HYDRA_OUTPUT_DIR" \
  "$TMPDIR" \
  "$LOCALDIR" \
  "$WANDB_DIR"

export HF_HOME="$CACHE_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$BASE"
export TMPDIR
export LOCALDIR
export WANDB_DIR
export HYDRA_FULL_ERROR=1

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "[INFO] REPO_DIR        = $REPO_DIR"
echo "[INFO] MODEL           = $MODEL"
echo "[INFO] GROUP_SIZE      = $GROUP_SIZE"
echo "[INFO] TRAIN_BATCH     = $TRAIN_BATCH_SIZE"
echo "[INFO] PPO_MINI_BATCH  = $PPO_MINI_BATCH_SIZE"
echo "[INFO] TOTAL_EPOCHS    = $EPOCH"
echo "[INFO] TRAIN_PATH      = $TRAIN_PATH"
echo "[INFO] VAL_PATH        = $VAL_PATH"
echo "[INFO] EXPERIMENT_NAME = $EXPERIMENT_NAME"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# REWARD_FUNCTION_PATH is kept for user customization.
# If your verl config supports a reward script argument, append it manually below.

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_PATH" \
  data.val_files="$VAL_PATH" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_UTIL" \
  actor_rollout_ref.rollout.n="$GROUP_SIZE" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  'trainer.logger=["console","wandb"]' \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$CHECKPOINT_ROOT/$PROJECT_NAME/$RUN_TAG" \
  trainer.n_gpus_per_node="$GPUS" \
  trainer.nnodes="$NNODES" \
  trainer.save_freq=100 \
  trainer.test_freq=5 \
  trainer.total_epochs="$EPOCH" \
  "actor_rollout_ref.actor.checkpoint.save_contents=['model']" \
  data.shuffle=True \
  hydra.run.dir="$HYDRA_OUTPUT_DIR"
