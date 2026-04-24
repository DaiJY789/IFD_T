#!/usr/bin/env bash
set -euo pipefail

# Usage (example):
# bash /data/home/yxzhou/jydai/work/IFD_T/train/run_verl_ifd_t_tool_train.sh \
#   --train-file /ssd2/jydai/data/IFD_T/ifd_t_105k_v1/train/train.parquet \
#   --val-file /ssd2/jydai/data/IFD_T/ifd_t_105k_v1/test/test.parquet \
#   --model-path /ssd2/jydai/model/Qwen3-VL-8B-Instruct \
#   --gpus 0,1,2,3,4,5,6,7

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
IFD_T_ROOT="$PROJECT_DIR/work/IFD_T"
VERL_REPO_DIR="$PROJECT_DIR/verl"
CONFIG_PATH="$PROJECT_DIR/verl/examples/sglang_multiturn/config"
CONFIG_NAME="gsm8k_multiturn_grpo"

TRAIN_FILE="${TRAIN_FILE:-/ssd2/jydai/data/IFD_T/ifd_t_105k_v1/train/train_rl.parquet}"
VAL_FILE="${VAL_FILE:-/ssd2/jydai/data/IFD_T/ifd_t_105k_v1/test/test_rl.parquet}"
MODEL_PATH="${MODEL_PATH:-/ssd2/jydai/model/Qwen3-VL-8B-Instruct}"

TOOL_CONFIG="${TOOL_CONFIG:-$PROJECT_DIR/work/IFD_T/tools/verl_tool_config.yaml}"
REWARD_FN="${REWARD_FN:-$PROJECT_DIR/work/IFD_T/reward/reward.py}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}"

RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/work/IFD_T/outputs/train_runs}"
RUN_NAME="${RUN_NAME:-ifd_t_tool_grpo_$(date +%Y%m%d_%H%M%S)}"

TRAIN_BSZ="${TRAIN_BSZ:-32}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-6144}"
MAX_RESP_LEN="${MAX_RESP_LEN:-2048}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-2800}"
TEST_FREQ="${TEST_FREQ:-400}"
SAVE_FREQ="${SAVE_FREQ:-400}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-4}"
MAX_USER_TURNS="${MAX_USER_TURNS:-3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEF="${ENTROPY_COEF:-0.001}"
ROLLOUT_TEMP="${ROLLOUT_TEMP:-0.2}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
TOKEN_SANITY_MODE="${TOKEN_SANITY_MODE:-ignore_strippable}"
USE_INFERENCE_CHAT_TEMPLATE="${USE_INFERENCE_CHAT_TEMPLATE:-True}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-3}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-file) TRAIN_FILE="$2"; shift 2 ;;
    --val-file) VAL_FILE="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --gpus) CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --n-gpus-per-node) N_GPUS_PER_NODE="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --train-bsz) TRAIN_BSZ="$2"; shift 2 ;;
    --max-prompt-len) MAX_PROMPT_LEN="$2"; shift 2 ;;
    --max-resp-len) MAX_RESP_LEN="$2"; shift 2 ;;
    --rollout-n) ROLLOUT_N="$2"; shift 2 ;;
    --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
    --test-freq) TEST_FREQ="$2"; shift 2 ;;
    --save-freq) SAVE_FREQ="$2"; shift 2 ;;
    --max-assistant-turns) MAX_ASSISTANT_TURNS="$2"; shift 2 ;;
    --max-user-turns) MAX_USER_TURNS="$2"; shift 2 ;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
    --actor-lr) ACTOR_LR="$2"; shift 2 ;;
    --kl-loss-coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy-coef) ENTROPY_COEF="$2"; shift 2 ;;
    --rollout-temp) ROLLOUT_TEMP="$2"; shift 2 ;;
    --rollout-top-p) ROLLOUT_TOP_P="$2"; shift 2 ;;
    --token-sanity-mode) TOKEN_SANITY_MODE="$2"; shift 2 ;;
    --use-inference-chat-template) USE_INFERENCE_CHAT_TEMPLATE="$2"; shift 2 ;;
    --tool-config) TOOL_CONFIG="$2"; shift 2 ;;
    --reward-fn) REWARD_FN="$2"; shift 2 ;;
    --resume-mode) RESUME_MODE="$2"; shift 2 ;;
    --resume-from-path) RESUME_FROM_PATH="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "[ERROR] train/val parquet not found"
  echo "        TRAIN_FILE=$TRAIN_FILE"
  echo "        VAL_FILE=$VAL_FILE"
  echo "        You can override via --train-file / --val-file"
  exit 3
fi

if [[ ! -f "$TOOL_CONFIG" ]]; then
  echo "[ERROR] TOOL_CONFIG not found: $TOOL_CONFIG"
  exit 4
fi

if [[ ! -f "$REWARD_FN" ]]; then
  echo "[ERROR] REWARD_FN not found: $REWARD_FN"
  exit 5
fi

if [[ ! -d "$VERL_REPO_DIR" ]]; then
  echo "[ERROR] verl repo not found: $VERL_REPO_DIR"
  exit 6
fi

if (( TRAIN_BSZ % N_GPUS_PER_NODE != 0 )); then
  echo "[ERROR] train-bsz ($TRAIN_BSZ) must be divisible by n-gpus-per-node ($N_GPUS_PER_NODE)"
  exit 7
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$IFD_T_ROOT:$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN_DIR="$RUN_ROOT/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
OUTPUT_DIR="$RUN_DIR/outputs"
STDOUT_LOG_PATH="$LOG_DIR/train_stdout.log"
METRICS_JSONL_PATH="$LOG_DIR/train_metrics.jsonl"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
ROLLOUT_DATA_DIR="$OUTPUT_DIR/rollout_data"
VALIDATION_DATA_DIR="$OUTPUT_DIR/validation_data"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$ROLLOUT_DATA_DIR" "$VALIDATION_DATA_DIR"

export VERL_FILE_LOGGER_PATH="$METRICS_JSONL_PATH"

echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] TRAIN_FILE=$TRAIN_FILE"
echo "[INFO] VAL_FILE=$VAL_FILE"
echo "[INFO] TOOL_CONFIG=$TOOL_CONFIG"
echo "[INFO] REWARD_FN=$REWARD_FN"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] N_GPUS_PER_NODE=$N_GPUS_PER_NODE"
echo "[INFO] RUN_DIR=$RUN_DIR"
echo "[INFO] MAX_ASSISTANT_TURNS=$MAX_ASSISTANT_TURNS"
echo "[INFO] MAX_USER_TURNS=$MAX_USER_TURNS"
echo "[INFO] ROLLOUT_TEMP=$ROLLOUT_TEMP"
echo "[INFO] MAX_RESP_LEN=$MAX_RESP_LEN"
echo "[INFO] MAX_PROMPT_LEN=$MAX_PROMPT_LEN"
echo "[INFO] USE_INFERENCE_CHAT_TEMPLATE=$USE_INFERENCE_CHAT_TEMPLATE"

RESUME_ARGS=("trainer.resume_mode=$RESUME_MODE")
if [[ "$RESUME_MODE" == "resume_path" ]]; then
  if [[ -z "$RESUME_FROM_PATH" ]]; then
    echo "[ERROR] RESUME_FROM_PATH is required when RESUME_MODE=resume_path"
    exit 8
  fi
  RESUME_ARGS+=("trainer.resume_from_path=$RESUME_FROM_PATH")
fi

cd "$VERL_REPO_DIR"

python3 -u -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name="$CONFIG_NAME" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BSZ" \
  data.max_prompt_length="$MAX_PROMPT_LEN" \
  data.max_response_length="$MAX_RESP_LEN" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  custom_reward_function.path="$REWARD_FN" \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LEN" \
  actor_rollout_ref.rollout.response_length="$MAX_RESP_LEN" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_USER_TURNS" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  actor_rollout_ref.rollout.multi_turn.use_inference_chat_template="$USE_INFERENCE_CHAT_TEMPLATE" \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode="$TOKEN_SANITY_MODE" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BSZ" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEF" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMP" \
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  trainer.logger='["console","file"]' \
  trainer.project_name='ifd_t_tool_rl' \
  trainer.experiment_name='ifd_t_tool_grpo' \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes=1 \
  "${RESUME_ARGS[@]}" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.val_before_train=True \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  2>&1 | tee "$STDOUT_LOG_PATH"

echo "[INFO] Training finished"
echo "[INFO] Stdout log: $STDOUT_LOG_PATH"
echo "[INFO] Metrics JSONL: $METRICS_JSONL_PATH"
echo "[INFO] Checkpoints: $CHECKPOINT_DIR"
