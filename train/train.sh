#!/usr/bin/env bash
# 用本地数据集训练。换任务只换第一个参数：
#
#   bash train/train.sh so101-adjust-bottle
#   bash train/train.sh so101-pick-cube --steps=20000
#
# 第一个参数是任务名，对应：
#   数据集  ${HF_USER:-yutongzhu}/<任务名>
#   输出    ${HOME}/outputs/act/<任务名>
# 其余参数原样传给 lerobot-train，可覆盖默认值。
#
# Docker 镜像入口相同：
#   docker run ... -p 6006:6006 lerobot-train so101-adjust-bottle
# 训练时 TensorBoard 在 6006 端口，浏览器看 loss 和 data_s / updt_s。
# 输出目录已存在时，训练会拒绝覆盖。换目录或删掉旧目录后再跑。

set -euo pipefail

HF_USER="${HF_USER:-yutongzhu}"

usage() {
  echo "用法: bash train/train.sh <任务名> [lerobot-train 参数...]" >&2
  echo "数据集: ${HF_USER}/<任务名>" >&2
  echo "输出:   \${HOME}/outputs/act/<任务名>" >&2
  exit 2
}

if [[ $# -lt 1 || "$1" == -* ]]; then
  usage
fi

TASK="$1"
shift

if [[ "$TASK" == */* ]]; then
  REPO_ID="$TASK"
  TASK="${TASK##*/}"
else
  REPO_ID="${HF_USER}/${TASK}"
fi
REPO_ID="${DATASET_REPO_ID:-$REPO_ID}"

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate lerobot-train
fi

# 限制缓存切块，空闲过多时把显存还给驱动。必须在 Python 启动前设置。
# 不用 expandable_segments：它会一次申请很大的连续显存，这块板在第 1 步就会失败。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.5}"

policy="${POLICY:-act}"
case "$policy" in
  act|diffusion|pi0|pi05) ;;
  pi0_5) policy=pi05 ;;
  *)
    echo "未知 POLICY=${policy}。可用: act、diffusion、pi0、pi05。" >&2
    exit 2
    ;;
esac

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  output_dir="$OUTPUT_DIR"
elif [[ -d /outputs ]]; then
  output_dir="/outputs/${policy}/${TASK}"
else
  output_dir="${HOME}/outputs/${policy}/${TASK}"
fi

args=(
  --policy.type="$policy"
  --dataset.repo_id="$REPO_ID"
  --dataset.video_backend=pyav
  --policy.device=cuda
  --policy.repo_id="$REPO_ID"
  --accelerator.mixed_precision=bf16
  --output_dir="$output_dir"
  --job_name="${JOB_NAME:-$TASK}"
  --batch_size="${BATCH_SIZE:-2}"
  --num_workers="${NUM_WORKERS:-0}"
  --steps="${STEPS:-100000}"
  --save_freq="${SAVE_FREQ:-2000}"
  --log_freq="${LOG_FREQ:-50}"
  --wandb.enable=false
)

if [[ -n "${DATASET_ROOT:-}" ]]; then
  args+=(--dataset.root="$DATASET_ROOT")
fi

if [[ "$policy" == "pi0" || "$policy" == "pi05" ]]; then
  pretrained="${PRETRAINED_PATH-lerobot/${policy}_base}"
  if [[ -n "$pretrained" ]]; then
    args+=(
      --policy.pretrained_path="$pretrained"
      --policy.dtype=bfloat16
      --policy.gradient_checkpointing=true
    )
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf '启动:'
printf ' %q' python "${SCRIPT_DIR}/log_train_metrics.py" "${args[@]}" "$@"
printf '\n'
exec python "${SCRIPT_DIR}/log_train_metrics.py" "${args[@]}" "$@"
