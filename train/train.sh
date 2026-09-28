#!/usr/bin/env bash
# 在带 NVIDIA GPU 的机器上：
#
#   docker build -t lerobot-train train
#   docker run --gpus all --rm --shm-size 16g \
#     -e HF_TOKEN \
#     -e DATASET_REPO_ID=yutongzhu/record-test \
#     -v "$HOME/.cache/huggingface:/cache/huggingface" \
#     -v /data/outputs:/outputs \
#     lerobot-train act
#
# policy：act、diffusion、pi0、pi05。
# 本地数据集再加：-v /path/to/dataset:/data -e DATASET_ROOT=/data
# 可选环境变量：BATCH_SIZE、STEPS、NUM_WORKERS、OUTPUT_DIR、JOB_NAME、PRETRAINED_PATH。
# PRETRAINED_PATH 默认是 lerobot/pi0_base 或 lerobot/pi05_base；设成空字符串则不加载预训练权重。
# 其余参数原样传给 lerobot-train。

set -euo pipefail

usage() {
  echo "用法: train act|diffusion|pi0|pi05" >&2
  echo "需要环境变量 DATASET_REPO_ID。" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
policy="$1"
shift

case "$policy" in
  act|diffusion|pi0|pi05) ;;
  pi0_5) policy=pi05 ;;
  *) usage ;;
esac

if [[ -z "${DATASET_REPO_ID:-}" ]]; then
  echo "需要设置 DATASET_REPO_ID，例如 yutongzhu/record-test" >&2
  exit 2
fi

output_dir="${OUTPUT_DIR:-/outputs/${policy}}"
mkdir -p "$output_dir"

args=(
  --policy.type="$policy"
  --dataset.repo_id="$DATASET_REPO_ID"
  --output_dir="$output_dir"
  --job_name="${JOB_NAME:-$policy}"
  --policy.device=cuda
  --wandb.enable=false
)

if [[ -n "${DATASET_ROOT:-}" ]]; then
  args+=(--dataset.root="$DATASET_ROOT")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  args+=(--batch_size="$BATCH_SIZE")
fi
if [[ -n "${STEPS:-}" ]]; then
  args+=(--steps="$STEPS")
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
  args+=(--num_workers="$NUM_WORKERS")
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

printf '启动:'
printf ' %q' lerobot-train "${args[@]}" "$@"
printf '\n'
exec lerobot-train "${args[@]}" "$@"
