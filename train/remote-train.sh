#!/usr/bin/env bash
# 在租用的 GPU 上训练，结束或 Ctrl+C 后把 last 权重拉回本机。
#
#   bash train/remote-train.sh so101-pick-cube-clean
#   bash train/remote-train.sh so101-pick-cube-clean --steps=20000
#   bash train/remote-train.sh so101-pick-cube-clean --overwrite
#   bash train/remote-train.sh so101-pick-cube-clean --resume=true
#   bash train/remote-train.sh --pull-only so101-pick-cube-clean
#
# 远程训练入口是 /root/run-train.sh。权重拉到：
#   ~/outputs/act/<任务名>/checkpoints/last/pretrained_model
# 和 control/rollout.sh 读的路径一致。SSH 主机默认是 ~/.ssh/config 里的 gpushare。

set -euo pipefail

REMOTE="${TRAIN_SSH:-gpushare}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-/root/run-train.sh}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/root/outputs/act}"

usage() {
  echo "用法: bash train/remote-train.sh [--pull-only] <任务名> [--overwrite] [lerobot-train 参数...]" >&2
  echo "权重拉到: \${HOME}/outputs/act/<任务名>/checkpoints/last/pretrained_model" >&2
  exit 2
}

pull_last() {
  local task="$1"
  local remote_model="${REMOTE_OUTPUT_ROOT}/${task}/checkpoints/last/pretrained_model/"
  local local_model="${HOME}/outputs/act/${task}/checkpoints/last/pretrained_model/"
  if ! ssh -o BatchMode=yes "$REMOTE" "test -f '${remote_model}model.safetensors'"; then
    echo "远程还没有 last 权重: ${remote_model}model.safetensors" >&2
    return 1
  fi
  mkdir -p "$local_model"
  rsync -avz --copy-links --partial --progress \
    -e "ssh -o BatchMode=yes" \
    "${REMOTE}:${remote_model}" "$local_model"
  echo "已同步到 ${local_model}"
}

PULL_ONLY=0
if [[ "${1:-}" == "--pull-only" ]]; then
  PULL_ONLY=1
  shift
fi

if [[ $# -lt 1 || "$1" == -* ]]; then
  usage
fi

TASK="$1"
shift

if [[ "$PULL_ONLY" -eq 1 ]]; then
  pull_last "$TASK"
  exit 0
fi

pulled=0
on_interrupt() {
  if [[ "$pulled" -eq 1 ]]; then
    return
  fi
  pulled=1
  echo "训练中断，正在把已有的 last 权重拉回本机..." >&2
  pull_last "$TASK" || true
  exit 130
}
trap on_interrupt INT TERM

ssh "$REMOTE" "bash ${REMOTE_SCRIPT} $(printf '%q ' "$TASK" "$@")"
echo "训练结束，正在把 last 权重拉回本机..." >&2
pulled=1
pull_last "$TASK"
