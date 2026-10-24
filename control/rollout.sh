#!/usr/bin/env bash
# 在开发板上用训练好的 ACT 驱动 follower。不用 leader，也不写数据集。
#
#   bash control/rollout.sh so101-adjust-bottle
#   bash control/rollout.sh so101-adjust-bottle --fp16
#   bash control/rollout.sh so101-pick-cube --duration=30
#   bash control/rollout.sh so101-pick-cube-clean --web-view
#   bash control/rollout.sh so101-pick-cube-clean --log-result
#   bash control/rollout.sh so101-pick-cube-clean --log-result=ensemble.json
#
# 第一个参数是任务名，对应训练输出目录：
#   ${HOME}/outputs/act/<任务名>/checkpoints/last/pretrained_model
# 其余参数原样传给 lerobot-rollout，可覆盖默认值，例如某个 checkpoint 或 --task。
#
# 终端每拍打印 capture/pre/infer/send/loop，结束时有 p50/p95/p99 和超时率。
# --log-result 跑完后询问成功/失败，把各阶段 p50/p99、整段耗时和实际节拍
# 追加到 ~/outputs/act/<任务名>/rollout_results.json。
# --log-result=ensemble.json 换文件名，仍在同一目录；也可写绝对路径。
# --fp16 会用 torch.autocast("cuda", dtype=torch.float16) 包住 select_action。
# 跑之前把物体和手臂摆回录数据时的起始状态，相机不要挪。
# 脚本一连上 follower 就会打开力矩，手臂会自己动。手边留着急停。
# 到时间或按 Ctrl+C 之后，手臂先回到启动时的姿势，再断开并关闭力矩。

set -euo pipefail

usage() {
  echo "用法: bash control/rollout.sh <任务名> [lerobot-rollout 参数...]" >&2
  echo "权重目录: \${HOME}/outputs/act/<任务名>/checkpoints/last/pretrained_model" >&2
  exit 2
}

if [[ $# -lt 1 || "$1" == -* ]]; then
  usage
fi

JOB_NAME="$1"
shift

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  # 录数据用的 lerobot 环境里，torch 按 CUDA 13 编译，这块板的驱动是 12.6，
  # 那个环境里 cuda 不可用。训练环境的 torch 2.8 能用到 Orin 的 GPU。
  conda activate lerobot-train
fi

# 训练输出目录下面，每 2000 步一个子目录。last 是指向最后一步的符号链接。
# 推理只读 pretrained_model：网络结构、权重、以及把关节和图像变回训练尺度的处理器。
# 旁边的 training_state 是给「接着训练」用的，指过去会加载失败。
POLICY_PATH="${HOME}/outputs/act/${JOB_NAME}/checkpoints/last/pretrained_model"
if [[ ! -f "${POLICY_PATH}/model.safetensors" ]]; then
  echo "找不到权重: ${POLICY_PATH}/model.safetensors" >&2
  exit 1
fi

args=(
  # base：策略自己动，不录数据、不接 leader。要存评估视频时再换成 episodic。
  --strategy.type=base

  # 指向含 config.json 和 model.safetensors 的目录。类型 act 写在 config.json 里，不用再传 --policy.type。
  --policy.path="${POLICY_PATH}"

  # 权重在 cuda 上训练。这个环境的 GPU 可用，推理也放在 cuda 上。
  --device=cuda

  # follower 的串口和校准 ID，和 record.py 一致。ID 用来找校准文件，错了手臂会按错误的角度动。
  --robot.type=so101_follower
  --robot.port=/dev/ttyACM1
  --robot.id=YutongZhu_awesome_follower_arm

  # 相机名字、分辨率必须和录数据时一样，否则处理器对不上图像。
  # 模型输入键是 observation.images.wrist 和 observation.images.front。
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: 90}, front: {type: opencv, index_or_path: /dev/video2, width: 480, height: 640, fps: 30, rotation: 90}}"

  # 录数据是 30 Hz。训练时 chunk_size=100、n_action_steps=100：一次推理给出 100 步，
  # 大约 3.3 秒才再看相机。temporal ensembling 要求每步都推理（n_action_steps=1），
  # 30 Hz 的预算是 33.3 ms。先只改步数、不打开 ensembling，看 Jetson 上一次
  # 原始 forward 要多久。确认能压进预算后，再打开下面的 temporal_ensemble_coeff。
  --fps=30
  --policy.n_action_steps=1
  # --policy.temporal_ensemble_coeff=0.01

  # 跑 60 秒，和录制时单集时长一样。改成 0 会一直跑，直到 Ctrl+C。
  --duration=60

  # ACT 的输入只有关节和图像，不读这句话。命令行仍要给一个任务名，日志里才知道这次在测什么。
  --task="${JOB_NAME}"

  # 结束时先回到脚本启动时读到的姿势，再断开。设成 false 就停在最后的姿势上。
  --return_to_initial_position=true

  # 不在开发板上做语音提示。
  --play_sounds=false

  # 要看画面时加 --web-view：先把这一段记到开发板 ~/lerobot-rerun/*.rrd，
  # 结束后开网页 Viewer，可以拖时间轴。不要用 --display_data 直播，两路相机会拖慢控制环。
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/log_act_inference_latency.py" "${args[@]}" "$@"
