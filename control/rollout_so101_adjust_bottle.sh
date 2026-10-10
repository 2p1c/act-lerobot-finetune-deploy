#!/usr/bin/env bash
# 在开发板上用训练好的 ACT 驱动 follower，看它会不会自己做「扶正瓶子」。
# 不用 leader，也不写数据集。
#
#   bash control/rollout_so101_adjust_bottle.sh
#
# 这次先测每步原始推理延迟：n_action_steps=1，不打开 temporal ensembling。
# 终端会打印 [act raw] step=N xx.x ms，结束时有 mean/p50/p95。
# 跑之前把瓶子和手臂摆回录数据时的起始状态，相机不要挪。
# 脚本一连上 follower 就会打开力矩，手臂会自己动。手边留着急停。
# 到时间或按 Ctrl+C 之后，手臂先回到启动时的姿势，再断开并关闭力矩。

set -euo pipefail

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  # 录数据用的 lerobot 环境里，torch 按 CUDA 13 编译，这块板的驱动是 12.6，
  # 那个环境里 cuda 不可用。训练环境的 torch 2.8 能用到 Orin 的 GPU。
  conda activate lerobot-train
fi

# 训练输出目录下面，每 2000 步一个子目录。last 是指向最后一步（100000）的符号链接。
# 推理只读 pretrained_model：网络结构、权重、以及把关节和图像变回训练尺度的处理器。
# 旁边的 training_state 是给「接着训练」用的，指过去会加载失败。
POLICY_PATH="${HOME}/outputs/act/so101-adjust-bottle/checkpoints/last/pretrained_model"
if [[ ! -f "${POLICY_PATH}/model.safetensors" ]]; then
  echo "找不到权重: ${POLICY_PATH}/model.safetensors" >&2
  exit 1
fi

# 数据集里这一集的任务句子。ACT 的输入只有关节和 wrist 图像，不读这句话。
# 命令行仍要给一个任务名，日志里才知道这次在测什么。
TASK="Pick up the bottle on the table and headup"

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

  # 相机在数据集里的名字是 wrist。模型输入键是 observation.images.wrist。
  # 名字、分辨率必须和录数据时一样，否则处理器对不上图像。
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"

  # 录数据是 30 Hz。训练时 chunk_size=100、n_action_steps=100：一次推理给出 100 步，
  # 大约 3.3 秒才再看相机。temporal ensembling 要求每步都推理（n_action_steps=1），
  # 30 Hz 的预算是 33.3 ms。先只改步数、不打开 ensembling，看 Jetson 上一次
  # 原始 forward 要多久。确认能压进预算后，再打开下面的 temporal_ensemble_coeff。
  --fps=30
  --policy.n_action_steps=1
  # --policy.temporal_ensemble_coeff=0.01

  # 跑 60 秒，和录制时单集时长一样。改成 0 会一直跑，直到 Ctrl+C。
  --duration=60
  --task="${TASK}"

  # 结束时先回到脚本启动时读到的姿势，再断开。设成 false 就停在最后的姿势上。
  --return_to_initial_position=true

  # 不在开发板上做语音提示。
  --play_sounds=false

  # 要在电脑上看画面时，先在电脑上开 rerun --bind 0.0.0.0 --port 9876，
  # 再把下面两行加进 args，并在启动前 unset 掉 http_proxy / https_proxy / all_proxy：
  # --display_data=true
  # --display_ip=192.168.18.9
  # --display_port=9876
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/log_act_inference_latency.py" "${args[@]}"
