# 训练

换任务只换第一个参数，和 `control/rollout.sh` 一样：

```bash
bash train/train.sh so101-adjust-bottle
bash train/train.sh so101-pick-cube --steps=20000
```

数据集是 `yutongzhu/<任务名>`，checkpoint 写到 `~/outputs/act/<任务名>`。输出目录已存在时，训练会拒绝覆盖。

训练开始后终端会打印 TensorBoard 地址，例如 `http://192.168.18.x:6006/`。电脑浏览器打开即可看 `loss`、读数据 `data_s`、反向更新 `updt_s`，以及两者之和 `step_s`。默认每 50 步记一次。电脑不要走会劫持局域网的代理。

默认策略是 ACT。Docker 或 π₀ 时用环境变量改 `POLICY`。其余参数原样传给 `lerobot-train`。

## 开发板

数据在默认缓存 `~/.cache/huggingface/lerobot/yutongzhu/<任务名>`，不需要再下载。脚本会激活 `lerobot-train` 环境。若 TensorBoard 没起来，先在该环境里 `pip install tensorboard`。

```bash
bash train/train.sh so101-adjust-bottle
```

训完后推理：

```bash
bash control/rollout.sh so101-adjust-bottle
```

## 租用 GPU

在这台 Mac 上启动，训完（或 Ctrl+C）会把 `last` 权重拉到本机 `~/outputs/act/<任务名>/checkpoints/last/pretrained_model`：

```bash
bash train/remote-train.sh so101-pick-cube-clean
bash train/remote-train.sh so101-pick-cube-clean --overwrite
bash train/remote-train.sh so101-pick-cube-clean --resume=true
bash train/remote-train.sh --pull-only so101-pick-cube-clean
```

输出目录已存在时，LeRobot 不会覆盖。重新训练加 `--overwrite`（会删掉远程该任务的 checkpoint 和 TensorBoard）。接着上次的步数加 `--resume=true`。

## Docker

镜像里是 CUDA 12.8、PyTorch 2.11 和 LeRobot 0.6.2。宿主机需要支持 CUDA 12.8 的 NVIDIA 驱动（570.86 及以上）和 NVIDIA Container Toolkit。

在仓库根目录构建：

```bash
docker build -t lerobot-train train
```

`POLICY` 取 `act`（默认）、`diffusion`、`pi0`、`pi05`。`pi0_5` 会转成 `pi05`。

### 普通训练

一张卡上直接跑。下面用全部可见 GPU 里的默认卡：

```bash
docker run --gpus all --rm --shm-size 16g \
  -p 6006:6006 \
  -e HF_TOKEN \
  -e BATCH_SIZE=8 \
  -e STEPS=20000 \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v /data/outputs:/outputs \
  lerobot-train record-test
```

数据集在本机目录时，再挂上数据并指定根目录：

```bash
docker run --gpus all --rm --shm-size 16g \
  -p 6006:6006 \
  -e HF_TOKEN \
  -e DATASET_ROOT=/data \
  -v /path/to/dataset:/data \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v /data/outputs:/outputs \
  lerobot-train record-test
```

Checkpoint 写到宿主机的 `/data/outputs/act/<任务名>`。

### DDP 训练

多张卡做数据并行时，每张卡放一份完整模型，各读不同样本，每步对梯度做平均。`--batch_size` 是每张卡的 batch，有效 batch 等于 `BATCH_SIZE × 进程数`。DDP 不降低单卡显存。π₀ 和 π₀.₅ 单卡放不下时，加卡仍然放不下。

用 Accelerate 启动同一个入口。4 张卡、每卡 batch 8 时，有效 batch 是 32：

```bash
docker run --gpus all --rm --shm-size 16g --ipc=host \
  --entrypoint accelerate \
  -p 6006:6006 \
  -e HF_TOKEN \
  -e BATCH_SIZE=8 \
  -e STEPS=20000 \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v /data/outputs:/outputs \
  lerobot-train \
  launch --multi_gpu --num_processes=4 --no_python /usr/local/bin/train record-test
```

`--num_processes` 要等于这次使用的 GPU 数量。日志和 checkpoint 由 0 号进程写入输出目录。

## 参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| 命令的第一个词 | 无，必须给 | 任务名。数据集为 `yutongzhu/<任务名>`，输出为 `~/outputs/act/<任务名>`。含 `/` 时当作完整 `repo_id` |
| `POLICY` | `act` | 训练哪种策略：`act`、`diffusion`、`pi0`、`pi05` |
| `HF_USER` | `yutongzhu` | 任务名不含 `/` 时拼到数据集仓库前面 |
| `DATASET_REPO_ID` | `yutongzhu/<任务名>` | 覆盖数据集仓库名 |
| `DATASET_ROOT` | 不设置则从 Hub 缓存读取 | 容器内的数据集目录。设置后对应 `--dataset.root` |
| `HF_TOKEN` | 空 | 读取私有数据集、下载 π₀ / π₀.₅ 权重时的 Hugging Face 令牌 |
| `BATCH_SIZE` | `2` | 每张卡的 batch。DDP 下有效 batch 为该值乘以进程数 |
| `STEPS` | `100000` | 优化器更新次数 |
| `SAVE_FREQ` | `2000` | 每隔多少步写一次 checkpoint |
| `LOG_FREQ` | `50` | 每隔多少步把 loss 和延迟写到 TensorBoard |
| `NUM_WORKERS` | `0` | 每个进程的 DataLoader 工作进程数。多卡时别开得过大 |
| `OUTPUT_DIR` | 有 `/outputs` 时为 `/outputs/<policy>/<任务名>`，否则 `~/outputs/<policy>/<任务名>` | checkpoint 目录。Docker 应落在挂载出来的卷上 |
| `JOB_NAME` | 与任务名相同 | 这次运行的名字，写入训练配置 |
| `PRETRAINED_PATH` | `pi0` 为 `lerobot/pi0_base`，`pi05` 为 `lerobot/pi05_base`；ACT 和 Diffusion 不使用 | 只对 `pi0`、`pi05` 生效。设成空字符串则不加载预训练权重。脚本还会为这两类打开 `bfloat16` 和 gradient checkpointing |
| 命令末尾的其它参数 | 无 | 原样传给 `lerobot-train`，例如 `--optimizer.lr=1e-4`、`--seed=1000`。脚本会先写上 `--wandb.enable=false`，要记日志时在末尾追加 `--wandb.enable=true --wandb.project=项目名` |
| `--num_processes` | 不走 DDP 时为 1 | Accelerate 的进程数，等于参与 DDP 的 GPU 数 |
| `--multi_gpu` | 仅 DDP 命令里写上 | 告诉 Accelerate 使用本机多卡 |
| `--shm-size` | 需要显式设置 | 共享内存。DataLoader 多进程建议 `16g` |
| `--ipc=host` | 仅 DDP 需要 | 让卡间通信能使用宿主机共享内存 |
