# 训练

镜像里是 CUDA 12.8、PyTorch 2.11 和 LeRobot 0.6.2。宿主机需要支持 CUDA 12.8 的 NVIDIA 驱动（570.86 及以上）和 NVIDIA Container Toolkit。

在仓库根目录构建：

```bash
docker build -t lerobot-train train
```

`policy` 取 `act`、`diffusion`、`pi0`、`pi05`。`pi0_5` 会转成 `pi05`。

## 普通训练

一张卡上直接跑。下面用全部可见 GPU 里的默认卡：

```bash
docker run --gpus all --rm --shm-size 16g \
  -e HF_TOKEN \
  -e DATASET_REPO_ID=yutongzhu/record-test \
  -e BATCH_SIZE=8 \
  -e STEPS=20000 \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v /data/outputs:/outputs \
  lerobot-train act
```

数据集在本机目录时，再挂上数据并指定根目录：

```bash
docker run --gpus all --rm --shm-size 16g \
  -e HF_TOKEN \
  -e DATASET_REPO_ID=yutongzhu/record-test \
  -e DATASET_ROOT=/data \
  -v /path/to/dataset:/data \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v /data/outputs:/outputs \
  lerobot-train act
```

Checkpoint 写到宿主机的 `/data/outputs/act`。

## DDP 训练

多张卡做数据并行时，每张卡放一份完整模型，各读不同样本，每步对梯度做平均。`--batch_size` 是每张卡的 batch，有效 batch 等于 `BATCH_SIZE × 进程数`。DDP 不降低单卡显存。π₀ 和 π₀.₅ 单卡放不下时，加卡仍然放不下。

用 Accelerate 启动同一个入口。4 张卡、每卡 batch 8 时，有效 batch 是 32：

```bash
docker run --gpus all --rm --shm-size 16g --ipc=host \
  --entrypoint accelerate \
  -e HF_TOKEN \
  -e DATASET_REPO_ID=yutongzhu/record-test \
  -e BATCH_SIZE=8 \
  -e STEPS=20000 \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v /data/outputs:/outputs \
  lerobot-train \
  launch --multi_gpu --num_processes=4 --no_python /usr/local/bin/train act
```

`--num_processes` 要等于这次使用的 GPU 数量。日志和 checkpoint 由 0 号进程写入 `OUTPUT_DIR`。

## 参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `policy`（命令的第一个词） | 无，必须给 | 训练哪种策略：`act`、`diffusion`、`pi0`、`pi05` |
| `DATASET_REPO_ID` | 无，必须给 | 数据集仓库名，例如 `yutongzhu/record-test` |
| `DATASET_ROOT` | 不设置则从 Hub 缓存读取 | 容器内的数据集目录。设置后对应 `--dataset.root` |
| `HF_TOKEN` | 空 | 读取私有数据集、下载 π₀ / π₀.₅ 权重时的 Hugging Face 令牌 |
| `BATCH_SIZE` | `lerobot-train` 的默认值 | 每张卡的 batch。DDP 下有效 batch 为该值乘以进程数 |
| `STEPS` | `lerobot-train` 的默认值 | 优化器更新次数 |
| `NUM_WORKERS` | `lerobot-train` 的默认值 | 每个进程的 DataLoader 工作进程数。多卡时别开得过大 |
| `OUTPUT_DIR` | `/outputs/<policy>` | checkpoint 目录，应落在挂载出来的卷上 |
| `JOB_NAME` | 与 `policy` 相同 | 这次运行的名字，写入训练配置 |
| `PRETRAINED_PATH` | `pi0` 为 `lerobot/pi0_base`，`pi05` 为 `lerobot/pi05_base`；ACT 和 Diffusion 不使用 | 只对 `pi0`、`pi05` 生效。设成空字符串则不加载预训练权重。脚本还会为这两类打开 `bfloat16` 和 gradient checkpointing |
| 命令末尾的其它参数 | 无 | 原样传给 `lerobot-train`，例如 `--optimizer.lr=1e-4`、`--seed=1000`。脚本会先写上 `--wandb.enable=false`，要记日志时在末尾追加 `--wandb.enable=true --wandb.project=项目名` |
| `--num_processes` | 不走 DDP 时为 1 | Accelerate 的进程数，等于参与 DDP 的 GPU 数 |
| `--multi_gpu` | 仅 DDP 命令里写上 | 告诉 Accelerate 使用本机多卡 |
| `--shm-size` | 需要显式设置 | 共享内存。DataLoader 多进程建议 `16g` |
| `--ipc=host` | 仅 DDP 需要 | 让卡间通信能使用宿主机共享内存 |
