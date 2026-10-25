# SO-101 · ACT：采集、训练、部署、TensorRT 优化

用 [LeRobot](https://github.com/huggingface/lerobot) 在 SO-101 机械臂上做抓取任务。整条链路都在这个仓库里：在 Jetson 上采集数据，在租用的 GPU 上训练 ACT，回到 Jetson 上驱动机械臂，最后手写 TensorRT 把推理压进 30 Hz 的控制预算。

- 权重：[`yutongzhu/so101-pick-cube-clean`](https://huggingface.co/yutongzhu/so101-pick-cube-clean)
- 数据集：[`yutongzhu/so101-pick-cube-clean`](https://huggingface.co/datasets/yutongzhu/so101-pick-cube-clean)（50 集，28160 帧）

## 结果

同一个 ACT 权重，在 Jetson Orin Nano 8GB 上运行，控制目标 30 Hz（每拍 33.3 ms）。

**单次 forward**（`deploy/trt/bench_torch.py`、`trtexec`）：

| | p50 | 相比 PyTorch FP32 |
|---|---:|---:|
| PyTorch FP32 | 59.3 ms | 1× |
| PyTorch autocast FP16 | 46.7 ms | 1.3× |
| TensorRT FP32 | 23.4 ms | 2.5× |
| **TensorRT FP16** | **9.6 ms** | **6.2×** |

**真机控制环**（temporal ensemble，`temporal_ensemble_coeff=0.01`）：

| | infer p50 | loop p50 | 实际频率 |
|---|---:|---:|---:|
| PyTorch FP32 | 67.2 ms | 72.1 ms | 13.8 Hz |
| **TensorRT FP16** | **13.7 ms** | **21.1 ms** | **29.8 Hz** |

TensorRT FP16 和 PyTorch 的输出差异：在 51 帧真实数据上，关节误差平均 0.016°、最大 0.26°，比舵机一步（约 0.09°）还小。

**真机成功率**（每组 24 次，[`eval/so101-pick-cube-clean`](eval/so101-pick-cube-clean)）：

| | 成功 | 成功用时中位数 |
|---|---:|---:|
| Action chunking（PyTorch） | 17/24 | 18.8 s |
| Temporal ensemble（PyTorch） | 18/24 | 33.8 s |
| Temporal ensemble（TensorRT FP16） | 待测 | 待测 |

## 目录

```
control/      在开发板上运行：遥操作、录数据、回放、rollout（含 --trt / --fp16 / --log-result / --web-view）
train/        训练：开发板本地、租用 GPU（remote-train.sh）、Docker
deploy/trt/   TensorRT：基线 → 导出 ONNX → 构建 engine → 推理 → 真实数据对比
eval/         真机评估记录、对比图、视频、模型卡
scripts/      在 Mac 上通过 SSH 操作开发板的小工具
.github/      push 到 main 后自动 rsync 到 Jetson
```

## 流程

| 阶段 | 在哪里 | 入口 | 详细说明 |
|---|---|---|---|
| 采集 | Jetson（conda `lerobot`） | `python control/record.py --dataset-name <任务名>` | [control/README.md](control/README.md) |
| 训练 | 租用 GPU 或 Jetson（conda `lerobot-train`） | `bash train/remote-train.sh <任务名>` | [train/README.md](train/README.md) |
| 部署 | Jetson（conda `lerobot-train`） | `bash control/rollout.sh <任务名>` | [control/README.md](control/README.md) 第 9 节 |
| 优化 | Jetson（conda `lerobot-train` + 系统 TensorRT 10.3） | 见下面 | [deploy/trt/PLAN.md](deploy/trt/PLAN.md) |

## TensorRT 优化

在开发板上，从仓库根目录按顺序运行：

```bash
python deploy/trt/bench_torch.py                                    # 1. PyTorch 基线
python deploy/trt/export_onnx.py                                    # 2. 导出 deploy/trt/act.onnx
/usr/bin/python3 deploy/trt/build_engine.py --onnx deploy/trt/act.onnx --engine deploy/trt/act_fp16.engine --fp16   # 3. 构建，约 8 分钟
python deploy/trt/trt_infer.py --engine deploy/trt/act_fp16.engine  # 4. 随机输入：精度 + 速度
python deploy/trt/check_real.py --engine deploy/trt/act_fp16.engine # 5. 真实数据：关节误差（度）
```

上真机：

```bash
bash control/rollout.sh so101-pick-cube-clean --trt=$HOME/Projects/pi0-lerobot-finetune-deploy/deploy/trt/act_fp16.engine --policy.temporal_ensemble_coeff=0.01 --log-result=trt_ensemble.json
```

几点要注意：

- engine 和 GPU 型号、TensorRT 版本绑定，换机器要重新构建。`.onnx`、`.engine` 不进 git。
- 构建时 CPU 内存峰值约 5 GB，要限制 workspace 为 1 GB，否则 8 GB 的板子会 OOM。
- `conda` 环境里用 TensorRT，要把系统的 `tensorrt` 包软链进来，做法见 [PLAN.md](deploy/trt/PLAN.md) 第 5 步。
- 每一步的原理、过关标准和踩过的坑，都记在 [PLAN.md](deploy/trt/PLAN.md) 里。

## 硬件

| | |
|---|---|
| 机械臂 | SO-101 follower + leader |
| 相机 | 腕部 `wrist` + 场景 `front`，各 640×480 顺时针旋转 90°，30 fps |
| 端侧 | Jetson Orin Nano 8GB Super（MAXN_SUPER），JetPack 6.2，CUDA 12.6，TensorRT 10.3 |
| 训练 | RTX 2080 Ti（租用） |
