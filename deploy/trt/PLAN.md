# ACT → TensorRT 学习与落地计划

目标：把 Hugging Face 上的 `yutongzhu/so101-pick-cube-clean`（ACT + ResNet18）在 Jetson Orin Nano 8GB 上用 TensorRT 跑起来，对比优化前后的延迟，并在真机上验证成功率没掉。TensorRT 的代码自己手写，每一步都知道发生了什么。

## 0. 现状（已经查过的事实）

| 项 | 值 |
|---|---|
| 板子 | Jetson Orin Nano 8GB Super，`nvpmodel` = MAXN_SUPER |
| 系统 | JetPack 6.2（L4T R36.4.7），CUDA 12.6，cuDNN 9.3 |
| TensorRT | 10.3.0，系统自带；`/usr/src/tensorrt/bin/trtexec` 可用 |
| TensorRT Python | 只装在系统 Python 3.10：`/usr/lib/python3.10/dist-packages/tensorrt` |
| 训练/推理环境 | conda `lerobot-train`：Python 3.10、torch 2.8（CUDA 可用）、onnx 1.20、lerobot 0.6.2。**没有** tensorrt、onnxruntime |
| 权重 | HF `yutongzhu/so101-pick-cube-clean`，板子上也有 `~/outputs/act/so101-pick-cube-clean` |
| 模型输入 | `observation.state` (6)，`wrist`、`front` 两路图像 (3, 640, 480) |
| 模型输出 | `action` (100, 6)，chunk_size=100 |

已有的实测（`eval/so101-pick-cube-clean`）：temporal ensemble 时 infer p50 = 67 ms，只能跑 13.8 Hz；30 Hz 预算是 33.3 ms。
**这次优化的意义：把一次完整 forward 压进 33 ms，让 temporal ensemble 也能跑满 30 Hz。**

注意：chunking 那组 infer p50 只有 3.5 ms，说明那组大部分 tick 只是从队列里取动作，并没有每拍都跑 forward。所以对比延迟不能只看 rollout 的 p50，要先用独立 benchmark 测「一次纯 forward」。

## 整体流程

```
PyTorch 模型 ──(1)基线计时──▶ 67 ms?
      │
      ├─(2) torch.onnx.export ──▶ act.onnx          「把计算图冻结成文件」
      │
      ├─(3) trtexec 先跑通 ──▶ act_fp16.engine      「不写代码，先看 TRT 能到多快」
      │
      ├─(4) 手写 build_engine.py ──▶ 同一个 engine   「搞懂 builder / network / config」
      │
      ├─(5) 手写 trt_infer.py ──▶ 精度对比 + 计时    「搞懂 runtime / context / 显存地址」
      │
      ├─(6) 接进 rollout.sh --trt ──▶ 真机 30 Hz?
      │
      └─(7) 真机 24 次评估 ──▶ 成功率、延迟对比图、写成技能
```

文件都放在 `deploy/trt/`，代码每个文件控制在 50 行左右。所有命令都在开发板上跑（`ssh nano0`，`conda activate lerobot-train`）。

---

## 第 1 步：PyTorch 基线 `bench_torch.py`

**你要写的**：加载策略，造一组假输入，只计 `policy.model(batch)` 这一次 forward。

要点：
- `ACTPolicy.from_pretrained(<目录>)`，`.to("cuda").eval()`。
- 输入是**已经归一化后**的张量（预处理器做完的那一步），形状 `state (1,6)`、`wrist/front (1,3,640,480)`。
- ACT 内部要的是 `batch["observation.images"] = [wrist, front]`，这一行在 `ACTPolicy.predict_action_chunk` 里（lerobot 源码 `modeling_act.py:128`），自己读一遍。
- 计时：先热身 20 次；每次前后都 `torch.cuda.synchronize()`；跑 200 次取 p50/p99。
- 包在 `torch.inference_mode()` 里。分别测 FP32 和 `torch.autocast("cuda", dtype=torch.float16)`。

**过关标准**：打印 `fp32 p50/p99` 和 `fp16 p50/p99`，fp32 应接近 rollout 里的 67 ms。把结果记下来，这就是「优化前」。

**实测（2026-10-24，MAXN_SUPER）**：FP32 p50 59.3 / p99 67.7 ms；autocast FP16 p50 46.7 / p99 52.6 ms。FP32 时 CPU 下发 41 ms，GPU 事件计时 59 ms。CPU 开销主要来自 `modeling_act.py:487-488` 的 `list(cam_features)`：每次 forward 切出约 1200 个小张量再 stack。

**要能回答**：为什么不 `synchronize` 测出来的时间会偏小？

## 第 2 步：导出 ONNX `export_onnx.py`

**你要写的**：一个 20 行的包装模块，把「字典进、元组出」变成「三个张量进、一个张量出」，然后导出。

```python
class ActForward(torch.nn.Module):
    def __init__(self, act_model):  # act_model = policy.model
        ...
    def forward(self, state, wrist, front):
        # 拼成 policy.model 要的 batch，返回 actions (1, 100, 6)
        ...
```

- `torch.onnx.export(..., opset_version=17, dynamo=False, input_names=[...], output_names=["actions"])`。固定 batch=1，先不要动态维度。
- 导完用 `onnx.checker.check_model` 检查。
- 用 [Netron](https://netron.app) 在 Mac 上打开 `act.onnx`，找到 ResNet18 的卷积、Transformer 的 MatMul/Softmax。

**过关标准**：`act.onnx` 生成，checker 不报错，Netron 里能看到三个输入、一个输出 `actions [1,100,6]`。

**要能回答**：`model.eval()` 对 ACT 的 VAE 分支和 dropout 各起了什么作用？如果忘了 eval，导出的图会多出什么？

## 第 3 步：先用 `trtexec` 跑通（不写代码）

```bash
/usr/src/tensorrt/bin/trtexec --onnx=act.onnx --saveEngine=act_fp32.engine
/usr/src/tensorrt/bin/trtexec --onnx=act.onnx --saveEngine=act_fp16.engine --fp16
```

- 读输出最后的 `GPU Compute Time` 的 median。这就是 TensorRT 的上限参考。
- 加 `--dumpLayerInfo --profilingVerbosity=detailed` 看层被融合成了什么。

**过关标准**：拿到 fp32 / fp16 两个 engine 和两个 median。

**要能回答**：engine 为什么不能从 2080 Ti 拷到 Jetson 上用？build 为什么要几分钟？

## 第 4 步：手写构建 `build_engine.py`

**你要写的**：和第 3 步做同一件事，用 Python API。大约 25 行，固定这几步：

1. `logger = trt.Logger(trt.Logger.WARNING)`
2. `builder = trt.Builder(logger)`，`network = builder.create_network(0)`
3. `parser = trt.OnnxParser(network, logger)`，`parser.parse(open(onnx, "rb").read())`，失败时把 `parser.get_error(i)` 都打出来
4. `config = builder.create_builder_config()`；`--fp16` 时 `config.set_flag(trt.BuilderFlag.FP16)`；工作区 `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)`
5. `serialized = builder.build_serialized_network(network, config)`，写到 `.engine`

**环境**：TensorRT 只在系统 Python 里。第一次先做一件事，让 conda 环境能 import 到它：

```bash
echo /usr/lib/python3.10/dist-packages/tensorrt/.. > $(python -c "import site;print(site.getsitepackages()[0])")/system_trt.pth
python -c "import tensorrt; print(tensorrt.__version__)"   # 期望 10.3.0
```

（两边都是 Python 3.10，所以能直接用。这一行会把整个系统 dist-packages 加进来；如果和 conda 里的包冲突，改成只软链 `tensorrt` 这一个目录。）

**过关标准**：生成的 engine 用 `trtexec --loadEngine=...` 测出来和第 3 步差不多。

**要能回答**：Builder、Network、Parser、BuilderConfig 各自负责什么？FP16 flag 是「强制 FP16」还是「允许 FP16」？

## 第 5 步：手写推理 `trt_infer.py`（核心）

**你要写的**：一个类，大约 35 行。不用 pycuda，显存直接用 torch 张量：

```python
class TrtAct:
    def __init__(self, engine_path):
        # runtime = trt.Runtime(logger)
        # engine = runtime.deserialize_cuda_engine(...)
        # context = engine.create_execution_context()
        # 为输出 "actions" 预先建好 torch.empty((1,100,6), device="cuda")
    def __call__(self, state, wrist, front):
        # 每个输入: context.set_tensor_address(name, tensor.contiguous().data_ptr())
        # 输出同理
        # context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        # return self.out
```

然后在同一个脚本里做两件事：
1. **精度**：同一组输入分别喂 PyTorch FP32 和 TRT，打印 `max abs diff` 和 `mean abs diff`（都在归一化空间里）。
2. **速度**：和第 1 步同样的计时方法，p50/p99。

**过关标准**：FP32 engine 的 max diff 在 1e-3 量级；FP16 一般在 1e-2 量级。速度明显快于第 1 步。如果 FP16 误差很大，先查是不是 Softmax / LayerNorm 溢出。

**要能回答**：输入张量为什么要 `.contiguous()`？dtype 不是 float32 会怎样？为什么要用 torch 当前 stream？

## 第 6 步：接进真机控制环

在 `control/log_act_inference_latency.py` 里加一个 `--trt <engine>` 参数，加载后把 `policy.model` 换成一个小包装：`forward(batch)` 里取出 state / 两张图，调 `TrtAct`，返回 `(actions, (None, None))`。预处理（归一化）和后处理（反归一化）都不用动，因为 engine 的输入输出就在归一化空间里。

```bash
bash control/rollout.sh so101-pick-cube-clean --trt=deploy/trt/act_fp16.engine \
  --policy.temporal_ensemble_coeff=0.01 --log-result=trt_ensemble.json
```

**过关标准**：终端每拍 `infer` 稳定小于 33 ms，超时率接近 0，实际频率约 30 Hz。

## 第 7 步：评估与总结

1. 真机跑 24 次 `trt_ensemble.json`，物体位置和之前两组一样。
2. 扩展 `eval/so101-pick-cube-clean/plot_eval.py`，加第三组。
3. 写一张总表：

| | PyTorch FP32 | PyTorch autocast FP16 | TRT FP32 | TRT FP16 |
|---|---:|---:|---:|---:|
| 纯 forward p50 / p99（benchmark） | | | | |
| rollout infer p50（ensemble） | | | | |
| 实际频率 | | | | |
| 成功率（24 次） | 18/24 | — | — | |

4. 把整个流程写成一页 `SKILL.md`：通用步骤（基线 → ONNX → trtexec → build → infer → 精度 → 上线）+ 这次踩过的坑。下次换 Diffusion Policy 或别的模型也能照着做。

---

## 你和我的分工

- **你**：写每一步的代码，在板子上运行，把输出贴给我。
- **我**：每一步开始前讲清概念和 API，卡住时帮你看报错，帮你 review 代码，解释数字为什么是这样。不直接替你写 TensorRT 部分。

建议节奏：第 1–2 步一次，第 3–4 步一次，第 5 步单独一次（最关键），第 6–7 步需要接真机。
