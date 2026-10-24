import argparse
import os
import time

import numpy as np
import torch
import tensorrt as trt

from lerobot.policies.act.modeling_act import ACTPolicy

POLICY = os.path.expanduser(
  "~/outputs/act/so101-pick-cube-clean/checkpoints/last/pretrained_model"
)

class TrtAct:
  def __init__(self, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)

    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
      self.engine = runtime.deserialize_cuda_engine(f.read())
    if self.engine is None:
      raise RuntimeError("engine 加载失败：" + engine_path)

    self.context = self.engine.create_execution_context()

    for i in range(self.engine.num_io_tensors):
      name = self.engine.get_tensor_name(i)
      print(
        self.engine.get_tensor_mode(name),
        name,
        tuple(self.engine.get_tensor_shape(name)),
        self.engine.get_tensor_dtype(name),
      )

    self.out = torch.empty((1, 100, 6), dtype=torch.float32, device="cuda")
    self.context.set_tensor_address("actions", self.out.data_ptr())

    self.stream = torch.cuda.Stream()

  def __call__(self, state, wrist, front):
    inputs = {"state": state, "wrist": wrist, "front": front}
    kept = []
    for name, x in inputs.items():
      x = x.contiguous()
      assert x.is_cuda and x.dtype == torch.float32, name
      self.context.set_tensor_address(name, x.data_ptr())
      kept.append(x)
    self._inputs = kept

    self.stream.wait_stream(torch.cuda.current_stream())
    ok = self.context.execute_async_v3(self.stream.cuda_stream)
    if not ok:
      raise RuntimeError("execute_async_v3 失败")
    torch.cuda.current_stream().wait_stream(self.stream)
    return self.out

def bench(fn, n=200, warmup=20):
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  times = []
  for _ in range(n):
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)
  return np.percentile(times, 50), np.percentile(times, 99)

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--engine", required=True)
  args = ap.parse_args()

  torch.manual_seed(0)
  state = torch.randn(1, 6, device="cuda")
  wrist = torch.randn(1, 3, 640, 480, device="cuda")
  front = torch.randn(1, 3, 640, 480, device="cuda")

  policy = ACTPolicy.from_pretrained(POLICY)
  policy.to("cuda")
  policy.eval()
  batch = {"observation.state": state, "oberservation.images": [wrist, front]}

  trt_act = TrtAct(args.engine)

  with torch.inference_mode():
    ref = policy.model(batch)[0]
    out = trt_act(state, wrist, front).clone()
    torch.cuda.synchronize()

    diff = (ref - out).abs()
    print(f"max abs diff  = {diff.max().item():.6f}")
    print(f"mean abs diff = {diff.mean().item():.6f}")
    print(f"参考输出范围  = [{ref.min().item():.3f}, {ref.max().item():.3f}]")
    print("有 nan:", torch.isnan(out).any().item())

    # ② 速度
    p50, p99 = bench(lambda: policy.model(batch)[0])
    print(f"PyTorch FP32  p50={p50:.1f} ms p99={p99:.1f} ms")
    p50, p99 = bench(lambda: trt_act(state, wrist, front))
    print(f"TensorRT      p50={p50:.1f} ms p99={p99:.1f} ms")

if __name__ == "__main__":
  main()


    