import os 
import time

import numpy as np
import torch

form lerobot.policies.act.modeling_act import ACTPolicy

POLICY = os.path.expanduser("~/outputs/act/so101-pick-cube-clean/checkpoints/last/pretrained_model")

policy = ACTPolicy.from_pretrained(POLICY)
policy.to("cuda")
policy.eval()

state = torch.randn(1, 6, device="cuda")
wrist = torch.randn(1, 3, 640, 480, device="cuda")
front = torch.randn(1, 3, 640, 480, device="cuda")

batch = {
  "observation.state": state,
  "observation.images": [wrist, front],
}

def run():
  return policy.model(batch)[0]

def bench(fn, n=200, warmup=20):

  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()

  times = []
  for _ in range(n):
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

  p50 = np.percentile(times, 50)
  p99 = np.percentile(times, 99)
  return p50, p99

def main():
  with torch.inference_mode():
    out = run()
    print("输出形状：", tuple(out.shape))

    p50, p99 = bench(run)
    print(f"FP32 p50={p50:.1f} ms p99={p99:.1f} ms")

    with torch.autocast("cuda", dtype=torch.float16):
      p50, p99 = bench(run)
    print(f"FP16 p50={p50:.1f} ms p99={p99:.1f} ms")

if __name__ == "__main__":
  main()


    