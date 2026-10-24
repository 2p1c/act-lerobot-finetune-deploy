import os 
import time

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from torch.profiler import profile, ProfilerActivity

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

def profiler(fn, warmup=20):
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()

  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])    as prof:
    for _ in range(10):
        run()
    torch.cuda.synchronize()

  print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
  

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
  p1  = np.percentile(times, 1)
  p2  = np.percentile(times, 2)
  p5  = np.percentile(times, 5)
  p50 = np.percentile(times, 50)
  p99 = np.percentile(times, 99)
  return p50, p99, p1, p2, p5

def bench_split(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    cpu_ms, gpu_ms = [], []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        t0 = time.perf_counter()
        start.record()          
        fn()
        end.record()            
        t1 = time.perf_counter()  

        torch.cuda.synchronize()
        cpu_ms.append((t1 - t0) * 1000)
        gpu_ms.append(start.elapsed_time(end))   

    print(f"CPU 下发  p50={np.percentile(cpu_ms, 50):.1f} ms")
    print(f"GPU 执行  p50={np.percentile(gpu_ms, 50):.1f} ms")

def main():
  with torch.inference_mode():
    out = run()
    print("输出形状：", tuple(out.shape))

    profiler(run)
    bench_split(run)
    p50, p99, p1, p2, p5 = bench(run)
    print(f"FP32 p50={p50:.1f} ms p99={p99:.1f} ms p1={p1:.1f} ms p2={p2:.1f} ms p5={p5:.1f} ms")

    with torch.autocast("cuda", dtype=torch.float16):
      profiler(run)
      bench_split(run)
      p50, p99, p1, p2, p5 = bench(run)
    print(f"FP16 p50={p50:.1f} ms p99={p99:.1f} ms p1={p1:.1f} ms p2={p2:.1f} ms p5={p5:.1f} ms")

if __name__ == "__main__":
  main()


    
