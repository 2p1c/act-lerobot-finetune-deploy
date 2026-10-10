"""Time each ACT forward pass, then run lerobot-rollout.

Wraps ``ACTPolicy.predict_action_chunk`` so the printed number is the raw
chunk inference, not camera I/O or serial. CUDA work is synchronized
before the timer stops. Temporal ensembling itself is a cheap weighted
average and is not included here.

Usage is the same as ``lerobot-rollout``; the shell script passes through
its args.
"""

from __future__ import annotations

import atexit
import statistics
import time

import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.lerobot_rollout import main

_WARMUP_STEPS = 5
_latencies_ms: list[float] = []
_orig_predict = ACTPolicy.predict_action_chunk


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _timed_predict_action_chunk(self, batch):
    _cuda_sync()
    t0 = time.perf_counter()
    actions = _orig_predict(self, batch)
    _cuda_sync()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    step = len(_latencies_ms)
    _latencies_ms.append(dt_ms)
    tag = " warmup" if step < _WARMUP_STEPS else ""
    print(f"[act raw] step={step}{tag} {dt_ms:.1f} ms", flush=True)
    return actions


def _print_summary() -> None:
    if not _latencies_ms:
        print("[act raw] no forward passes recorded", flush=True)
        return
    steady = _latencies_ms[_WARMUP_STEPS:] or _latencies_ms
    ordered = sorted(steady)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    print(
        "[act raw] summary "
        f"n={len(steady)} (skip first {_WARMUP_STEPS}) "
        f"mean={statistics.fmean(steady):.1f} ms "
        f"p50={p50:.1f} ms p95={p95:.1f} ms "
        f"min={ordered[0]:.1f} ms max={ordered[-1]:.1f} ms "
        "(30 Hz budget is 33.3 ms)",
        flush=True,
    )


ACTPolicy.predict_action_chunk = _timed_predict_action_chunk
atexit.register(_print_summary)


if __name__ == "__main__":
    main()
