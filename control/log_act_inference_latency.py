"""Time each stage of the ACT control loop, then run lerobot-rollout.

Wraps the board's own rollout loop instead of copying a newer control
loop. Splits one tick the same way as a bare control loop:

    capture  robot.get_observation
    pre      observation processor + policy preprocessor
    infer    policy.select_action + cuda.synchronize
    send     postprocess + interpolator + robot.send_action
    loop     capture start through send end (sleep is not included)

Usage is the same as ``lerobot-rollout``; the shell script passes through
its args.
"""

from __future__ import annotations

import argparse
import atexit
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.scripts.lerobot_rollout import main

_WARMUP_STEPS = 5
_KEYS = ("capture", "pre", "infer", "send", "loop")
_rec = {k: [] for k in _KEYS}
_t = {}
_fps = 30.0
_fp16 = False

_orig_run = BaseStrategy.run
_orig_select_action = ACTPolicy.select_action


def _ms(a: float, b: float) -> float:
    return (b - a) * 1000.0


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _record_tick() -> None:
    if not {"t0", "t1", "t2", "t3", "t4"} <= _t.keys():
        return
    sample = {
        "capture": _ms(_t["t0"], _t["t1"]),
        "pre": _ms(_t["t1"], _t["t2"]),
        "infer": _ms(_t["t2"], _t["t3"]),
        "send": _ms(_t["t3"], _t["t4"]),
        "loop": _ms(_t["t0"], _t["t4"]),
    }
    _t.clear()
    for key, value in sample.items():
        _rec[key].append(value)
    step = len(_rec["loop"]) - 1
    tag = " warmup" if step < _WARMUP_STEPS else ""
    print(
        f"[tick {step}]{tag} "
        f"capture={sample['capture']:.1f} "
        f"pre={sample['pre']:.1f} "
        f"infer={sample['infer']:.1f} "
        f"send={sample['send']:.1f} "
        f"loop={sample['loop']:.1f} ms",
        flush=True,
    )


def _infer_context():
    if _fp16:
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _timed_select_action(self, batch):
    _t["t2"] = time.perf_counter()
    with _infer_context():
        action = _orig_select_action(self, batch)
    _cuda_sync()
    _t["t3"] = time.perf_counter()
    return action


def _timed_run(self, ctx):
    global _fps
    _fps = float(ctx.runtime.cfg.fps)
    robot = ctx.hardware.robot_wrapper
    orig_get_obs = robot.get_observation
    orig_send = robot.send_action

    def timed_get_observation(*args, **kwargs):
        _t.clear()
        _t["t0"] = time.perf_counter()
        obs = orig_get_obs(*args, **kwargs)
        _t["t1"] = time.perf_counter()
        return obs

    def timed_send_action(*args, **kwargs):
        result = orig_send(*args, **kwargs)
        _t["t4"] = time.perf_counter()
        _record_tick()
        return result

    robot.get_observation = timed_get_observation
    robot.send_action = timed_send_action
    try:
        return _orig_run(self, ctx)
    finally:
        robot.get_observation = orig_get_obs
        robot.send_action = orig_send


def _print_summary() -> None:
    if not _rec["loop"]:
        print("[tick] no control ticks recorded", flush=True)
        return
    period_ms = 1000.0 / _fps
    rec = {k: np.array(v[_WARMUP_STEPS:] or v) for k, v in _rec.items()}
    print(
        {k: np.percentile(v, [50, 95, 99]).round(1).tolist() for k, v in rec.items()},
        flush=True,
    )
    loop = rec["loop"]
    print(
        "超时率:",
        float((loop > period_ms).mean()),
        "实际频率约:",
        1000.0 / max(float(np.mean(loop)), period_ms),
        flush=True,
    )


ACTPolicy.select_action = _timed_select_action
BaseStrategy.run = _timed_run
atexit.register(_print_summary)


def _parse_wrapper_args() -> None:
    global _fp16
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run ACT select_action under torch.autocast(cuda, float16).",
    )
    args, remaining = parser.parse_known_args()
    _fp16 = args.fp16
    sys.argv = [sys.argv[0], *remaining]
    if _fp16:
        print("[tick] FP16 autocast enabled for select_action", flush=True)


if __name__ == "__main__":
    _parse_wrapper_args()
    main()
