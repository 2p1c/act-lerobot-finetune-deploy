"""Time each stage of the ACT control loop, then run lerobot-rollout.

Splits one tick the same way as a bare control loop:

    capture  robot.get_observation
    pre      observation processor + policy preprocessor
    infer    policy.select_action + cuda.synchronize
    send     postprocess + interpolator + robot.send_action
    loop     capture start through send end (sleep is not included)

Usage is the same as ``lerobot-rollout``; the shell script passes through
its args.
"""

from __future__ import annotations

import atexit
import logging
import time
from contextlib import nullcontext
from copy import copy

import numpy as np
import torch

from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action
from lerobot.scripts.lerobot_rollout import main
from lerobot.utils.robot_utils import precise_sleep

_WARMUP_STEPS = 5
_KEYS = ("capture", "pre", "infer", "send", "loop")
_rec = {k: [] for k in _KEYS}
_t = {}
_fps = 30.0

logger = logging.getLogger("lerobot.rollout.strategies.base")


def _ms(a: float, b: float) -> float:
    return (b - a) * 1000.0


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
    if obs_frame is None:
        return None
    observation = copy(obs_frame)
    autocast_ctx = (
        torch.autocast(device_type=self._device.type)
        if self._device.type == "cuda" and self._policy.config.use_amp
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        observation = prepare_observation_for_inference(
            observation, self._device, self._task, self._robot_type
        )
        observation = self._preprocessor(observation)
        _t["t2"] = time.perf_counter()
        action = self._policy.select_action(observation)
        _cuda_sync()
        _t["t3"] = time.perf_counter()
        action = self._postprocessor(action)
    action_tensor = action.squeeze(0).cpu()
    action_dict = make_robot_action(action_tensor, self._dataset_features)
    return torch.tensor([action_dict[k] for k in self._ordered_action_keys])


def _timed_run(self, ctx) -> None:
    global _fps
    engine = self._engine
    cfg = ctx.runtime.cfg
    robot = ctx.hardware.robot_wrapper
    interpolator = self._interpolator
    control_interval = interpolator.get_control_interval(cfg.fps)
    _fps = float(cfg.fps)

    start_time = time.perf_counter()
    engine.resume()
    logger.info("Base strategy control loop started")

    while not ctx.runtime.shutdown_event.is_set():
        loop_start = time.perf_counter()
        _t.clear()
        _t["t0"] = loop_start

        if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
            logger.info("Duration limit reached (%.0fs)", cfg.duration)
            break

        obs = robot.get_observation()
        _t["t1"] = time.perf_counter()
        obs_processed = self._process_observation_and_notify(ctx.processors, obs)

        if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
            continue

        action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
        _t["t4"] = time.perf_counter()
        self._log_telemetry(obs_processed, action_dict, ctx.runtime)

        if {"t0", "t1", "t2", "t3", "t4"} <= _t.keys():
            sample = {
                "capture": _ms(_t["t0"], _t["t1"]),
                "pre": _ms(_t["t1"], _t["t2"]),
                "infer": _ms(_t["t2"], _t["t3"]),
                "send": _ms(_t["t3"], _t["t4"]),
                "loop": _ms(_t["t0"], _t["t4"]),
            }
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

        dt = time.perf_counter() - loop_start
        if (sleep_t := control_interval - dt) > 0:
            precise_sleep(sleep_t)
        else:
            logger.warning(
                f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
            )


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


SyncInferenceEngine.get_action = _timed_get_action
BaseStrategy.run = _timed_run
atexit.register(_print_summary)


if __name__ == "__main__":
    main()
