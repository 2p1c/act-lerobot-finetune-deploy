"""Mirror lerobot-train metrics to TensorBoard and serve them on the LAN.

LeRobot already prints, every log_freq steps:

    loss      training loss
    data_s    batch load + preprocess
    updt_s    forward + backward + optimizer
    step_s    data_s + updt_s (this wrapper)

This wrapper writes those scalars and starts TensorBoard on port 6006.
Usage is the same as ``lerobot-train``; train.sh passes through its args.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]

_PROBE_HOST = "192.168.18.9"
_TB_PORT = 6006
_writer = None
_last_step = -1

_orig_info = logging.Logger.info

try:
    from lerobot.utils.logging_utils import MetricsTracker
except ImportError:  # pragma: no cover
    MetricsTracker = ()  # type: ignore[assignment,misc]


def lan_ip(probe_host: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((probe_host, 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _output_dir_from_argv() -> Path:
    args = sys.argv
    for i, token in enumerate(args):
        if token == "--output_dir" and i + 1 < len(args):
            return Path(args[i + 1])
        if token.startswith("--output_dir="):
            return Path(token.split("=", 1)[1])
    return Path("outputs/train")


def _rename(key: str) -> str:
    return {
        "grdn": "grad_norm",
        "updt_s": "update_s",
        "dataloading_s": "data_s",
    }.get(key, key)


def _write(step: int, values: dict[str, float]) -> None:
    global _last_step
    if _writer is None or step == _last_step:
        return
    _last_step = step
    data_s = values.get("data_s")
    update_s = values.get("update_s")
    if data_s is not None and update_s is not None:
        values = {**values, "step_s": data_s + update_s}
    groups = {
        "loss": ("loss", "grad_norm"),
        "latency": ("data_s", "update_s", "step_s"),
        "opt": ("lr",),
        "progress": ("samples", "episodes", "epochs"),
    }
    grouped = {item for items in groups.values() for item in items}
    for group, keys in groups.items():
        for key in keys:
            if key in values:
                _writer.add_scalar(f"{group}/{key}", values[key], step)
    for key, value in values.items():
        if key not in grouped and key != "steps":
            _writer.add_scalar(f"train/{key}", value, step)
    _writer.flush()


def _from_tracker(tracker: object) -> None:
    raw = tracker.to_dict()  # type: ignore[attr-defined]
    values = {_rename(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    step = int(values.get("steps", 0))
    _write(step, values)


def _from_line(text: str) -> None:
    if "step:" not in text or "loss:" not in text:
        return
    parsed: dict[str, float] = {}
    for match in re.finditer(r"([A-Za-z_]+):([0-9.+-eE]+)", text):
        parsed[_rename(match.group(1))] = float(match.group(2))
    if "step" not in parsed or "loss" not in parsed:
        return
    step = int(parsed.pop("step"))
    _write(step, parsed)


def _info(self: logging.Logger, msg: object, *args: object, **kwargs: object) -> None:
    if MetricsTracker and isinstance(msg, MetricsTracker):
        _from_tracker(msg)
    elif isinstance(msg, str):
        _from_line(msg)
    return _orig_info(self, msg, *args, **kwargs)


def _clear_proxy(env: dict[str, str]) -> dict[str, str]:
    cleaned = env.copy()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        cleaned.pop(key, None)
    cleaned["NO_PROXY"] = "*"
    cleaned["no_proxy"] = "*"
    return cleaned


def _start_tensorboard(logdir: Path) -> subprocess.Popen | None:
    log_path = logdir / "tensorboard.log"
    cmd = [
        sys.executable,
        "-m",
        "tensorboard",
        "--logdir",
        str(logdir),
        "--port",
        str(_TB_PORT),
        "--host",
        "0.0.0.0",
        "--reload_interval",
        "5",
    ]
    try:
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            env=_clear_proxy(os.environ),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        print(f"TensorBoard 没启动: {exc}", flush=True)
        print(f"日志在 {logdir}", flush=True)
        return None
    time.sleep(1.5)
    if proc.poll() is not None:
        print(log_path.read_text(encoding="utf-8")[-2000:], flush=True)
        print(f"TensorBoard 没启动。日志在 {logdir}", flush=True)
        return None
    host = lan_ip(_PROBE_HOST)
    print(f"TensorBoard: http://{host}:{_TB_PORT}/", flush=True)
    print(f"Scalars: {logdir}", flush=True)
    return proc


def main() -> None:
    global _writer
    output_dir = _output_dir_from_argv()
    tb_dir = output_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    if SummaryWriter is None:
        print("未安装 tensorboard，只训练、不画曲线。需要时: pip install tensorboard", flush=True)
    else:
        _writer = SummaryWriter(log_dir=str(tb_dir))
    logging.Logger.info = _info  # type: ignore[method-assign]
    tb_proc = _start_tensorboard(tb_dir) if _writer is not None else None
    from lerobot.scripts.lerobot_train import main as train_main

    try:
        train_main()
    finally:
        if _writer is not None:
            _writer.flush()
            _writer.close()
        if tb_proc is not None and tb_proc.poll() is None:
            tb_proc.terminate()


if __name__ == "__main__":
    main()
