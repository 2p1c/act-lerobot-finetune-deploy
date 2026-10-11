"""Start Rerun as a local window, a remote native viewer, or a post-run replay."""

from __future__ import annotations

import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path

from lerobot.utils.rerun_visualization import log_rerun_data
from lerobot.utils.visualization_utils import init_visualization

DEFAULT_WEB_PORT = 9090
RRD_DIR = Path.home() / "lerobot-rerun"


def lan_ip(probe_host: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((probe_host, 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def compress_images(remote_view: bool, web_view: bool) -> bool:
    return remote_view or web_view


def start_rerun(
    session_name: str,
    *,
    remote_view: bool = False,
    web_view: bool = False,
    rerun_ip: str = "192.168.18.9",
    rerun_port: int = 9876,
    web_port: int = DEFAULT_WEB_PORT,
) -> Path | None:
    if remote_view and web_view:
        raise SystemExit("不能同时使用 --remote-view 和 --web-view。")
    if web_view:
        return _start_save(session_name)

    if remote_view:
        print(f"Sending Rerun view to {rerun_ip}:{rerun_port}")
        init_visualization("rerun", session_name=session_name, ip=rerun_ip, port=rerun_port)
        return None

    os.environ.setdefault("DISPLAY", ":1")
    print(f"Opening Rerun on this machine (DISPLAY={os.environ['DISPLAY']})")
    init_visualization("rerun", session_name=session_name)
    return None


def replay_rrd(
    path: Path,
    *,
    web_port: int = DEFAULT_WEB_PORT,
    probe_host: str = "192.168.18.9",
) -> None:
    if not path.is_file():
        print(f"No Rerun clip to review: {path}")
        return
    host = lan_ip(probe_host)
    page = f"http://{host}:{web_port}"
    print(f"Review clip: {page}")
    print(f"File: {path}")
    print("Ctrl+C to stop the viewer.")
    subprocess.run(
        [
            "rerun",
            str(path),
            "--web-viewer",
            "--bind",
            "0.0.0.0",
            "--web-viewer-port",
            str(web_port),
        ],
        check=False,
    )


def _start_save(session_name: str) -> Path:
    import rerun as rr

    log_rerun_data.blueprint = None
    os.environ.setdefault("RERUN_FLUSH_NUM_BYTES", "8000")
    rr.init(session_name)
    orig_log = rr.log

    def log_timed(*args, **kwargs):
        if kwargs.get("static"):
            kwargs = {**kwargs, "static": False}
        return orig_log(*args, **kwargs)

    rr.log = log_timed
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RRD_DIR / f"{session_name}_{stamp}.rrd"
    path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(path))
    print(f"Recording this clip to {path}")
    print("Live view is off. Rerun opens after the clip ends, joints and images at 30 Hz.")
    return path
