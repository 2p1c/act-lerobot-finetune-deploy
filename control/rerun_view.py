"""Start Rerun as a local window, a remote native viewer, or a web page."""

from __future__ import annotations

import os
import socket
import threading
import time
from urllib.parse import quote

import numpy as np

from lerobot.utils import rerun_visualization, visualization_utils
from lerobot.utils.rerun_visualization import log_rerun_data
from lerobot.utils.visualization_utils import init_visualization

DEFAULT_WEB_PORT = 9090
# Joints stay at the control rate. Only camera frames are limited.
WEB_IMAGE_HZ = 1.0
WEB_IMAGE_STRIDE = 4  # 640x480 -> 160x120


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
) -> None:
    if remote_view and web_view:
        raise SystemExit("不能同时使用 --remote-view 和 --web-view。")
    if web_view:
        _install_viz_throttle(WEB_IMAGE_HZ)
        _start_web(session_name, grpc_port=rerun_port, web_port=web_port, probe_host=rerun_ip)
        return

    if remote_view:
        print(f"Sending Rerun view to {rerun_ip}:{rerun_port}")
        init_visualization("rerun", session_name=session_name, ip=rerun_ip, port=rerun_port)
        return

    os.environ.setdefault("DISPLAY", ":1")
    print(f"Opening Rerun on this machine (DISPLAY={os.environ['DISPLAY']})")
    init_visualization("rerun", session_name=session_name)


def _start_web(session_name: str, grpc_port: int, web_port: int, probe_host: str) -> None:
    import rerun as rr

    log_rerun_data.blueprint = None
    os.environ.setdefault("RERUN_FLUSH_NUM_BYTES", "8000")
    rr.init(session_name)
    host = lan_ip(probe_host)
    origin = f"http://{host}:{web_port}"
    # Browser page is :web_port, data is :grpc_port. Different ports are different
    # origins; Rerun otherwise only allows localhost and rerun.io.
    rr.serve_grpc(
        grpc_port=grpc_port,
        server_memory_limit="64MiB",
        newest_first=True,
        cors_allow_origin=[origin, f"http://{host}:*", "http://127.0.0.1:*", "*"],
    )
    connect_uri = f"rerun+http://{host}:{grpc_port}/proxy"
    rr.serve_web_viewer(open_browser=False, web_port=web_port, connect_to=connect_uri)
    page = f"http://{host}:{web_port}/?url={quote(connect_uri, safe='')}"
    print(f"Rerun web viewer: {page}")
    print(f"Joints at control rate; images {WEB_IMAGE_HZ:.0f} Hz JPEG 160x120.")
    print("Open that URL on the computer. Do not start a native Rerun window.")


def _split_observation(observation):
    joints = {}
    images = {}
    if not observation:
        return joints, images
    for key, value in observation.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            images[key] = value
        else:
            joints[key] = value
    return joints, images


def _shrink_images(images):
    if not images:
        return images
    out = {}
    for key, value in images.items():
        out[key] = np.ascontiguousarray(value[::WEB_IMAGE_STRIDE, ::WEB_IMAGE_STRIDE])
    return out


def _install_viz_throttle(image_hz: float) -> None:
    period = 1.0 / image_hz
    last_image = {"t": 0.0}
    first = {"done": False}
    busy = threading.Lock()
    orig = rerun_visualization.log_rerun_data

    def send_images(images):
        try:
            orig(observation=images, action=None, compress_images=True)
        finally:
            busy.release()

    def throttled(*args, **kwargs):
        observation = kwargs.get("observation", args[0] if args else None)
        action = kwargs.get("action", args[1] if len(args) > 1 else None)
        joints, images = _split_observation(observation)

        if not first["done"]:
            first_obs = dict(joints)
            first_obs.update(_shrink_images(images))
            orig(
                observation=first_obs or None,
                action=action,
                compress_images=True,
            )
            first["done"] = True
            last_image["t"] = time.perf_counter()
            return

        orig(observation=joints or None, action=action, compress_images=False)

        now = time.perf_counter()
        if not images or now - last_image["t"] < period:
            return
        if not busy.acquire(blocking=False):
            return
        last_image["t"] = now
        threading.Thread(target=send_images, args=(_shrink_images(images),), daemon=True).start()

    rerun_visualization.log_rerun_data = throttled
    visualization_utils.log_rerun_data = throttled
