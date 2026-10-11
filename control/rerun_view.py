"""Start Rerun as a local window, a remote native viewer, or a web page."""

from __future__ import annotations

import os
import socket
from urllib.parse import quote

from lerobot.utils.rerun_visualization import log_rerun_data
from lerobot.utils.visualization_utils import init_visualization

DEFAULT_WEB_PORT = 9090


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
    rr.serve_grpc(grpc_port=grpc_port)
    connect_uri = f"rerun+http://{host}:{grpc_port}/proxy"
    rr.serve_web_viewer(open_browser=False, web_port=web_port, connect_to=connect_uri)
    page = f"http://{host}:{web_port}/?url={quote(connect_uri, safe='')}"
    print(f"Rerun web viewer: {page}")
    print("Open that URL on the computer. Do not start a native Rerun window.")
