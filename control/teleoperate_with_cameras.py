"""Teleoperate SO-101 with cameras.

Adapted from the API example in:
https://huggingface.co/docs/lerobot/main/en/il_robots
"""

import argparse
import time

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.visualization_utils import log_visualization_data, shutdown_visualization

from rerun_view import compress_images, replay_rrd, start_rerun

FOLLOWER_PORT = "/dev/ttyACM1"
FOLLOWER_ID = "YutongZhu_awesome_follower_arm"
LEADER_PORT = "/dev/ttyACM0"
LEADER_ID = "YutongZhu_awesome_leader_arm"

TARGET_HZ = 30
TIME_PER_FRAME = 1.0 / TARGET_HZ
RERUN_IP = "192.168.18.9"
RERUN_PORT = 9876


def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate the SO-101 follower from the leader arm.")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not open the camera. Rerun still shows joint positions.",
    )
    parser.add_argument("--rerun-ip", default=RERUN_IP, help="IP of the computer running the Rerun viewer.")
    parser.add_argument("--rerun-port", type=int, default=RERUN_PORT)
    view = parser.add_mutually_exclusive_group()
    view.add_argument(
        "--remote-view",
        action="store_true",
        help="Send the Rerun view to --rerun-ip instead of opening a window on this machine.",
    )
    view.add_argument(
        "--web-view",
        action="store_true",
        help="Record a clip to an .rrd, then open it in a Rerun web viewer.",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="HTTP port for --web-view. Default is 9090.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds to run. Default is 60 with --web-view, otherwise until Ctrl+C.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cameras = {}
    if not args.no_camera:
        cameras = {
            "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
            "front": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=30),
        }
    robot_config = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras=cameras,
    )
    teleop_config = SO101LeaderConfig(
        port=LEADER_PORT,
        id=LEADER_ID,
    )

    rrd_path = start_rerun(
        "teleoperation",
        remote_view=args.remote_view,
        web_view=args.web_view,
        rerun_ip=args.rerun_ip,
        rerun_port=args.rerun_port,
        web_port=args.web_port,
    )
    duration = args.duration
    if duration is None:
        duration = 60.0 if args.web_view else 0.0
    if duration > 0:
        print(f"Recording {duration:.0f}s, then Rerun replay. Ctrl+C to stop early.")

    robot = SO101Follower(robot_config)
    teleop_device = SO101Leader(teleop_config)
    t_end = None if duration <= 0 else time.perf_counter() + duration
    try:
        robot.connect()
        teleop_device.connect()
        while t_end is None or time.perf_counter() < t_end:
            start_time = time.perf_counter()

            observation = robot.get_observation()
            action = teleop_device.get_action()
            robot.send_action(action)
            log_visualization_data(
                "rerun",
                observation=observation,
                action=action,
                compress_images=compress_images(args.remote_view, args.web_view),
            )

            elapsed_time = time.perf_counter() - start_time
            sleep_time = TIME_PER_FRAME - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        if teleop_device.is_connected:
            teleop_device.disconnect()
        if robot.is_connected:
            robot.disconnect()
        shutdown_visualization("rerun")
        if rrd_path is not None:
            replay_rrd(rrd_path, web_port=args.web_port, probe_host=args.rerun_ip)


if __name__ == "__main__":
    main()
