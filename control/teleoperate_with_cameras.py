"""Teleoperate SO-101 with cameras.

Adapted from the API example in:
https://huggingface.co/docs/lerobot/main/en/il_robots
"""

import argparse
import os
import time

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)

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
    parser.add_argument(
        "--remote-view",
        action="store_true",
        help="Send the Rerun view to --rerun-ip instead of opening a window on this machine.",
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

    rerun_ip = args.rerun_ip if args.remote_view else None
    rerun_port = args.rerun_port if args.remote_view else None
    if args.remote_view:
        print(f"Sending Rerun view to {rerun_ip}:{rerun_port}")
    else:
        os.environ.setdefault("DISPLAY", ":1")
        print(f"Opening Rerun on this machine (DISPLAY={os.environ['DISPLAY']})")
    init_visualization(
        "rerun",
        session_name="teleoperation",
        ip=rerun_ip,
        port=rerun_port,
    )

    robot = SO101Follower(robot_config)
    teleop_device = SO101Leader(teleop_config)
    robot.connect()
    teleop_device.connect()

    try:
        while True:
            start_time = time.perf_counter()

            observation = robot.get_observation()
            action = teleop_device.get_action()
            robot.send_action(action)
            log_visualization_data(
                "rerun",
                observation=observation,
                action=action,
                compress_images=args.remote_view,
            )

            elapsed_time = time.perf_counter() - start_time
            sleep_time = TIME_PER_FRAME - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        teleop_device.disconnect()
        robot.disconnect()
        shutdown_visualization("rerun")


if __name__ == "__main__":
    main()
