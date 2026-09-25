"""Teleoperate SO-101 with cameras.

Adapted from the API example in:
https://huggingface.co/docs/lerobot/main/en/il_robots
"""

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


def main():
    robot_config = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras={
            "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
            "top": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
        },
    )
    teleop_config = SO101LeaderConfig(
        port=LEADER_PORT,
        id=LEADER_ID,
    )

    # pass "foxglove" to stream to Foxglove instead
    init_visualization("rerun", session_name="teleoperation")

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
            log_visualization_data("rerun", observation=observation, action=action)

            elapsed_time = time.perf_counter() - start_time
            sleep_time = TIME_PER_FRAME - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_visualization("rerun")
        teleop_device.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
