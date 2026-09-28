"""Replay one recorded SO-101 episode.

Adapted from the API example in:
https://huggingface.co/docs/lerobot/main/en/il_robots
"""

import argparse
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

FOLLOWER_PORT = "/dev/ttyACM1"
FOLLOWER_ID = "YutongZhu_awesome_follower_arm"
HF_USER = "yutongzhu"
DEFAULT_DATASET_NAME = "record-test"


def parse_args():
    parser = argparse.ArgumentParser(description="Replay one episode on the SO-101 follower.")
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name. Loaded as yutongzhu/<name>. A value containing '/' is used as the full repo id.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index to replay, starting at 0.",
    )
    return parser.parse_args()


def repo_id_from_name(dataset_name: str) -> str:
    name = dataset_name.strip().strip("/")
    if not name:
        raise SystemExit("数据集名字不能为空。")
    if "/" in name:
        return name
    return f"{HF_USER}/{name}"


def main():
    args = parse_args()
    if args.episode < 0:
        raise SystemExit("集数从 0 开始，不能是负数。")

    repo_id = repo_id_from_name(args.dataset_name)
    root = Path.home() / ".cache/huggingface/lerobot" / repo_id
    dataset = LeRobotDataset(repo_id, root=root if root.exists() else None, episodes=[args.episode])
    actions = dataset.select_columns("action")
    action_names = dataset.features["action"]["names"]

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
    robot.connect()
    try:
        log_say(f"Replaying episode {args.episode} ({dataset.num_frames} frames)")
        for idx in range(dataset.num_frames):
            t0 = time.perf_counter()
            action = {name: float(actions[idx]["action"][i]) for i, name in enumerate(action_names)}
            robot.send_action(action)
            precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
