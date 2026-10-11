"""Record an SO-101 dataset.

The control loop runs on the Jetson. --web-view writes an .rrd on the board
and opens a browser replay when the run ends. Frames are always written on
the Jetson first; --save-to mac copies the finished dataset afterwards.
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import shutdown_visualization

from rerun_view import compress_images, replay_rrd, start_rerun

FOLLOWER_PORT = "/dev/ttyACM1"
FOLLOWER_ID = "YutongZhu_awesome_follower_arm"
LEADER_PORT = "/dev/ttyACM0"
LEADER_ID = "YutongZhu_awesome_leader_arm"

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
HF_USER = "yutongzhu"
DEFAULT_DATASET_NAME = "record-test"
RERUN_IP = "192.168.18.9"
RERUN_PORT = 9876
MAC_HOST = "zyt@192.168.18.9"
MAC_DIR = "/Users/zyt/lerobot-datasets"


def parse_args():
    parser = argparse.ArgumentParser(description="Record an SO-101 dataset while watching it in Rerun.")
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name. Saved as yutongzhu/<name>. A value containing '/' is used as the full repo id.",
    )
    parser.add_argument(
        "--auto-name",
        action="store_true",
        help="Append the current time to the dataset name, for example so101-pick-cube_20260928_193000.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not open the camera. The dataset stores joint data only.",
    )
    parser.add_argument(
        "--save-to",
        choices=("board", "mac"),
        default="board",
        help="board keeps the dataset on the Jetson. mac copies it to the computer after recording.",
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
        help="Record a clip to an .rrd, then open it in a Rerun web viewer when recording ends.",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="HTTP port for --web-view. Default is 9090.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=NUM_EPISODES,
        help="How many episodes to record. Default is 5.",
    )
    parser.add_argument(
        "--reset-time",
        type=float,
        default=RESET_TIME_SEC,
        help="Seconds to wait between episodes. Press n again during the reset to skip it.",
    )
    parser.add_argument("--mac-host", default=MAC_HOST, help="SSH target used when --save-to mac.")
    parser.add_argument("--mac-dir", default=MAC_DIR, help="Directory on the computer used when --save-to mac.")
    return parser.parse_args()


def copy_dataset_to_mac(dataset_root: Path, mac_host: str, mac_dir: str, repo_id: str) -> None:
    remote_dir = f"{mac_dir.rstrip('/')}/{repo_id}"
    print(f"Copying dataset to {mac_host}:{remote_dir}")
    subprocess.run(["ssh", mac_host, "mkdir", "-p", remote_dir], check=True)
    subprocess.run(["rsync", "-az", f"{dataset_root}/", f"{mac_host}:{remote_dir}/"], check=True)
    print(f"Dataset copied to {mac_host}:{remote_dir}")


def repo_id_from_name(dataset_name: str, auto_name: bool) -> str:
    name = dataset_name.strip().strip("/")
    if not name:
        raise SystemExit("数据集名字不能为空。")
    if auto_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{stamp}"
    if "/" in name:
        return name
    return f"{HF_USER}/{name}"


def main():
    args = parse_args()
    if args.num_episodes < 1:
        raise SystemExit("录制集数至少为 1。")
    repo_id = repo_id_from_name(args.dataset_name, args.auto_name)
    print(f"Dataset id: {repo_id}")
    task_description = ""
    while not task_description:
        task_description = input("任务描述: ").strip()
    print(f"Task: {task_description}")
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

    robot = SO101Follower(robot_config)
    teleop = SO101Leader(teleop_config)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
        streaming_encoding=True,
    )
    dataset_root = Path(dataset.root)
    print(f"Recording on the Jetson at {dataset_root}")

    _, events = init_keyboard_listener()
    rrd_path = start_rerun(
        "recording",
        remote_view=args.remote_view,
        web_view=args.web_view,
        rerun_ip=args.rerun_ip,
        rerun_port=args.rerun_port,
        web_port=args.web_port,
    )
    send_compressed = compress_images(args.remote_view, args.web_view)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    timer = CycleTimer(FPS)
    completed = False

    try:
        robot.connect()
        teleop.connect()
        episode_idx = 0
        while episode_idx < args.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {episode_idx + 1} of {args.num_episodes}")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=task_description,
                display_data=True,
                display_compressed_images=send_compressed,
                timer=timer,
            )

            if not events["stop_recording"] and (episode_idx < args.num_episodes - 1 or events["rerecord_episode"]):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=args.reset_time,
                    single_task=task_description,
                    display_data=True,
                    display_compressed_images=send_compressed,
                )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                timer.log_episode_summary("discarded episode")
                timer.restart()
                continue

            dataset.save_episode()
            timer.log_episode_summary(f"episode {episode_idx + 1}")
            timer.restart()
            episode_idx += 1

        completed = True
    finally:
        timer.log_run_summary()
        if completed:
            log_say("Finalizing dataset...")
            dataset.finalize()
            if args.save_to == "mac":
                copy_dataset_to_mac(dataset_root, args.mac_host, args.mac_dir, repo_id)
            dataset.push_to_hub()
        log_say("Stop recording")
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        shutdown_visualization("rerun")
        if rrd_path is not None:
            replay_rrd(
                rrd_path,
                web_port=args.web_port,
                grpc_port=args.rerun_port,
                probe_host=args.rerun_ip,
            )


if __name__ == "__main__":
    main()
