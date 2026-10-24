import argparse
import os

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from trt_infer import TrtAct

POLICY = os.path.expanduser(
    "~/outputs/act/so101-pick-cube-clean/checkpoints/last/pretrained_model"
)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--frames", type=int, default=50)
    args = ap.parse_args()

    policy = ACTPolicy.from_pretrained(POLICY)
    policy.to("cuda")
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=POLICY)
    trt_act = TrtAct(args.engine)
    ds = LeRobotDataset("yutongzhu/so101-pick-cube-clean")

    norm_max, deg_all = [], []
    step = max(1, len(ds) // args.frames)
    with torch.inference_mode():
        for n, i in enumerate(range(0, len(ds), step)):
            batch = pre(ds[i])
            if n == 0:
                img = batch["observation.images.wrist"]
                print("wrist:", tuple(img.shape), img.dtype, img.device)

            ref = policy.predict_action_chunk(batch)              # (1, 100, 6)
            out = trt_act(
                batch["observation.state"],
                batch["observation.images.wrist"],
                batch["observation.images.front"],
            ).clone()
            torch.cuda.synchronize()

            norm_max.append((ref - out).abs().max().item())
            # 把 100 步当成 batch=100 的动作送进后处理器，反归一化成角度
            deg = (post(ref[0]) - post(out[0])).abs().cpu()       # (100, 6)
            deg_all.append(deg)

    deg_all = torch.cat(deg_all)                                  # (帧数*100, 6)
    print(f"帧数: {len(norm_max)}")
    print(f"归一化空间 max diff: {max(norm_max):.4f}")
    print(f"关节误差  mean={deg_all.mean():.3f}°  max={deg_all.max():.3f}°")
    for j, name in enumerate(JOINTS):
        print(f"  {name:14s} mean={deg_all[:, j].mean():.3f}  max={deg_all[:, j].max():.3f}")


if __name__ == "__main__":
    main()