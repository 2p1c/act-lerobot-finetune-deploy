# 录制数据集

在开发板上采集 SO-101 数据，在自己的电脑上用 Rerun 看实时画面。舵机和相机都在开发板上，所以 `record.py` 必须在开发板的 SSH 终端里跑。

## 硬件

| 设备 | 端口 | ID |
|------|------|-----|
| Follower（被带动的臂） | `/dev/ttyACM1` | `YutongZhu_awesome_follower_arm` |
| Leader（用手掰的臂） | `/dev/ttyACM0` | `YutongZhu_awesome_leader_arm` |
| 腕部相机 | 索引 0，640×480 | 数据集里的名字是 `wrist` |
| 场景相机 | `/dev/video2`，640×480 | 数据集里的名字是 `front` |

先确认 USB 没有对调：

```bash
ls -l /dev/ttyACM*
```

## 1. 用浏览器看画面

两路相机推到电脑上的原生 Rerun 窗口容易卡住。开发板自己开网页 Viewer，电脑用浏览器打开终端里打印的地址：

```bash
python record.py --dataset-name so101-pick-cube --web-view
python teleoperate_with_cameras.py --web-view
```

地址类似 `http://192.168.18.x:9090/?url=...`。电脑上不用再开 `rerun` 窗口。推荐 Chrome / Edge。

走 `--remote-view` 才需要电脑上先开原生 Viewer，版本要和开发板一致（0.33.1）：

```bash
pipx install --backend pip rerun-sdk==0.33.1
rerun --bind 0.0.0.0 --port 9876
```

## 2. 在开发板上进入环境

```bash
ssh nano0
conda activate lerobot
cd ~/Projects/pi0-lerobot-finetune-deploy/control
```

录制过程中这个 SSH 窗口要保持在前台，按键才会生效。

`--web-view` 不需要关代理，电脑上也先不用开 Rerun。只有 `--remote-view` 才必须先让局域网直连，否则开发板代理会把发往电脑的连接转到 `127.0.0.1:7897`：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

连上之后，相机画面只显示最新一帧，不能往回拖；关节曲线可以回放。

## 3. 需要时先校准

校准文件在：

```text
~/.cache/huggingface/lerobot/calibration/robots/so_follower/YutongZhu_awesome_follower_arm.json
~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/YutongZhu_awesome_leader_arm.json
```

已有校准且和舵机一致时，脚本不会重新校准。要重做时，把这两个文件移走，再运行：

```bash
python teleoperate_with_cameras.py --no-camera
```

先校准 follower（`/dev/ttyACM1`），再校准 leader（`/dev/ttyACM0`）。每条手臂：

1. 把关节摆到行程中间，按回车。
2. 除 `wrist_roll` 外，每个关节都转到两端极限。
3. 再按回车保存。

Follower 校准结束后会打开力矩，手臂可能自己动一下。Leader 校准时力矩是关的，可以用手掰。

## 4. 录制

同一位置的每一集，都从同一种初始状态开始：手臂回到起始姿势，物体放回同一位置，相机固定不动。

```bash
python record.py --dataset-name so101-pick-cube --web-view
```

常用参数：

```bash
# 不录相机，只录关节
python record.py --dataset-name so101-pick-cube --no-camera

# 电脑用浏览器看画面
python record.py --dataset-name so101-pick-cube --web-view

# 集与集之间的重置时间，默认 10 秒
python record.py --dataset-name so101-pick-cube --reset-time 2

# 录完后再拷到电脑
python record.py --dataset-name so101-pick-cube --save-to mac
```

默认一共 5 集，每集最多 60 秒。集数和单集时长在 `record.py` 顶部的 `NUM_EPISODES`、`EPISODE_TIME_SEC`。

数据集名字会加上用户名 `yutongzhu`。上面的例子保存为 `yutongzhu/so101-pick-cube`。名字里如果自己带了 `/`，就按完整仓库 ID 使用。

同一个名字的目录已经存在时，脚本会拒绝覆盖。确认没有要保留的数据后删除再录：

```bash
rm -rf ~/.cache/huggingface/lerobot/yutongzhu/so101-pick-cube
```

## 5. 录制时的按键

在 SSH 终端里按，不要按到电脑上的浏览器或 Rerun 窗口。

| 按键 | 作用 |
|------|------|
| `n` | 提前结束当前这一集，或结束重置，进入下一集 |
| `r` | 丢掉当前这一集，重新录 |
| `q` | 结束整个录制并保存 |

一次 `n` 只结束当前集，然后进入重置。看到 `Reset the environment` 后把现场摆回起始状态，再按一次 `n` 才开始下一集。

启动后会先刷 TorchCodec 警告和 `Svt[info]` 编码器日志，这时键盘监听还没装上。等这些日志停了、录制已经开始再按 `n`。按对了会打印 `Exiting loop...`，字母本身不会留在屏幕上。如果终端里出现你刚按的 `n`，这一下没有进到录制控制。

下一集开始前还要等当前帧发完，以及这一集的视频收尾。

## 6. 每一集怎么录

1. 摆好起始姿势和物体。
2. 用手掰 leader，让 follower 做完任务。动作尽量一致，物体要一直在 `wrist` 和 `front` 画面里。
3. 做完按 `n`。
4. 在重置时间里把物体和手臂放回这一组的起始位置，再按 `n`。
5. 同一位置先录大约 10 集，再换下一个位置。一个抓放任务建议至少 50 集。

结束时按 `q`，或等 5 集录满。脚本会先打印 Cadence 报告，再整理数据集并上传到 Hugging Face。

Cadence 里的实际频率如果明显低于 30 Hz，数据仍能保存，但帧数会比 30 Hz 少。重置阶段不计入这个报告。

## 7. 数据在哪里

开发板上：

```text
~/.cache/huggingface/lerobot/yutongzhu/<数据集名字>
```

`--save-to mac` 会在录完后用 `rsync` 拷到电脑的 `/Users/zyt/lerobot-datasets/yutongzhu/<数据集名字>`。开发板要能 SSH 登录 `zyt@192.168.18.9`。拷贝失败时，开发板上的那份还在。

启动时如果出现 TorchCodec / `libnppicc.so.13` 的长日志，可以忽略。解码会改用 PyAV，录制会继续。

## 8. 结束和回放

用 `Ctrl+C` 结束时，脚本会先关两只手臂的力矩再退出。不要用 `kill -9`，否则力矩可能保持打开。

回放某一集（从 0 开始计数）：

```bash
python replay.py --dataset-name so101-pick-cube --episode 0
```

`teleoperate_with_cameras.py` 只做实时遥操作，不写数据集。

## 9. 推理

任务名对应训练输出目录 `~/outputs/act/<任务名>/checkpoints/last/pretrained_model`。手臂和相机配置写在 `rollout.sh` 里，换任务只换第一个参数：

```bash
bash rollout.sh so101-adjust-bottle
bash rollout.sh so101-adjust-bottle --fp16
bash rollout.sh so101-pick-cube --duration=30
```

脚本一连上 follower 就会打开力矩。跑之前把物体和手臂摆回录数据时的起始状态。旧的单相机权重和现在的双相机对不上，不能拿来跑。
