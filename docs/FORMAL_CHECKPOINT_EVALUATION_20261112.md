# 三个指定 Checkpoint 的正式评估与 Replay（2026-11-12）

## 范围与方法

- Checkpoint：速度梯度胸推 `model_31995.pt`、速度梯度肩推 `model_12000.pt`、Conditional v2 `model_17000.pt`。
- 正式统计：每个固定条件使用 256 个环境、seed 42、确定性 Actor 均值策略，独立运行 1,024 episodes；共 8 个条件、8,192 episodes。
- Replay：另用单环境、相同 seed/checkpoint/prior/力学参数运行，每个条件录制 600 控制帧（50 Hz、12.0 s、1280×720）。8 段视频均经逐帧解码和 SHA-256 核验。
- 硬件：A100-SXM4-80GB，物理 GPU0，UUID `GPU-d0c5535e-c4e1-8b72-e6f1-ae82f0b75771`；进程内设备 `cuda:0`，EGL 无头渲染。
- 全部正式统计均为有限值，无 NaN/Inf。`model_12000.pt` 是用户指定的 B2 中间 checkpoint，`model_17000.pt` 是用户指定的 C 中间 checkpoint，二者不表述为完整预算终值。

## Checkpoint 身份

| Checkpoint | SHA-256 | 评估语义 |
|---|---|---|
| 胸推 `model_31995.pt` | `82ddfe95ee383cd4cf62c65d6f080a46138ad43fbdd92f19e28d9272ea4e0d33` | C，胸口，`0.20 m/s` |
| 肩推 `model_12000.pt` | `ea925cc60875137dd9ffc4f70e799b20c1c7ebd8151743c2f3a70aab446ff648` | B2 中间点，左右肩分别固定，`0.10 m/s` |
| Conditional `model_17000.pt` | `9bfb8c308f143d10771d5828b97b043c46e866a1dcb0a6e3d1227b93217ec29e` | C 中间点，无接触及四接触部位分别固定，`0.05 m/s` |

## 速度梯度策略结果

| 条件 | 原阶段 gate | Event success | 紧急力 | 自碰撞终止 | 关节限位 | 峰值力 mean / p95（mg） | 足滑 p95（m/s） | 接触前速度 p95（m/s） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 胸推 `model_31995` | 通过 | 98.83% | 0% | 1.17% | 0% | 0.329 / 0.383 | 1.777 | 0.370 |
| 左肩 `model_12000` | 通过 | 27.93% | 0.20% | 8.20% | 64.75% | 0.277 / 0.647 | 1.496 | 0.445 |
| 右肩 `model_12000` | 未通过 | 0% | 0% | 10.16% | 91.21% | 0.434 / 0.534 | 1.432 | 0.434 |

胸推 checkpoint 稳定通过 C 的成功率、紧急力和非力学失败门槛，但接触前速度长尾及足滑仍明显。肩推 checkpoint 存在强烈左右不对称；左肩虽满足原 B2 gate，但该 gate 只检查成功率和紧急力，没有否决 64.75% 的关节限位。右肩成功率为 0%，且 91.21% episode 因关节硬限位终止。因此不能把左肩的 gate 通过解释为该肩推策略整体可用。

## Conditional 策略结果

无接触条件通过 A 门槛：接触前/站立速度 p95 `0.1308 m/s`、位移 p50 `0.0481 m`、非力学失败率 `0.1953%`；另有自碰撞终止 `0.8789%`。

| 条件 | C gate | Event success | 接触覆盖 | 方向正确 | 恢复静止 | 紧急力 | 自碰撞 | 工作区失败 | 峰值力 mean / p95（mg） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 左肩 | 通过 | 79.10% | 85.94% | 16.99% | 80.96% | 0.29% | 0.49% | 1.66% | 0.207 / 0.391 |
| 右肩 | 未通过 | 43.46% | 57.23% | 21.78% | 52.93% | 0% | 0.88% | 40.82% | 0.151 / 0.351 |
| 胸口 | 通过 | 77.34% | 90.23% | 32.32% | 84.28% | 0% | 1.07% | 8.40% | 0.157 / 0.224 |
| 背部 | 未通过 | 31.54% | 31.64% | 28.52% | 31.15% | 0% | 78.52% | 9.18% | 0.118 / 0.197 |

Conditional 的 event success 表示力卸载成功，不等同于条件动作方向正确。左右肩的两项指标差异很大：方向正确率仅 16.99%/21.78%；episode 末端腰 yaw 均值分别为 `-0.0360/+0.0149 rad`，与设计的左肩正向、右肩负向相反，说明条件 prior 的方向意图尚未可靠传递到 PPO 策略。胸口的平均纵向响应为 `-0.1050 m`，符合后退方向；背部为 `+0.1044 m`，方向符号正确，但 78.52% 自碰撞导致大部分 episode 在有效接触前结束。右肩的主要瓶颈是 40.82% 工作区失败和不足的接触覆盖，背部的主要瓶颈是自碰撞。

按原有各阶段 gate 计，8 个条件中 5 个通过、3 个未通过；但 gate 不包含自碰撞与关节硬限位，因此工程结论严于 gate：胸推可作为当前最稳定的候选；`model_12000` 肩推和 `model_17000` 的右肩/背部仍不具备验收条件。

## 产物

- 正式汇总：[summary.json](../logs/reproduction/formal_evaluation_20261112/summary.json)
- 正式运行清单：[manifest.json](../logs/reproduction/formal_evaluation_20261112/manifest.json)
- 正式视频核验：[video_validation.json](../logs/reproduction/formal_evaluation_20261112/video_validation.json)
- 单环境 replay 清单：[video_validation.json](../logs/reproduction/formal_replay_20261112/video_validation.json)
- 抽帧概览：[速度梯度](../logs/reproduction/formal_replay_20261112/replay_contact_sheets/speed_clean_overview.jpg)、[Conditional](../logs/reproduction/formal_replay_20261112/replay_contact_sheets/conditional_clean_overview.jpg)

单环境 replay：

- [胸推 model_31995](../logs/reproduction/formal_replay_20261112/replays/speed_chest_model31995_chest/rl-video-step-0.mp4)
- [左肩 model_12000](../logs/reproduction/formal_replay_20261112/replays/speed_shoulders_model12000_left/rl-video-step-0.mp4)
- [右肩 model_12000](../logs/reproduction/formal_replay_20261112/replays/speed_shoulders_model12000_right/rl-video-step-0.mp4)
- [Conditional 无接触](../logs/reproduction/formal_replay_20261112/replays/conditional_model17000_stand/rl-video-step-0.mp4)
- [Conditional 左肩](../logs/reproduction/formal_replay_20261112/replays/conditional_model17000_left_shoulder/rl-video-step-0.mp4)
- [Conditional 右肩](../logs/reproduction/formal_replay_20261112/replays/conditional_model17000_right_shoulder/rl-video-step-0.mp4)
- [Conditional 胸口](../logs/reproduction/formal_replay_20261112/replays/conditional_model17000_chest/rl-video-step-0.mp4)
- [Conditional 背部](../logs/reproduction/formal_replay_20261112/replays/conditional_model17000_back/rl-video-step-0.mp4)

正式指标目录和 replay 目录均通过工作区符号链接指向 `/personal_fast_storage/smp_experiments/`，未覆盖训练期评估、checkpoint 或旧 replay。
