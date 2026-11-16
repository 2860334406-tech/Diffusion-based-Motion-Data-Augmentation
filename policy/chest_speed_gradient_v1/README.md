# EXP-022 胸推卸力：model_31995.pt

这是胸推速度梯度课程的 C 阶段策略，任务为 `Smp-Force-Unloading-G1`。策略、prior 和原始训练配置按同一实验身份归档，供 force estimator 联调及后续评估使用。

## 文件

| 文件 | 来源与用途 |
| --- | --- |
| `policy.pt` | 原始 `model_31995.pt`，字节不变；包含 Actor、Critic、各自的观测归一化、optimizer、iteration 和环境恢复信息。 |
| `prior.pt` | 原始 `pretrained_lafan_low_speed_static80.pt`，字节不变；内置 `q_low/q_high` 特征归一化统计。 |
| `prior.metadata.json` | 原 prior 元数据，包含权重身份及训练参数。 |
| `env.yaml`、`agent.yaml` | C 阶段运行实际保存的环境与 PPO 配置，原样保留。 |
| `training.json` | 原课程配置、阶段血缘和 C 阶段实际训练命令。 |
| `training_source.diff` | 训练运行保存的 Git commit、状态和跟踪文件差异，用于追溯。 |
| `evaluation.json` | 2026-11-12 的原始正式评估报告，原样保留。 |
| `eval_config.json` | 相同任务条件的便携评估配置；文件和输出路径改为仓库相对路径，默认关闭录屏。 |
| `evaluate.sh` | 从任意目录调用的评估入口，先校验文件，再使用匹配参数运行现有评估脚本。 |
| `manifest.json`、`SHA256SUMS` | 文件来源、SHA-256、代码版本、力接口和校验清单。 |

这套任务使用无条件 SMP prior 和运行时在线评分归一化，不需要独立 `calibration.pt`。Actor 观测归一化在 `policy.pt` 的 `actor_state_dict` 中，加载策略时应一同恢复。

## 原始训练配置

| 参数 | 值 |
| --- | --- |
| 原始 C run | `2026-11-10_17-32-59_gsi015_decay10_static80_r010_v005_010_020_chest_c_seed42` |
| C 阶段起点 | B2 的 `model_12996.pt`，恢复完整 PPO 状态 |
| C 阶段新增训练量 | 19,000 iterations；最终 checkpoint iteration 为 31,995 |
| 并行环境 / seed | 4096 / 42 |
| Actor / Critic 输入维度 | 128 / 992 |
| 动作维度 / MLP | 29 / 512、256、128，ELU |
| 初始动作标准差 / 学习率 | 0.10 / 0.0001 |
| Prior | static80，无条件；运动窗口 `(10,59)` |
| GSI 最大初始平面速度 | 0.15 m/s |
| 静止速度尺度 / 静止奖励权重 | 10.0 / 0.1 |
| 球半径 / 质量 | 0.10 m / 5 kg |
| 推速 / 控制力上限 / ramp | 0.20 m/s / 0.15 倍机器人重量 / 0.25 s |
| 等待时间 / 接触部位 | 1–3 s / 固定胸口 |
| 推进方向 | `front_to_back` |
| 控制周期 / 物理步长 | 0.02 s（50 Hz）/ 0.005 s |

`env.yaml` 和 `agent.yaml` 保留了原机器上的绝对路径。它们是训练快照；便携评估入口会显式指定本目录下的策略和 prior，以及上述实验参数。继续训练前，按接收方路径核对快照和 `training.json`，并另建输出目录。

## 校验与评估

在仓库根目录检查交接文件：

```bash
cd policy/chest_speed_gradient_v1
sha256sum -c SHA256SUMS
cd ../..
```

安装锁定依赖后，执行与历史正式评估相同规模的评估：

```bash
uv sync --locked
bash policy/chest_speed_gradient_v1/evaluate.sh
```

默认使用 `cuda:0`、256 个环境、1024 个 episode、seed 42。输出位于 `logs/reproduction/policy_handoff/chest_speed_gradient_v1/`，与包内历史报告分开保存。默认不录屏。

先做少量回合的接口联调时，可以指定 GPU 和规模：

```bash
POLICY_DEVICE=cuda:0 POLICY_NUM_ENVS=16 POLICY_NUM_EPISODES=16 \
  bash policy/chest_speed_gradient_v1/evaluate.sh
```

少量回合只用于检查加载和接口；正式比较使用 1024 回合及相同任务条件。`eval_config.json` 记录默认配置，脚本支持上面三个环境变量覆盖运行设备和评估规模。

## force estimator 接口

当前每帧接触观测共 8 维，Actor 使用 4 帧历史：

```text
[active,
 clip(F_filtered / (M_robot*g), 0, 0.4),
 direction_local_x, direction_local_y, direction_local_z,
 position_local_x, position_local_y, position_local_z]
```

- 力标量按总接触法向力定义，以牛顿计；经过控制子步聚合与 EMA 后按机器人重量归一化。
- 方向和接触位置表达在机器人当前 heading 局部坐标系中；位置相对 root，单位为米。
- 无接触时 8 维全部为零。控制频率为 50 Hz，接触历史长度为 4。
- estimator 接入后保持观测字段、顺序、尺度与历史长度，使用这份策略自身的 Actor 归一化。
- 仿真真值仍用于评价实际卸力效果。联合实验另建输出目录，记录 estimator 的代码和权重版本。

## 已有正式评估与局限

以下数值来自包内 `evaluation.json`：确定性 Actor、胸推 0.20 m/s、1024 episodes、seed 42。

| 指标 | 结果 |
| --- | --- |
| Event success | 98.828125% |
| 紧急力终止 | 0% |
| 自碰撞终止 | 1.171875% |
| 关节硬限位终止 | 0% |
| 峰值接触力 mean / p95 | 0.32855 / 0.38259 倍机器人重量 |
| 接触足滑速度 p95 | 1.77736 m/s |
| 接触前平面速度 p95 | 0.37007 m/s |

策略通过原 C 阶段的成功率、紧急力和非力学失败门槛。足滑和接触前速度长尾仍明显，作为当前胸推卸力联调基线使用；本实验没有验收肩部定向转腰。

## 代码身份与验证范围

训练时的 Git 基础提交为 `28d1dfd1b7b4447977fbdb2062167d9f87a2faaa`，当时有未提交和未跟踪源码。`training_source.diff` 记录了基础提交和跟踪文件差异，但不包含当时所有未跟踪文件，因此仅作辅助追溯。

本交接使用的仓库代码基线记录在 `manifest.json` 的 `code.runtime_commit`。准备交接时检查了权重 SHA-256、策略张量有限性、归一化维度、真实 `EvalCfg` 的 CLI 参数解析及脚本语法；历史正式评估未重新运行，未占用训练 GPU。
