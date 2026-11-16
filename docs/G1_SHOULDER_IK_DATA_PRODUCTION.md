# G1 连续随机肩部退让全身轨迹 IK 数据生产规范

- 文档版本：v2.0（连续幅度）
- 日期：2026-11-13
- 实验：EXP-032
- 范围：只生产和验收合成动作数据，不训练 Prior 或 PPO

## 1. 目标与边界

从 G1 中立站姿生成 `stand / left_shoulder / right_shoulder` 三条件纯净数据。肩部轨迹连续完成“中立站立 → 退让转腰 → 到达目标幅度与肩部位置 → 稳定保持”，不生成回中。双脚全程固定，不迈步，不施加导轨、推球或外力。

肩部目标幅度是相对转动前站立基线的 `waist_yaw_joint` 变化量，在 `15°–70°` 内连续随机分布；左肩为正、右肩为负。转动速度同样变化，不再使用离散角度或离散完成时间网格。训练数据只使用当前 G1 模型在浮动基座、完整碰撞和原位置执行器下的实际 MuJoCo 执行轨迹，不使用 IK 参考，也不混入 LAFAN、G1 Moves 或 AMASS-G1 肩部正例。

EXP-032 的逻辑产物目录为 `logs/reproduction/g1_shoulder_ik_v2/`。EXP-031、v1 pilot 和独立 60° 诊断均为只读历史证据，不作为 v2 pilot 或正式数据来源。

## 2. 固定环境与入口

- Python `>=3.11,<3.14`，MuJoCo 3.11.0，mjlab 1.6.0，Mink 1.4.0。
- seed 42，50 Hz 输出，MuJoCo timestep `0.005 s`，控制周期 `0.02 s`。
- 生成器：`src/smp/data/shoulder_ik.py`。
- 主驱动：`scripts/run_g1_shoulder_ik_data.py`。
- 边界验证：`scripts/smoke_g1_shoulder_ik_v2.py`。
- 汇总与回放：`src/smp/data/shoulder_ik_dataset.py`、`scripts/render_g1_shoulder_ik_review.py`。

manifest 固化源码、依赖、seed、抽样 schema、全部 clip spec SHA-256、门槛和配额。已有输出身份不同则拒绝覆盖；相同身份恢复时只补齐缺失 slot。

## 3. 数量、连续抽样与划分

| 数据集 | stand | left | right | 总数 |
|---|---:|---:|---:|---:|
| pilot | 48 | 96 | 96 | 240 |
| formal | 480 | 960 | 960 | 2400 |

六种站姿为站距 `0.22/0.24/0.26 m` 与膝屈曲 `0.55/0.70 rad` 的组合。Pilot 每种站姿包含 16 个左右配对 family；formal 每种站姿包含 160 个配对 family。

正式划分在每个站姿内固定为 train/validation/test=`112/24/24` 对，因此总配额为：

| split | stand | left | right | 总数 |
|---|---:|---:|---:|---:|
| train | 336 | 672 | 672 | 1680 |
| validation | 72 | 144 | 144 | 360 |
| test | 72 | 144 | 144 | 360 |

每个 `(stage, split, 站姿)` 独立执行 randomized Latin-hypercube：

- 目标幅度连续采样于 `[15°,70°]`，不取整；首末 strata 仍为连续随机值，但约束在 `[15°,16°]`、`[69°,70°]`，保证边界覆盖。
- 目标平均角速度使用独立随机流，在 `[12°,18°]/s` 分层采样，防止与幅度相关。
- 完成帧数为 `ceil(幅度/速度×50)`，完成时间为帧数除以 50，约 `0.84–5.84 s`。
- 左右配对共享幅度、速度、站姿、保持和平衡高级参数，但分别求解 IK 和动力学，不复制或镜像关节数组。
- 保持时长由固定 seed 在 `1.2–2.0 s` 内生成。

同 family 的左右动作只能进入同一 split。Retry 保持幅度、速度、完成时间、family 和 split 不变，仅改变既有的手臂平衡、COM 偏置和保持变体；两次失败即 blocked，不能改抽样角度补位。

## 4. IK 与动力学执行

每条肩部轨迹先稳定站立 1 s，再用五次最小加加速度曲线 `10s³-15s⁴+6s⁵` 到达目标，最后保持。每个 50 Hz 输出帧执行 4 个 Mink 子步。

约束包括双足六自由度位姿、pelvis 姿态与 yaw、COM 支撑、torso 相对 pelvis yaw、90% 软关节限位、速度限制和自碰撞规避。腿、腰和手臂可共同参与平衡。记录前对应方向的 `0.002 rad` 预稳定仅消除数值偏置；幅度始终按实际站立段 waist yaw 中位数到最终 waist yaw 的变化计算，不把预稳定算入目标幅度。

动力学回放使用地面摩擦 0.6、完整碰撞、浮动基座、无随机化、无 root 锁定和无外力。位置执行器及其原始增益、阻尼和力矩上限不改写；高层支撑一致跟踪器的 base 平移/转动/普通关节/waist-yaw `kp` 为 `80/100/80/180`，`kd=2√kp`，命令力矩限制为原上限的 95%。

## 5. 单条与整组硬门槛

每条轨迹必须同时满足：

- 数值有限，关节位于 90% 软限位；足端 IK 误差 `≤2 mm/1°`。
- root 高度 `≥0.60 m`，roll/pitch `≤20°`，pelvis yaw 变化 `≤0.05 rad`，root 平移 `≤0.05 m`。
- 双脚接触率分别 `≥95%`，始终至少一脚接触；单脚滑移 `≤0.015 m`，foot yaw 漂移 `≤2°`。
- 全关节/waist 跟踪 RMSE `≤0.08/0.04 rad`，最终参考误差 `≤0.03 rad`。
- 实际 signed waist 幅度方向正确、位于 `15°–70°`，闭区间比较只允许 `0.001°` 数值容差；与本条目标误差 `≤0.03 rad`。
- 力矩 p99 `≤95%`，饱和帧 `≤1%`，关节速度不超过电机限制。
- 目标肩局部 x 后移终点 `≥0.015 m`；在完成截止前进入 `≥0.014 m`，最后 1 s 始终 `≥0.014 m`，三维保持偏差 `≤0.002 m`。
- 最后 1 s root 平面速度 p95 `≤0.05 m/s`、waist 速度 p95 `≤0.10 rad/s`。
- 无自碰撞、非脚部触地、跌倒或穿透。

完整 pilot/formal 还必须在每个 side/split 的实际幅度中覆盖全部 11 个 5°区间 `[15,20),…,[65,70]`，最小值 `≤16°`、最大值 `≥69°`。单条全部通过但整组分布不通过时，manifest 仍为 blocked。

## 6. 文件与窗口契约

每条轨迹保存 IK reference、实际 execution NPZ、无表头 `(T,36)` CSV 和 JSON。CSV/NPZ 遵循 root xyz、root quaternion xyzw、29 个 G1 关节顺序；实际执行状态是唯一训练来源。

汇总器通过真实 G1 FK 生成 stride 5 的 `(N,10,59)` 窗口，仅保留完全属于 stand 或对应肩部条件的窗口。每个 clip 和窗口额外记录：

- target/actual waist amplitude；
- target mean yaw speed 与完成时间；
- family、split、clip ID 和帧区间；
- `clip_balanced_weight=1/该 clip 有效窗口数`。

原始 NPZ 保留全部窗口，同时输出 clip-level 与 window-weighted 幅度分布。后续 Prior 应使用 clip-first sampler 或等价权重，避免持续时间更长的大幅度动作因窗口更多而自动占优。正式归一化统计只由 train 计算；pilot 统计仅作诊断。

## 7. 执行与恢复

边界 smoke：

```bash
uv run scripts/smoke_g1_shoulder_ik_v2.py   --output-dir logs/reproduction/g1_shoulder_ik_v2/boundary_smoke
```

必须使左右肩 `15°/42.5°/70°` 六条全部通过，之后才生成 pilot：

```bash
uv run scripts/run_g1_shoulder_ik_data.py   --stage pilot   --output-dir logs/reproduction/g1_shoulder_ik_v2/pilot   --seed 42 --workers 16

uv run scripts/report_g1_shoulder_ik_generation.py   --input-dir logs/reproduction/g1_shoulder_ik_v2/pilot
```

Pilot 自动门槛通过后，以 EGL 渲染 60 条：12 stand，左右肩各 24；每侧四个幅度四分位各 6 条。人工检查方向、迈步、整体转身、抖动和明显碰撞，并在 `visual_review.json` 固化 60/60 结论。

```bash
MUJOCO_GL=egl uv run scripts/render_g1_shoulder_ik_review.py   --input-dir logs/reproduction/g1_shoulder_ik_v2/pilot   --manifest manifest.json --count 60
```

只有 pilot 自动与视觉门槛都通过后才能生成正式集：

```bash
uv run scripts/run_g1_shoulder_ik_data.py   --stage formal   --output-dir logs/reproduction/g1_shoulder_ik_v2/formal   --pilot-manifest logs/reproduction/g1_shoulder_ik_v2/pilot/manifest.json   --seed 42 --workers 16
```

正式集完成后同样生成 gate report 和 60 条分层回放。任何身份不一致、两次候选失败、分布缺口或视觉问题均停止；不放宽阈值、不换角度补样、不启动 Prior/PPO。
