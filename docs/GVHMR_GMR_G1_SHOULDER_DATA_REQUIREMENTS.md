# GMR 输出到 SMP 的 G1 肩部动作数据要求

- 文档版本：v2.0
- 日期：2026-11-13
- 范围：只规定 **GMR 已完成 Unitree G1 重定向后**的数据格式与验收要求
- 目标任务：`stand / left_shoulder / right_shoulder` 三条件 Shoulder80 Prior

本文不规定视频采集或 GVHMR 输入。上游采用何种方式不影响最终交付标准；只有满足本文要求的 GMR→G1 数据才能进入 SMP 数据集。

## 1. 最终必须交付的文件

每条动作以唯一 `clip_id` 标识，至少交付：

| 文件 | 要求 |
|---|---|
| `<clip_id>.npz` | 推荐的安全数值归档，保存 GMR 输出数组 |
| `<clip_id>.csv` | SMP 导入用无表头 36 列 CSV |
| `<clip_id>.json` | FPS、标签、主体、split、GMR 身份和哈希 |

GMR 官方脚本保存的 pickle 只能作为可信中间产物，不作为最终训练输入。最终归档应转换为不执行代码的 NPZ/CSV。

## 2. GMR 数值数组契约

同一 clip 的所有数组必须共享相同帧数 `T`：

| NPZ key | 形状 | 类型 | 单位/顺序 |
|---|---:|---|---|
| `fps` | 标量 | `float32/float64` | 实际 GMR 输出 FPS，必须 `> 0` |
| `root_pos` | `(T, 3)` | `float32/float64` | 米，`x y z` |
| `root_rot` | `(T, 4)` | `float32/float64` | 单位四元数，`x y z w` |
| `dof_pos` | `(T, 29)` | `float32/float64` | 弧度，按第 4 节固定顺序 |

硬要求：

- `T ≥ 10`，且所有数组第一维完全一致；
- 全部数值有限，不允许 NaN/Inf；
- 不允许 object array、pickle object 或可执行对象；
- 不允许用填零、复制末帧或删除异常帧来掩盖 GMR 失败；
- `fps` 使用实测输出值，不能无依据写死为 30；
- 必须核对实际输出帧数，不能假定与上游输入逐帧等长；
- 一个文件只能包含一条连续动作，不能拼接不同 take。

## 3. 最终 G1 坐标与四元数

最终统一为右手坐标系：

- `+x`：G1 正前方；
- `+y`：G1 左侧；
- `+z`：竖直向上；
- root position 单位为米；
- joint position 单位为 rad；
- root quaternion 顺序必须为 `xyzw`。

四元数要求：

- 每帧满足 `abs(norm(q)-1) ≤ 1e-3`；
- 归一化后仍必须保持原旋转；
- 利用 `q` 与 `-q` 等价性统一相邻帧符号，使 `dot(q[t-1], q[t]) ≥ 0`；
- 禁止欧拉角直接按列当作四元数；
- 禁止在 CSV 中使用 GMR 内部可能采用的 `wxyz` 顺序。

## 4. 29 个 G1 关节的固定顺序

`dof_pos[:, i]` 和 CSV 第 `7+i` 列必须严格对应：

| i | 关节名 |
|---:|---|
| 0 | `left_hip_pitch_joint` |
| 1 | `left_hip_roll_joint` |
| 2 | `left_hip_yaw_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_pitch_joint` |
| 7 | `right_hip_roll_joint` |
| 8 | `right_hip_yaw_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |
| 12 | `waist_yaw_joint` |
| 13 | `waist_roll_joint` |
| 14 | `waist_pitch_joint` |
| 15 | `left_shoulder_pitch_joint` |
| 16 | `left_shoulder_roll_joint` |
| 17 | `left_shoulder_yaw_joint` |
| 18 | `left_elbow_joint` |
| 19 | `left_wrist_roll_joint` |
| 20 | `left_wrist_pitch_joint` |
| 21 | `left_wrist_yaw_joint` |
| 22 | `right_shoulder_pitch_joint` |
| 23 | `right_shoulder_roll_joint` |
| 24 | `right_shoulder_yaw_joint` |
| 25 | `right_elbow_joint` |
| 26 | `right_wrist_roll_joint` |
| 27 | `right_wrist_pitch_joint` |
| 28 | `right_wrist_yaw_joint` |

必须按 GMR 输出的关节名称显式建立映射，不能仅凭数组长度或当前 GMR 内部索引猜测顺序。

## 5. SMP 导入 CSV

CSV 必须：

- 无表头；
- 逗号分隔；
- 每行一帧；
- 形状严格为 `(T, 36)`；
- 全部为有限十进制浮点数；
- 不包含时间戳列、索引列或标签列。

列布局：

| CSV 列 | 内容 |
|---|---|
| `0:3` | `root_pos[x,y,z]` |
| `3:7` | `root_rot[x,y,z,w]` |
| `7:36` | 29 个 `dof_pos`，按第 4 节顺序 |

CSV 与 NPZ 必须逐元素一致；允许的差异仅来自文本浮点保存精度。CSV、NPZ 均需保存 SHA-256。

## 6. 条件标签

每条 clip 只保存一个动作条件，不再保存或推断起转、转入、保持等阶段标签：

| condition | ID | G1 动作方向 |
|---|---:|---|
| `stand` | 0 | 腰部接近中立 |
| `left_shoulder` | 1 | 左肩在 pelvis-heading 坐标中沿局部 `-x` 后退 |
| `right_shoulder` | 2 | 右肩在 pelvis-heading 坐标中沿局部 `-x` 后退 |

NULL ID `3` 只用于模型训练时的条件丢弃，不是数据标签，也不是胸口条件。10 帧窗口只用于 SMP 特征转换，不承担整条动作是否合格的判定。

## 7. G1 运动学和 MuJoCo 安全要求

每一帧必须通过当前 G1 模型检查：

- root 高度不低于 `0.60 m`；
- root roll 和 pitch 的绝对值分别不超过 `20°`；
- 29 个关节均位于模型限位内，并保留 `1e-4 rad` margin；
- 支撑脚最低点位于地面下 `0.02 m` 到地面上 `0.03 m` 范围；
- 非地面几何接触穿透超过 `0.0001 m` 时剔除；
- 非脚部与地面穿透超过 `0.002 m` 时剔除；
- 轨迹连续，无 root、四元数或关节角单帧跳变；
- 不允许通过裁剪关节角来伪造限位通过。

这些检查只证明运动学可接受，不证明机器人动力学可执行。

## 8. 50 Hz 整条轨迹终点门槛

肩部动作按整条连续轨迹验收，不再按 10 帧窗口划分 `early/strong/hold`，也不固定 waist yaw 幅度或转腰速度。规定位置定义为目标肩相对 pelvis、表达在逐帧 pelvis heading 坐标系中的局部 `x` 后移量。

每条肩部 clip 必须记录转动开始帧和允许完成时间，并同时满足：

- 目标肩规定后移量为 `0.015 m`；到位判定允许 `0.001 m` 数值容差，即后移量达到 `0.014 m` 视为进入目标区；
- 从转动开始到首次进入目标区的时间不得超过该 clip 的 `completion_deadline_s`；
- 轨迹终点的目标肩后移量必须实际达到 `0.015 m`，不能用到位容差代替终点要求；
- 最后 1 秒保持段内，目标肩后移量始终不低于 `0.014 m`；
- 最后 1 秒内目标肩三维局部位置相对该段中位位置的最大偏差不超过 `0.002 m`；
- 最后 1 秒 root 平面速度 p95 不超过 `0.05 m/s`，waist yaw 速度 p95 不超过 `0.10 rad/s`；
- pelvis yaw 全轨迹变化不超过 `0.05 rad`，防止整体转身冒充腰部退让；
- waist yaw 只检查左右方向、关节/电机安全与动力学可执行性，不要求固定终点角度或固定速度曲线。

输出 10 帧、stride 5 的窗口时，完全位于中立段的窗口标为 `stand`，完全位于转动开始帧之后的窗口标为对应肩部条件；跨越条件边界的混合窗口剔除。

## 9. 完整动作语义要求

肩部 clip 必须在同一条连续 G1 轨迹中包含：

```text
中立站立 → 连续退让转腰 → 到达规定肩部位置 → 稳定保持
```

即使通过数值门槛，出现以下任一情况仍必须剔除：

- 行走、跑步、迈步、交叉步、跳跃或落地；
- 舞蹈、格斗、出拳或大幅手势；
- 双臂平举、摆臂或扶髋与腰转强耦合；
- 骨盆整体旋转代替腰部相对骨盆旋转；
- root 明显平移、脚滑、悬空或穿地；
- 中立、转动和保持来自不连续片段；
- 左右方向错误或镜像后标签未交换；
- GMR IK 产生手臂反折、关节尖峰或姿态突变；
- 需要事后放大 yaw、改变播放速度或复制帧才能过门槛。

## 10. 转换为 SMP `(N,10,59)`

当 GMR 输出实测为 30 FPS 时：

```bash
uv run scripts/csv_to_npz.py \
  --input-dir /path/to/g1_csv \
  --output-dir /path/to/npz \
  --input-fps 30 \
  --output-fps 50 \
  --window-size 10 \
  --stride 5
```

不同 FPS 的文件必须分组转换。脚本只非递归读取 `input-dir` 直接目录下的 CSV。

59 维逐帧布局：

| 特征索引 | 内容 | 维数 |
|---|---|---:|
| `0:3` | root position | 3 |
| `3:9` | root rotation 6D tan-normal | 6 |
| `9:38` | 29 个 joint position | 29 |
| `38:53` | 5 个 end-effector position | 15 |
| `53:56` | root linear velocity | 3 |
| `56:59` | root angular velocity | 3 |

`waist_yaw_joint` 在 59 维特征中的索引为 `21`。空间量由转换器统一锚定到窗口最后一帧的 pelvis yaw heading，GMR 输出端不得提前做另一套窗口局部化。

## 11. 每条 clip 的 JSON 元数据

至少包含：

```json
{
  "clip_id": "subject1_session01_left_shoulder_take001",
  "subject_id": "subject1",
  "session_id": "session01",
  "condition": "left_shoulder",
  "condition_id": 1,
  "split": "train",
  "fps": 30.0,
  "frames": 150,
  "coordinate_system": "+x forward, +y left, +z up",
  "quaternion_order": "xyzw",
  "joint_order": "smp_g1_29_v1",
  "gmr_repository": "https://github.com/YanjieZe/GMR",
  "gmr_commit": "<40-character commit>",
  "gmr_config_sha256": "<sha256>",
  "source_motion_sha256": "<sha256>",
  "npz_sha256": "<sha256>",
  "csv_sha256": "<sha256>",
  "review_status": "accepted",
  "rejection_reason": ""
}
```

同一主体的所有 clip、镜像、重叠窗口和不同重定向版本必须留在同一 split。镜像数据必须交换左右肩标签，且不能计为新的独立动作来源。

## 12. 最终验收检查表

```text
[ ] NPZ 包含 fps/root_pos/root_rot/dof_pos，形状和帧数一致
[ ] CSV 为无表头 (T,36)，与 NPZ 数值一致
[ ] 所有数值有限，四元数为连续的 xyzw 单位四元数
[ ] 29 个关节已按名称映射到固定顺序
[ ] 坐标为 +x 前、+y 左、+z 上，单位为 m/rad
[ ] condition、subject、split 和 GMR commit 已记录
[ ] 左肩 waist yaw 为正，右肩为负，目标肩沿 -x 后退
[ ] 关节限位、支撑、自碰撞和地面检查通过
[ ] 同一连续轨迹包含中立站立、连续转动、按时到位和稳定保持
[ ] 没有步行、跳跃、舞蹈、格斗、手势或整体转身污染
[ ] 重采样后整条轨迹满足第 8 节终点、完成时间和保持门槛
[ ] CSV、NPZ、配置和源动作 SHA-256 已固化
```

任何一项未通过时，该 clip 不得进入正式 Shoulder80 Prior 数据集。
