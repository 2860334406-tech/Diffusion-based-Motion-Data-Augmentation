# 一维导轨球形推头与弹簧阻尼装置复现规范

> 1. 版本：2026-11-12，`conditional_contact_v2` 实际运行配置  
> 目的：让另一个实验中的 Agent 无需阅读本仓库历史对话，即可复现相同的机械装置、控制器、接触测量和课程阶段参数。  
> 范围：本文冻结的是导轨球形推头装置及其任务接口，不包含 SMP Prior、PPO 网络或奖励改进方案。

## 1. 必须保持不变的核心定义

该装置由一个可重定位的固定导轨基座、一个单自由度滑块和一个刚性球头组成：

- 导轨基座在每次推力事件开始时，根据机器人当时的 heading 和目标部位放置；事件期间不再跟随机器人。
- 滑块只有一个沿导轨局部 `+x` 的平移自由度，坐标记为 `u`。
- 滑块不能横向移动，也不能旋转；球头因此不会滚动。
- slide joint 自身的 MuJoCo damping 为 `0`。
- “弹簧阻尼”由控制器每个控制步计算，并通过 `qfrc_applied` 施加到 slide joint；不要同时在 joint 上增加同值阻尼。
- 球头是动态刚体，机器人反作用力可以使其减速、停止或后退；不能用运动学位置插值替代。
- 导轨行程达到上限只记录遥测，不终止 episode；物理 joint limit 仍然生效。

当前实验使用 200 Hz 物理仿真和 50 Hz 控制：

| 项目 | 值 |
|---|---:|
| MuJoCo timestep | `0.005 s` |
| control decimation | `4` |
| 控制周期 `dt` | `0.020 s` |
| episode 长度 | `12.0 s` |
| 积分器 | `implicitfast` |
| 接触锥 | `pyramidal` |
| solver | `newton`，10 iterations，tolerance `1e-8` |
| `nconmax` | `64` |
| `contact_sensor_maxmatch` | `64` |

## 2. 坐标系、目标编号与推进方向

机器人 heading 局部坐标定义为：

- `+x`：机器人正前方；
- `+y`：机器人左侧；
- `+z`：竖直向上。

目标编号必须保持：

| PushTarget | ID | 碰撞几何 |
|---|---:|---|
| CHEST | 0 | `torso_collision` 前表面 |
| LEFT_SHOULDER | 1 | `left_shoulder_yaw_collision` |
| RIGHT_SHOULDER | 2 | `right_shoulder_yaw_collision` |
| BACK | 3 | `torso_collision` 后表面 |

当前 Conditional 实验采用 `push_direction_mode="front_to_back"`：

| 目标 | 球头初始侧 | 球头沿机器人局部方向运动 |
|---|---|---|
| 胸口 | 正前方 | `-x`，从前向后推 |
| 左肩 | 左肩正前方 | `-x`，从前向后推 |
| 右肩 | 右肩正前方 | `-x`，从前向后推 |
| 背部 | 正后方 | `+x`，从后向前推 |

导轨的世界方向由事件开始时的机器人 heading 旋转得到，并固定到事件结束。不要在机器人移动后重新瞄准或旋转导轨。

仓库还保留历史模式 `target_default`，其胸口/左肩/右肩/背部方向依次为 `(-x, -y, +y, +x)`；它不属于本文冻结的 Conditional v2 配置。

## 3. MuJoCo 机械模型

### 3.1 一维滑块与球头

等价的 MuJoCo 结构如下。实际仓库使用 `MjSpec` 创建实体，mjlab 再用 mocap body 包装其固定基座：

```xml
<body name="rail_base" mocap="true">
  <body name="slider">
    <joint name="rail_joint"
           type="slide"
           axis="1 0 0"
           limited="true"
           range="0 0.375"
           damping="0"/>
    <geom name="sphere"
          type="sphere"
          size="0.10"
          mass="5.0"
          friction="0.2 0.005 0.0001"
          solref="0.01 1.0"
          rgba="0.9 0.22 0.08 1"/>
  </body>
</body>
```

如果所用框架不能将带 joint 的实体自动包装为 mocap 基座，应显式建立一个 world 的 mocap 子 body，再把带 slide joint 的 `slider` 作为其子 body。基座只用于事件开始时定位，不能参与动力学。

### 3.2 冻结的几何与材料参数

| 参数 | 符号 | 当前值 | 单位/说明 |
|---|---|---:|---|
| 球半径 | `sphere_radius` | `0.10` | m |
| 球/滑块质量 | `sphere_mass` | `5.0` | kg；惯量由球几何推断 |
| 滑动摩擦 | `friction[0]` | `0.2` | MuJoCo sliding friction |
| 扭转摩擦 | `friction[1]` | `0.005` | MuJoCo torsional friction |
| 滚动摩擦 | `friction[2]` | `0.0001` | MuJoCo rolling friction |
| contact `solref` | — | `(0.01, 1.0)` | time constant、damping ratio |
| 导轨最小坐标 | — | `0.0` | m |
| 导轨最大坐标 | `max_travel` | `0.375` | m |
| joint damping | — | `0.0` | N·s/m；不要与虚拟阻尼重复 |
| 初始 slide 位置/速度 | — | `0.0 / 0.0` | m、m/s |
| 停放高度 | — | `z=-2.0` | 无事件时移到地面下方 |

注意：历史基线曾使用半径 `0.175 m`；本文复现的是后续 Conditional v2 使用的 `0.10 m` 球头，质量仍为 `5 kg`。

## 4. 球头的初始放置

当前配置使用几何表面间隙，而不是固定球心距离：

| 参数 | 值 |
|---|---:|
| `start_surface_gap` | `0.020 m` |
| `start_distance` | `0.35 m`，但在 `start_surface_gap` 非空时不生效 |

对目标 capsule/sphere 几何，求球心沿导轨方向的距离 `d`，使球表面与目标几何表面的最短间距正好为 `0.02 m`。实现应基于碰撞几何半径、capsule 半长和姿态计算，而不是将目标 body 原点当作表面。

肩部 capsule 必须瞄准其上端点：

```text
target_point = capsule_center
             + capsule_axis * half_length * sign(capsule_axis.z)
```

这是为了避免 G1 默认弯肘姿态下，瞄准肩 capsule 中心时球头先撞到前臂或手腕。胸口和背部使用 torso capsule/sphere 的中心线几何，不添加肩部端点偏移。

放置过程：

1. 读取事件开始时目标碰撞几何的世界位置和姿态。
2. 计算固定的世界导轨方向 `n_world`。
3. 求满足 2 cm 表面间隙的球心距离 `d`。
4. 令导轨基座位置 `p0 = target_point - d * n_world`。
5. 令基座 yaw 与 `n_world` 对齐，使 slide joint 局部 `+x` 等于 `n_world`。
6. 将 `u=0`、`u_dot=0`，随后进入 DRIVE。

## 5. 虚拟弹簧阻尼驱动

### 5.1 固定参数

| 参数 | 符号 | 值 |
|---|---|---:|
| 纵向刚度 | `k` | `700.0 N/m` |
| 纵向阻尼 | `c` | `118.32159566199232 N·s/m` |
| 阻尼计算 | — | `2*sqrt(700*5)`，对应自由球质量的临界阻尼 |
| 撤回参考速度 | `retract_speed` | `0.15 m/s` |

令 `u`、`u_dot` 为 slide joint 的实际位置和速度，`u_ref`、`v_ref` 为控制器参考。每个 50 Hz 控制步执行：

```text
F_pd = k * (u_ref - u) + c * (v_ref - u_dot)
F_drive = clamp(F_pd, -F_max, +F_max)
qfrc_applied[rail_joint] = F_drive
```

必须使用双向截断 `[-F_max, +F_max]`。早期设计草案中的 `[0, F_max]` 不是当前代码行为；双向力用于阻尼超调和执行撤回。

最大驱动力按机器人重量缩放，而不是按球质量：

```text
M_robot = 仿真中机器人全部 body mass 之和
W_robot = M_robot * 9.81
F_max = max_force_weight_ratio * W_robot
```

不要将 `max_force_weight_ratio` 直接当作牛顿值，也不要把球的 `5 kg` 代入该上限。

### 5.2 DRIVE 参考轨迹

进入 DRIVE 时：

```text
u_ref = 0
v_ref = 0
t_drive = 0
```

之后每个控制步：

```text
t_drive += dt
ramp = clamp(t_drive / ramp_time, 0, 1)
v_ref = reference_speed * ramp
u_ref = u_ref + v_ref * dt
```

这是速度线性 ramp 后的离散积分，不是直接给球头指定位置或速度。实际 `u` 和 `u_dot` 由质量、接触反力、PD 力和 joint limit 共同决定。

### 5.3 RETRACT 参考轨迹

完成卸力后：

```text
v_ref = -0.15
u_ref = max(0, u_ref + v_ref * dt)
```

仍使用同一个双向 PD 和同一个 `F_max`。当 `u < 0.005 m` 且 `abs(u_dot) < 0.02 m/s` 时进入 DONE，并把整个装置停放到 `z=-2 m`。

## 6. 课程阶段的驱动参数

除了下表，其他机械参数在所有阶段保持一致：

| 阶段 | 是否实际启动 | `reference_speed` | `max_force_weight_ratio` | `ramp_time` | 等待时间 |
|---|---|---:|---:|---:|---:|
| A | 否 | `0.05 m/s`（不生效） | `0.15`（不生效） | `0.25 s`（不生效） | `(1e9, 1e9) s` |
| A 扩展 | 否 | 同 A | 同 A | 同 A | `(1e9, 1e9) s` |
| B1 | 是 | `0.01 m/s` | `0.05` | `1.0 s` | 均匀采样 `1–3 s` |
| B2 | 是 | `0.03 m/s` | `0.10` | `0.5 s` | 均匀采样 `1–3 s` |
| C | 是 | `0.05 m/s` | `0.15` | `0.25 s` | 均匀采样 `1–3 s` |

Conditional v2 在每次事件开始时按以下概率抽样目标：

```text
(CHEST, LEFT_SHOULDER, RIGHT_SHOULDER, BACK)
= (0.25, 0.25, 0.25, 0.25)
```

当前 12 秒 episode 只有一次推力事件。完成后进入 DONE，不在同一 episode 再次抽样。

## 7. WAIT → DRIVE → RETRACT → DONE 状态机

### WAIT

- 装置停放在 `z=-2 m`。
- 从阶段指定范围均匀抽取等待时间。
- WAIT 时 `qfrc_applied=0`。
- 等待结束时抽样目标、固定导轨姿态、设置 2 cm 间隙并进入 DRIVE。

### DRIVE

- 使用 ramp 后的正向 `v_ref` 和虚拟弹簧阻尼力。
- 接触发生后持续测量、滤波并判断是否完成卸力。
- 到达 `u=0.375 m` 只会被 joint limit 阻挡并置位 `travel_limit_reached`，不会因此结束 episode。

### RETRACT

- 只在卸力完成条件成立后进入。
- 使用 `v_ref=-0.15 m/s` 撤回。
- 撤回产生的主动降力不能计为机器人卸力成功。

### DONE

- 当球回到导轨起点附近且速度足够低时进入。
- 清零外力并把装置移到地面以下。

## 8. 接触测量、滤波与安全阈值

### 8.1 接触传感器

建立球头到机器人整棵 pelvis 子树的接触传感器：

```text
primary:   entity=rail_pusher, geom=sphere
secondary: entity=robot, subtree=pelvis
fields:    found, force, pos, normal
reduce:    none
num_slots: 8
history_length: 4 physical substeps
```

只使用 contact-frame force 的法向分量 `abs(force[..., 0])`。对所有接触槽求和，不能把切向摩擦力混入卸力指标。

在一个控制周期的 4 个物理子步内：

- `control_force`：各子步总法向力的平均值；
- `raw_force_peak`：各子步总法向力的最大值；
- `raw_force_peak` 用于紧急力检测，避免平均值掩盖瞬态峰值。

### 8.2 EMA 滤波

```text
F_filtered[t] = 0.8 * F_filtered[t-1] + 0.2 * F_control[t]
dF_dt = (F_filtered[t] - F_filtered[t-1]) / 0.02
```

| 参数 | 值 |
|---|---:|
| `ema_alpha` | `0.8` |
| 接触激活阈值 `force_on_ratio` | `0.0075 * M_robot*g` |
| 完成力阈值 `force_done_ratio` | `0.03 * M_robot*g` |
| 完成力变化率阈值 | `0.05 * M_robot*g / s` |
| 紧急力阈值 | `0.30 * M_robot*g` |
| 接触锁存 | 连续 `3` 个控制步 |
| 紧急力判定 | 连续 `3` 个物理子步超过阈值 |
| 最短 DRIVE 时间 | `0.5 s` |
| 最短参考行程（二选一条件） | `0.03 m` |
| 完成稳定时间 | `0.4 s` |

卸力完成必须同时满足：

1. 接触已连续激活 3 个控制步并锁存；
2. 事件历史峰值至少达到 `0.03 mg`，防止从未加载也被判成功；
3. DRIVE 时间达到 `0.5 s`，或参考行程达到 `0.03 m`；
4. 当前滤波力低于 `0.03 mg`；
5. `abs(dF/dt) < 0.05 mg/s`；
6. 条件 4、5 连续保持 `0.4 s`。

紧急力超过 `0.30 mg` 且持续 3 个物理子步时，任务应触发紧急终止并将该步任务奖励清零。

## 9. 力方向、作用位置与策略接口

多个接触点的合成法向和力加权作用位置为：

```text
F_i = abs(contact_force_i.normal_component)
F_vector = sum_i(F_i * contact_normal_i)
direction_world = F_vector / max(norm(F_vector), eps)
position_world = sum_i(F_i * contact_position_i) / max(sum_i(F_i), eps)
```

随后使用机器人当前 heading 转到局部坐标：

```text
direction_local = R_heading^T * direction_world
position_local  = R_heading^T * (position_world - root_position_world)
```

单帧接触观测为 8 维：

```text
[active,
 clip(F_filtered / (M_robot*g), 0, 0.4),
 direction_local_x, direction_local_y, direction_local_z,
 position_local_x,  position_local_y,  position_local_z]
```

无接触时 8 维必须严格为零。Actor 和 Critic 都使用 4 帧接触历史：

- Actor：力幅值乘以均匀 `±2%` 相对噪声；方向加标准差 `0.5°` 的高斯噪声后重新单位化；位置加标准差 `3 mm` 的高斯噪声；无接触零值保持不变。
- Critic：相同物理量，无接触噪声。
- 不向任何网络提供预定目标、启动倒计时、参考位置、参考速度或 future force。

如果新实验只复现机械装置而不复现 RL 接口，可省略观测噪声，但不得改变力聚合和控制器使用的滤波值。

## 10. Conditional 接触区域分类

如果需要复现五条件 SMP 任务，还需要第二个区域传感器，将实际接触分为左肩、右肩、胸口和背部。不能直接使用预定 PushTarget 作为条件标签。

- 左肩：`left_shoulder_yaw_collision` 与球的实测法向力；
- 右肩：`right_shoulder_yaw_collision` 与球的实测法向力；
- 躯干：`torso_collision` 与球的实测法向力；
- 胸/背：用实测球接触位置在 torso 局部坐标的 `x >= 0` 或 `< 0` 区分；
- 多部位同时接触：取当前滤波后法向力最大的区域。

区域力以机器人重量归一化，并使用相同形式的 `0.8/0.2` EMA。条件迟滞参数：

| 项目 | 值 |
|---|---:|
| 激活阈值 | `0.0075 mg` |
| 解除阈值 | `0.005 mg` |
| 激活/解除 dwell | 连续 3 个控制帧 |
| 条件历史长度 | 10 个控制帧 |
| 过渡条件 | 历史未全部属于新条件时使用独立 NULL 条件 |

条件 ID 与 PushTarget ID 不同，必须使用：

```text
0 = stand/no contact
1 = left shoulder
2 = right shoulder
3 = chest
4 = back
5 = NULL/unconditional transition branch
```

## 11. 最小配置清单

另一个 Agent 实现时，可直接以此对象作为冻结配置：

```python
rail_cfg = {
    "sphere_radius": 0.10,
    "sphere_mass": 5.0,
    "friction": 0.2,
    "reference_speed": 0.05,          # C；B1/B2 见阶段表
    "stiffness": 700.0,
    "damping": 118.32159566199232,
    "max_force_weight_ratio": 0.15,   # C；B1/B2 见阶段表
    "max_travel": 0.375,
    "ramp_time": 0.25,                # C；B1/B2 见阶段表
    "retract_speed": 0.15,
    "start_distance": 0.35,           # 当前因 surface gap 非空而不生效
    "start_surface_gap": 0.02,
    "conditional": True,
    "wait_range_s": (1.0, 3.0),
    "target_probabilities": (0.25, 0.25, 0.25, 0.25),
    "push_direction_mode": "front_to_back",
    "ema_alpha": 0.8,
    "force_obs_clip": 0.4,
    "force_on_ratio": 0.0075,
    "force_done_ratio": 0.03,
    "force_rate_done_ratio_s": 0.05,
    "force_emergency_ratio": 0.30,
    "contact_latch_steps": 3,
    "emergency_substeps": 3,
    "min_drive_time_s": 0.5,
    "min_reference_travel": 0.03,
    "stable_time_s": 0.4,
}
```

`stationary_reward_weight=0.1` 和 `stationary_speed_scale=10.0` 属于任务奖励，不属于机械装置；如果要完整复现 Conditional v2 PPO 环境，应同时设置。A 阶段的 stationary weight 为 `0.9`。

## 12. 与具体框架无关的控制伪代码

```text
on_reset:
    phase = WAIT
    park_base_at(z=-2)
    set_joint_state(u=0, u_dot=0)
    qfrc_applied = 0
    wait_left = uniform(wait_min, wait_max)

each_control_step(dt=0.02):
    contact = aggregate_normal_contacts_over_last_4_physics_steps()
    update_force_ema_and_safety(contact)

    if phase == WAIT:
        qfrc_applied = 0
        wait_left -= dt
        if wait_left <= 0:
            target = categorical(target_probabilities)
            n_world = fixed_direction_from_current_robot_heading(target)
            place_rail_at_exact_surface_gap(target, n_world, gap=0.02)
            set_joint_state(u=0, u_dot=0)
            u_ref = 0
            v_ref = 0
            drive_time = 0
            phase = DRIVE

    elif phase == DRIVE:
        drive_time += dt
        v_ref = reference_speed * clamp(drive_time / ramp_time, 0, 1)
        u_ref += v_ref * dt
        apply_force(clamp(k*(u_ref-u) + c*(v_ref-u_dot), -F_max, +F_max))
        if unloading_completion_is_stable():
            phase = RETRACT

    elif phase == RETRACT:
        v_ref = -0.15
        u_ref = max(0, u_ref + v_ref*dt)
        apply_force(clamp(k*(u_ref-u) + c*(v_ref-u_dot), -F_max, +F_max))
        if u < 0.005 and abs(u_dot) < 0.02:
            phase = DONE
            park_base_at(z=-2)
            qfrc_applied = 0

    elif phase == DONE:
        qfrc_applied = 0
```

接触刷新必须在一个控制步内幂等：observation、reward、termination 和 metric 应读取同一个缓存，不能各自再次更新 EMA，否则滤波速度会随 manager term 数量改变。

## 13. 复现验收测试

实现完成后至少执行以下检查：

1. **模型结构**：恰好 1 个 slide joint、0 个旋转 joint；joint range 为 `[0, 0.375]`。
2. **球体参数**：实际编译后的半径 `0.10 m`、质量 `5 kg`、joint damping `0`。
3. **表面间隙**：胸、背、左右肩四个目标初始几何表面间隙均为 `0.020 m`，不是球心距离。
4. **推进方向**：胸/左右肩为 heading 局部 `-x`，背部为 `+x`；左右肩不是侧向推。
5. **导轨约束**：整个事件中球心相对基座的局部 `y-z` 范数 `<1e-5 m`。
6. **动力学响应**：固定机器人时球可被 PD 推动；机器人施加反力时实际球速能够低于参考速度并可短暂为负。
7. **力上限**：记录的 `abs(qfrc_applied)` 永不超过阶段 `ratio*M_robot*9.81`。
8. **接触测量**：仅法向力进入卸力指标；4 个物理子步的平均值和峰值分别用于滤波与紧急检测。
9. **零观测**：接触前和 DONE 后，8 维接触观测严格为零。
10. **四区域 smoke**：固定机器人姿态，以 B1 `0.01 m/s` 分别触发胸、背、左右肩，确认均产生真实接触且条件分类正确。
11. **安全**：连续 3 个物理子步超过 `0.30 mg` 能触发紧急终止。
12. **行程语义**：到达 `0.375 m` 只置位遥测，不以“到限”作为失败终止。

本仓库已有 Conditional smoke 的参考结果：64 个环境下四部位均发生真实接触，首次条件激活约 `2.54–2.86 s`，最大导轨横向误差约 `7.45e-9 m`。

## 14. 常见错误

- 把球改成 kinematic object：机器人将无法通过反作用力卸载弹簧。
- 同时设置 joint damping 和虚拟 `c`：会得到过阻尼装置。
- 将 PD 截断为 `[0, F_max]`：撤回和速度阻尼行为与当前装置不一致。
- 用固定球心距离代替 2 cm 表面间隙：不同目标半径会产生不同接触时刻。
- 肩部瞄准 capsule 中心：默认弯肘姿态可能先撞前臂。
- 每帧跟随机器人更新导轨：会把固定外部装置错误地变成随动执行器。
- 把预定目标当作实测接触条件：会向策略/Prior 泄漏未来信息。
- 将摩擦切向力并入法向力：改变摩擦时会同时改变卸力判据。
- 对每个 manager term 重复更新接触 EMA：会产生不可控的滤波时间常数。
- 到达导轨上限即终止：这不符合当前实验的“到限只遥测”语义。


复现时应优先满足本文的冻结数值与验收测试；如果修改球半径、质量、刚度、阻尼、速度、力上限、表面间隙或接触滤波，应将其视为新的装置版本，而不是同一装置的等价实现。
