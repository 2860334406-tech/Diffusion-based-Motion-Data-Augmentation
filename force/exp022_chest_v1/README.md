# EXP-022 胸推力估计器：运动域校正与策略输入映射

这一包用于 `policy/chest_speed_gradient_v1/` 的胸前球头推力实验。输入来自 EXP-022 机器人每个 20 ms 控制步的 10 帧本体观测和关节执行器力矩历史。它输出 `torso_link` 坐标系的三维外力估计，并提供胸推策略接触观测所需的法向和作用点。模型和映射均已冻结；当前完成了保存轨迹重放与策略输入离线对照。

## 文件

| 文件 | 作用 |
| --- | --- |
| `candidate.joblib` | 接触概率分类器、受力帧三维力回归器和输入归一化。 |
| `policy_mapping_calibration.json` | 胸推法向和作用点的躯干坐标常量。 |
| `manifest.json`、`SHA256SUMS` | 模型身份、哈希、输入输出及适用范围。 |
| `evaluation/` | 开发分组、封存测试、逐回合指标、源文件哈希和力跟踪图。 |

加载时检查模型、映射和特征代码的 SHA-256。原 V3.92 包仍在 [`force/v392/`](../v392/)；它适用于原站立 SuperMimic 数据，本包针对 EXP-022 胸运动域。

## 为什么重新校正

原 V3.92 在站立 G1、1 kg 球头接触数据上训练。EXP-022 策略的上身姿态与动作分布不同，实验还包含 5 kg 球头。逐层重放证实输入关节顺序、50 Hz 历史、力矩和 `torso_link` 坐标变换一致；基础网络已在接触前报出几十牛顿，低接触概率补偿分支还出现过约 1.2 kN 的外推误报。单纯平移姿态输入不能稳定修复。V2 用 EXP-022 的真值输入和遮蔽输入回合共同训练，选择时直接检查策略约 2.45 N 的滤波接触开关。

训练数据为种子 61–108 和 145–156 的 60 个种子、120 个回合；157–168 的 24 回合用于选型；169–180 的 24 回合在冻结模型和门槛后才评估。前一版在另一组新回合的接触开关误触发为 6/24，已记录为[开发失败](evaluation/v1_actor_input_failed.json)，相关回合从此只作为 V2 开发数据。训练和验证按完整种子分组，每个种子的真值与遮蔽策略输入都留在同一组。完整身份见 [`evaluation/training_manifest.json`](evaluation/training_manifest.json)、[`evaluation/v2_redesign_precollection.json`](evaluation/v2_redesign_precollection.json)。

## 封存测试结果

前胸球头推手、标称驱动力 10/20/30/40 N、球质量 1/5 kg；12 个新种子，每种策略输入模式各一次，24 回合全部发生真实接触。真值是四个 5 ms 子步的三维接触合力在控制步内的平均值。受力帧定义为真值合力模长 ≥10 N；每回合先算均值，再取回合中位数。

| 指标 | V3.92 | 本包 V2 |
| --- | ---: | ---: |
| 接触前最大估计力 | 1218.25 N | 0 N |
| 受力回合中位三维向量 MAE | 62.18 N | 8.60 N |
| 最差回合三维向量 MAE | 86.20 N | 16.28 N |
| 本包峰值幅值误差中位数 | — | 9.69 N |
| 本包检出受力帧的方向角中位数 | — | 13.39° |
| 本包首次 ≥10 N 受力帧漏检 | — | 8/24 回合 |

策略读取的是 8 维接触项及其 4 帧历史。它的真值方向来自接触法向，力标量来自法向力之和；三维合力方向和模长不能逐项直接代替。此包用估计合力的模长，加上训练集校准的胸推法向 `[-0.99935, 0.03535, -0.00684]` 和躯干作用点 `[0.09977, -0.00164, 0.13541]` m，再按原策略的滤波、heading 坐标和历史规则生成输入。

| 策略输入指标，24 回合中位数 | 新估计力 + V3.92 作用点及合力方向 | 新估计力 + 胸推映射 |
| --- | ---: | ---: |
| 单帧 8 维误差范数 | 0.618 | 0.154 |
| 四帧 32 维历史误差范数 | 1.265 | 0.353 |
| 双方都检出时的方向角 | 33.00° | 8.41° |
| 双方都检出时的作用点距离 | 16.38 cm | 2.31 cm |
| 策略力标量误差 | 2.73 N | 2.73 N |
| 接触前开关误触发回合 | 0/24 | 0/24 |

全部预先设定的力与策略输入门槛均通过。逐回合数据在 [`evaluation/v2_test_metrics.json`](evaluation/v2_test_metrics.json)、[`evaluation/v2_force_locked_audit.json`](evaluation/v2_force_locked_audit.json)、[`evaluation/v2_actor_input_locked_audit.json`](evaluation/v2_actor_input_locked_audit.json)。[跟踪图](evaluation/force_tracking_locked.png)同时展示最差受力回合 `176_oracle` 和首次检出延迟最长的 `169_masked`；图中 V3.92 曲线超出坐标轴的部分有标注。原始轨迹未上传仓库，SHA-256 见 [`evaluation/source_manifest.json`](evaluation/source_manifest.json)。

## 加载和接口

从仓库根目录运行：

```bash
uv sync --locked --extra force-estimator
sha256sum -c force/exp022_chest_v1/SHA256SUMS
uv run --locked --extra force-estimator scripts/smoke_exp022_chest_estimator.py
```

```python
from smp.force_estimation import load_exp022_chest_estimator

estimator = load_exp022_chest_estimator()
estimator.reset()  # 每个回合开始时调用
output = estimator.step(x930, tau_hist_10x29)
force_body_n = output["force_body_n"]
normal_for_actor_body_n = output["policy_normal_force_body_n"]
point_for_actor_body_m = output["policy_point_body_m"]
```

`x930` 的切片和关节顺序遵循 [`force/v392/README.md`](../v392/README.md) 的 SuperMimic 合同；`tau_hist_10x29` 是同步的关节空间 `qfrc_actuator`。本模型选用 273 维因果特征，门控阈值 0.6，回合开始 10 帧输出为零。`Exp022ContactAdapter` 接收上述策略法向和作用点以及当前 torso/root 位姿，生成 8 维当前帧和 32 维历史。单回合保存轨迹重放的最大力分量差为 4.77×10⁻⁶ N。

## 接下来

主线第 1 步的离线输入对照已经通过。下一项是把估计接触项实际送入冻结 EXP-022 Actor 跑配对闭环回合，再比较接触持续时间、摔倒和动作变化。现在的遮蔽模式让 Actor 在整个回合都看不到真实接触；它还没有形成“估计值影响动作、动作再影响估计值”的闭环。10 N 低力档仍有最长 10 帧的首次检出延迟，真实零力但已接触后的 ≥10 N 假阳性仍有 61 帧。封存回合中 20/24 在发生接触后触发失败终止；这些轨迹只用于估计与输入误差评估，不能作为卸力性能证据。

胸推法向与作用点是本场景固定几何的校准结果。肩、背、随机多点、多个独立外力、真实 G1 传感器可用性及硬件时序尚未在本包验证。当前 Python/scikit-learn 逐实例推理适合单回合联调；多环境 PPO 训练前还需实现并核对批量推理。
