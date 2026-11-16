# G1 球头接触力估计器：V3.92 冻结包

这是站立 G1、球头导轨物理接触实验使用的冻结推理链，供 `integration/force` 分支加载和后续联调。包内同时保留 V3.57、V3.82、V3.92 三档输出。**本次验收范围是仓库内加载与保存轨迹重放**；当前 SMP 卸力任务仍使用原有接触传感器，尚未用估计力替换策略观测。

## 文件与模型身份

| 文件 | 来源与作用 |
| --- | --- |
| `base.pt` | V3.5 基础双潜变量网络，训练 checkpoint epoch 14；内置 930D 与关节力矩归一化。 |
| `calibrator.joblib` | V3.15 历史校准，使用当前和过去两帧的基础力、点预测。 |
| `residual_v338.pt` | V3.38 本体状态残差，epoch 15。 |
| `lowp_classifier.joblib`、`lowp_force_head.joblib` | V3.57 低接触概率补偿。 |
| `residual_v370.pt`、`benefit_gate.joblib` | V3.82 高概率方向修正，V3.70 残差 epoch 10、门控阈值 0.5、实际增益 0.25。 |
| `force_control_v391.pt`、`region_classifier.joblib`、`region_norms.npz` | V3.92 胸背目标区域变化修正；力分支 epoch 3，区域分类仅用于门控。 |
| `manifest.json`、`SHA256SUMS` | 原始权重身份、文件哈希、输入输出和适用范围。 |
| `evaluation/` | 四组物理球头测试的逐回合指标、汇总及源文件哈希。 |

权重按原始字节复制，加载器先核对全部十个权重与归一化文件的 SHA-256，再反序列化。模型训练发生在独立的 SuperMimic 力估计实验目录；本仓库提供的是冻结推理实现，不把本次代码提交当成训练版本。`joblib` 文件按保存它们时的 scikit-learn 1.9.0 加载，需 Python 3.11 或更新版本。原训练 checkpoint 中的绝对路径是训练记录，不参与本仓库加载。

## 选用依据与逐回合评估

我们在相同轨迹上比较了能输出完整三维力的 V3.5、V3.15、V3.38、V3.57、V3.82、V3.92。V3.89 是接触检测器，没有力输出，不列为候选。默认加载 V3.92，是针对**当前胸背目标**的有限选择：封存胸背 56 回合中，最差受力单帧的三维向量误差由 V3.57 的 442.7 N、V3.82 的 363.1 N 降到 224.9 N；最差回合受力帧向量 MAE 依次为 20.1、18.1、17.4 N；峰值帧平均方向误差依次为 24.0°、20.2°、15.4°。

V3.92 并非所有指标最好。胸背封存集**最大峰值幅值绝对误差**为 166.7 N，高于 V3.82 的 128.0 N 和 V3.57 的 115.4 N；峰值帧方向误差最大仍有 155.2°。对应的胸推峰值帧，真实力 265.4 N，V3.82 预测幅值 144.0 N、方向误差 122.1°，V3.92 预测幅值 98.7 N、方向误差 55.6°。方向改善伴随峰值幅值低估加重，后续控制不能忽略。

| 测试 | 回合 | 接触前最大预测力 | 首次真实接触帧预测力 <10 N | 最差回合受力帧向量 MAE | 最差受力单帧向量误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 封存胸背，V3.92 | 56 | 0.23 N | 0 | 17.4 N | 224.9 N |
| 左右肩独立测试，V3.82 | 47 | 0.24 N | 7 | 12.3 N | 100.3 N |
| 右肩起点间隙修正，V3.92 | 64 | 0.23 N | 8 | 13.3 N | 67.5 N |
| 六方向随机偏移球头，V3.92 | 6 | 0.21 N | 0 | 40.8 N | 1149.0 N |

首次接触按求解器第一次实际碰撞定义，接触前只统计该帧以前；“漏检”在表中按首次接触帧预测幅值低于 10 N 计。接触概率低于 0.5 的次数另见逐回合 CSV。无接触误报并非全发生在首次接触之前：封存胸背集的 9,352 个真实零力帧中，183 帧预测力达到 10 N；六方向集有 27 帧。六方向左肩后方来力的一帧真实力为 `[99.66, 25.76, -37.89]` N，V3.92 输出 `[-662.01, -273.43, 768.68]` N，向量误差约 1149 N。该异常在 V3.15 历史校准后放大，V3.92 没有消除。

完整的逐回合结果在 [`evaluation/candidate_episode_metrics.csv`](evaluation/candidate_episode_metrics.csv)，汇总在 [`evaluation/candidate_summary.json`](evaluation/candidate_summary.json)，原缓存和预测文件的 SHA-256 在 [`evaluation/source_manifest.json`](evaluation/source_manifest.json)。三版预测在各测试内按 `run_id`、`step` 和真实力逐行核对一致。六方向每条件只有一条随机轨迹，不能据此声称总体泛化精度。左右肩目标不触发 V3.92 胸背修正；前一组左右肩测试只保存了 V3.82 输出，V3.92 与其相同属于门控规则推断。

## 输入、输出与坐标系

每个实例只处理一个 50 Hz 回合；回合开始调用 `reset()`，之后每 20 ms 调用一次 `step()`。输入只包含当前和过去的可观测量：

| 输入 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `(930,)` | 原 SuperMimic 策略观测：10 帧、按观测项连续排列。 |
| `tau_hist` | `(10, 29)` | 与 `x` 同步的 G1 关节执行器力矩历史。 |
| `target_site_id` | 标量 | 计划推点：0 前胸、1 左肩、2 右肩、3 后背；V3.92 必填。 |

`x` 的切片依次为相对默认姿态的关节角 `[0:290]`、关节角速度 `[290:580]`、机体角速度 `[580:610]`、投影重力 `[610:640]`、上一动作 `[640:930]`；每项内部均为从旧到新的 10 帧。原采集用 SuperMimic 策略观测，训练时包含观测噪声；`tau_hist` 是 `env.step` 后记录的 `qfrc_actuator`。训练时的原始关节顺序和动作语义必须保持一致，不能仅凭 930 这个维度判断兼容。

`force_body_n` 输出的是机器人受到的三维外接触合力，单位 N，表达在**控制步结束时的 `torso_link` 坐标系**；`point_body_m` 是基础模型输出的合力点估计，单位 m、同一坐标系；`contact_probability` 未被 V3.82/V3.92 修改。评估真值由四个 5 ms 物理子步的三维接触力平均，再旋转到控制步末的 `torso_link`。求解器真值、实际接触几何体和未来状态都不参与推理。`target_site_id` 是导轨计划目标，不代表实际接触点。

## 加载与重放

从仓库根目录运行：

```bash
uv sync --locked --extra force-estimator
sha256sum -c force/v392/SHA256SUMS
uv run --locked --extra force-estimator scripts/smoke_force_estimator.py
```

加载后调用：

```python
from smp.force_estimation import load_force_estimator

estimator = load_force_estimator(version="v392", device="cpu")
estimator.reset()
output = estimator.step(x, tau_hist, target_site_id=0)
force_in_torso_link_n = output["force_body_n"]
```

保存轨迹的重放检查：

```bash
uv run --locked --extra force-estimator scripts/smoke_force_estimator.py \
  --cache /absolute/path/to/cache/test.npz \
  --predictions /absolute/path/to/evaluations/v392/test_predictions.npz
```

脚本重放第一个完整回合，要求力最大分量差 <0.001 N、点 <0.00001 m、接触概率 <0.0001。已在本地额外重放封存胸推、封存背推和六方向中最差的左肩后推，共三条完整回合；对原 V3.92 预测的最大差分别不超过 0.00025 N、0.000001 m、0.00001。评估 NPZ 不随仓库上传，因此在没有原始数据的机器上先运行无参数加载检查。

## 与现有策略的接口边界

`policy/chest_speed_gradient_v1/` 的 Actor 接收 8 维接触观测：接触开关、按机器人重量归一化并滤波的力标量、heading 局部方向三维、相对 root 的 heading 局部位置三维，另带 4 帧历史。本模型给的是 `torso_link` 三维合力及点，接触标量和正负方向的物理定义也不同。当前 `smp` actor/critic 观测还额外包含基座线速度，顺序和历史方式与模型所需的 930D 不同。估计器原球头质量为 1 kg，而已归档胸推策略的球头为 5 kg。下一阶段须先构造并逐项验证 SuperMimic 输入，再验证力标量、滤波、坐标变换、位置基点、无接触处理、4 帧时序转换及球头参数变化；这次没有把模型输出直接写入策略观测。

目前只有站立 SuperMimic 仿真与球头导轨物理接触的证据。真实 G1 的力矩可采集性、行走、多个独立同时接触、非球形物体，以及用估计力驱动卸力策略后的效果均未验证。
