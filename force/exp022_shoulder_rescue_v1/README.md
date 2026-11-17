# EXP-022 肩部接触迁移补偿：实验候选

这包记录冻结 EXP-022 胸推 Actor 在持续球头推压下的力估计联调。原 [V1 胸推估计器](../exp022_chest_v1/README.md)在球头从胸前移到右肩或右肘时，常把高峰值力判为无接触；策略输入仍使用固定胸前作用点和法向。这里保留 V1，在高置信度肩部迁移时补一个三维力分支，并给出实验性的肩部接触映射。**当前不替换默认 V1，也不作为已通过卸力验收的策略输入。**

## 模型和接口

模型读取同一控制步的 50 Hz `x930` 和 `tau_hist_10x29`，在 `torso_link` 坐标输出三维合力。肩部分类器只用这两项可观测量构成的 273 维因果特征；训练时的真实接触几何、接触力、未来帧都不参与推理。选定门槛为肩部概率 ≥0.95、V1 力模长 <10 N、补偿分支建议力 ≥10 N；其他力帧保留 V1。肩部作用点是训练集校准的单个常量，法向由可观测特征回归。它仍是单点近似，不是多接触点识别。

```bash
uv sync --locked --extra force-estimator
(cd force/exp022_shoulder_rescue_v1 && sha256sum -c SHA256SUMS)
```

```python
from smp.force_estimation.exp022_shoulder_rescue import Exp022ShoulderRescueEstimator

estimator = Exp022ShoulderRescueEstimator(
    "force/exp022_shoulder_rescue_v1/candidate.joblib",
    expected_sha256="c74e7ca92771b17849e4cbf4ddbd2425dff4e10056c0adc4542e9b4de940327c",
)
estimator.reset()  # 每回合开始时调用
output = estimator.step(x930, tau_hist_10x29)
force_body_n = output["force_body_n"]
```

输出还包含 `contact_probability`、策略接触法向与点、`shoulder_probability`、`shoulder_force_rescued` 和 `shoulder_mapping_active`。这段代码只说明模型可加载；现有胸推策略仍默认使用 V1。打包模型在 253 号闭环输入上逐帧重放 256 帧，力、门控和映射与原采集结果完全一致，见[打包重放记录](evaluation/package_replay_smoke.json)。

## 数据和冻结验证

训练补充了 213–236 号种子的真值/估计输入完整回合，保留 V1 原训练数据；237–244 号新种子及旧 V1 验证集用于选型。一个候选冻结后，才打开 245–252 号独立种子。训练、门槛、模型和评估身份见[预采集清单](evaluation/frozen_precollection.json)、[候选合同](evaluation/region_rescue_contract.json)、[封存清单](evaluation/candidate_frozen_pretest.json)。直接整体重训的线性候选在旧场景误报和首次接触漏检上退化，因此未采用；[选型结果](evaluation/linear_selection.json)保留了失败候选。

独立封存测试的 8 条**估计输入保存轨迹**如下。力误差是先算每回合真实合力 ≥10 N 帧的三维向量误差均值，再取回合中位数。峰值漏检表示真实峰值帧估计力 <10 N。两列在同一批保存轨迹上推理，可以直接比较估计器。

| 指标 | V1 | 肩部候选 |
| --- | ---: | ---: |
| 峰值漏检 | 6/8 | 0/8 |
| 受力回合三维向量 MAE 中位数 | 22.39 N | 11.66 N |
| 最差回合三维向量 MAE | 25.60 N | 21.01 N |
| 峰值三维向量误差中位数 | 94.49 N | 13.53 N |
| 首次受力帧漏检 | 5/8 | 5/8 |
| 接触前 ≥10 N 误报回合 | 0/8 | 0/8 |
| 实际零力时 ≥10 N 误报帧 | 19 | 21 |
| 峰值策略作用点距离中位数 | 18.32 cm | 4.16 cm |
| 峰值策略接触法向夹角中位数 | 25.75° | 10.43° |

旧验证集 24 回合的受力 MAE 中位数仍为 7.99 N，首次受力漏检仍为 8/24，零力误报 38→39 帧。[逐回合结果](evaluation/locked_offline_per_episode.csv)和[完整审核](evaluation/locked_offline_result.json)保留全部回合。候选新增检出 70 个受力帧，这些帧的三维力方向平均误差 13.64°；V1 原本已检出的 84 帧仍为原输出，平均方向误差 29.85°，[方向审核](evaluation/locked_direction_audit.json)单独列出。因此本次没有修复原有已检出帧的方向误差。

## 闭环结果与映射排查

冻结后用新种子 253–260 做 20 N 标称推力、5 kg 球头的四路闭环：遮蔽、真值、V1 估计、候选估计。实际力由 MuJoCo 接触求解得到，可能明显高于标称推力。四路每组 8/8 发生接触，接触前估计误报均为 0。候选相对 V1：真实峰值帧漏检 **8/8→0/8**，受力回合向量 MAE 中位数 **22.56→15.99 N**，最差回合 **28.85→17.90 N**。两路是不同闭环轨迹，误差只描述各自轨迹。见[闭环审核](evaluation/live_result.json)、[逐回合表](evaluation/live_per_episode.csv)和[253 号力曲线](evaluation/live_seed253_force_trace.png)。

持续推压仍未通过策略验收：遮蔽输入 4/8 失败，真值、V1、候选输入均 **8/8 失败**。遮蔽输入的连续躯干 ≥10 N 受力中位数为 5.73 s；真值为 0.08 s，候选为 0.13 s。真值输入在可持续物理加载的种子上也很快触发自碰撞、工作区或关节限制，当前没有稳定的真值输入策略供“真值换估计”PPO 阶段直接使用。

又在标称 15 N 和 25 N 各 4 个新种子上检验同一冻结候选。两档峰值漏检均由 V1 的 4/4 降为 0/4，受力 MAE 中位数分别从 18.91→13.01 N、22.96→13.27 N；但真值、V1、候选策略全部失败。15 N 遮蔽输入有 3/4 个回合满足整回合安全、持续 ≥1 s 和峰值 ≤80 N 的严格对照条件；25 N 遮蔽输入 0/4 满足。[分档审核](evaluation/force_shift_result.json)包含失败回合和实际峰值。

25 N 的完整候选 4/4 同时触发关节硬限位与自碰撞，V1 为 0/4 硬限位。保持候选力、把策略点和法向都恢复为 V1 胸前常量后，硬限位为 0/4；只使用肩部点或只使用肩部法向也都是 0/4，但这些设置仍全部失败。不同设置会生成不同闭环轨迹，当前证据提示完整点与法向组合存在策略分布外风险，尚未建立单项因果结论。见[力与映射拆分](evaluation/mapping_ablation_result.json)和[点/法向拆分](evaluation/point_normal_ablation_result.json)。

原 EXP-022 短脉冲协议的新种子 261–268 也做了回归：V1 与候选均 8/8 完成接触事件、各 1/8 失败、0/8 接触前误报；肩部补偿未在这 8 条轨迹触发。有效受力连续时长中位数只有 0.04 s，这组用于确认旧协议未受影响，[审核](evaluation/original_safety_result.json)保留数据。

## 当前使用范围

把此包作为运动域肩部迁移的**可加载实验候选**和下一轮策略训练数据来源。默认策略输入仍用 V1。下一步先用真值输入把持续球头协议下的 Actor 稳定性做出来，并明确肩部点/法向应如何进入策略；再检查首次接触漏检和已检出帧的方向误差，最后做估计输入 PPO 微调及独立闭环验收。以上只覆盖胸前导轨球头导致的右肩/右肘接触迁移仿真，不代表真实 G1、其他受力部位、多独立外力或卸力成功。

实验原始逐帧数据与训练脚本保存在本地移动硬盘 `files/中关村相关/力估计器结果/2026-11-17_EXP022_肩部迁移补偿_冻结闭环与消融/`；GitHub 只放模型、冻结清单、逐回合审核、复现实验代码快照和曲线。三次评估所用脚本版本分别见 `evaluation/code_snapshot_frozen/`、`evaluation/code_snapshot_force_only/` 和 `evaluation/code_snapshot_point_normal/`；文件哈希与对应冻结清单一致。
