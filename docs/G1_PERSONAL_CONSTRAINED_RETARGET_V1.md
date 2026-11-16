# 个人采集 G1 动作二次受约束重定向 v1（EXP-034）

本流水线将 `datasets/G1-GVHMR-GMR/` 的 27 条个人采集动作作为软参考，在 SMP 当前 G1 模型上重新求解，随后进行原执行器、自由 root、无外力 MuJoCo 验证。仅接受实际执行轨迹；不启动 Prior/PPO。

## 已确认的动作定义

- 双脚原地固定，六自由度足端约束；允许腰、髋和骨盆协同转动。
- 不设置 pelvis yaw 上限，不固定相对腰角度或速度；均受模型关节限位、支撑和动力学约束。
- 左右肩标记点固定在 torso 上，初始位置取对应肩碰撞几何中心。摆动手臂不能改变该标记点相对 torso 的位置。
- 以初始 pelvis heading 的前向轴测量后退量，同时计算世界位移和扣除 pelvis 平移后的位移。两者终点均须 ≥0.015 m；不能仅靠平移 root 冒充转动。
- 起转时间由源 torso signed yaw 超过 `max(0.02 rad, 其p95的5%)` 持续10帧检测。首1秒须中立，最后保留1秒保持验收。参考信号仅使用固定11帧/3阶 Savitzky–Golay 滤波，不改源文件。
- 原 GMR 的 root/torso 旋转与上身关节变化作为软目标；下肢重新优化，手臂以安全中立姿态加源变化为参考。保留原始时长，位置/关节 PCHIP 与旋转 SLERP 重采样到50Hz，不拼接、镜像扩充、放大角度或补末帧。
- 末段稳定目标：使用最后2秒中的前1秒已观察姿态的中位值，在该1秒内以五次曲线平滑收敛，最后1秒保持该软目标。若源视频末尾开始回中，二次动作以此前观察到的姿态稳定保持；这一修改仅发生于求解目标，原始数据和实际执行状态均不复制、不补帧，原时长不变。

## 物理边界

IK 使用 Mink 1.4.0、每帧4个0.005秒子步，COM位于内缩1cm的双足支撑凸包。碰撞约束覆盖所有有效非相邻几何对，并遵守原模型碰撞排除关系，含手腕/手掌与髋部。

| 检查 | 门槛 |
|---|---|
| IK足端位姿误差 | ≤2mm / 1° |
| 实际关节限位余量 | ≥1e-4 rad |
| root最低高度 / 最大roll、pitch | ≥0.60m / ≤20° |
| root最大平面位移 | ≤0.10m |
| 每脚接触比例 / 连续支撑 | ≥95% / 始终至少一脚接触 |
| 足部水平漂移 / yaw漂移 | ≤0.015m / ≤2° |
| 全关节 / waist跟踪RMSE | ≤0.08 / 0.04 rad，相对安全IK参考 |
| 力矩p99 / 饱和帧比例 | ≤95%上限 / ≤1%（一帧任一电机饱和即计入） |
| 电机速度 | 不超现有G1电机限制 |
| 最后1秒root速度p95 | ≤0.05m/s |
| 最后1秒腰、骨盆yaw速度p95 | 各≤0.10rad/s |
| 保持肩部后移 / 三维位置偏差 | ≥0.014m / ≤0.005m |
| 自碰撞 / 非脚部触地穿透 | ≤0.1mm / ≤2mm |

仿真步长0.005秒、控制周期0.02秒，地面摩擦0.6。首帧预稳定1秒、末帧额外保持1秒只用于验证，不进入训练数据。出现NaN/Inf、MuJoCo警告或自动重置即拒绝候选，即使数组随后恢复为有限值。

每条最多6个候选，依次为`(站距m,膝屈曲rad)=(0.24,0.70),(0.24,0.55),(0.22,0.70),(0.26,0.70),(0.22,0.55),(0.26,0.55)`。seed42控制可复现的初始手臂微小变化，接受首个通过者，拒绝尝试保留。

## 执行与恢复

```bash
uv run --no-sync python scripts/run_g1_personal_retarget.py --stage smoke --workers 3
uv run --no-sync python scripts/run_g1_personal_retarget.py --stage full --workers 3
uv run --no-sync python scripts/render_g1_personal_retarget.py --workers 3
```

smoke固定选择每名参与者每条件按clip ID排序的首条，共9条；全部通过才允许full处理其余18条。独立逻辑目录`logs/reproduction/g1_personal_retarget_v1/`指向持久盘时间戳目录。manifest冻结源视频/NPZ/CSV/元数据、参数、依赖与核心代码身份；身份变化拒绝原地恢复。恢复验证已有产物SHA，复用已完成候选，不重置尝试预算。

CPU数值处理与OSMesa回放均不占用训练GPU。视频包含原始GMR正面/侧面与实际执行正面/侧面四视图，50fps，完成后全视频解码核验；另提供分段关键帧和腰/骨盆/躯干yaw曲线。

检查回放后逐条记录具名复核结果，不能将pending自动当作pass：

```bash
uv run --no-sync python scripts/render_g1_personal_retarget.py \
  --action review --clip-ids P01_session01_left_shoulder_take001 \
  --reviewer <reviewer> --decision pass --review-note <evidence>
uv run --no-sync python scripts/run_g1_personal_retarget.py --stage export
```

处理全部27条，并且每个主体的stand/left/right至少各1条物理合格，全部接受轨迹回放复核通过，才发布训练子集。不满足则保留报告与回放，不降低门槛补数。

## 数据契约

- 固定`P01=train / P02=validation / P03=test`，全部派生版本随主体划分。
- `stand=0 / left_shoulder=1 / right_shoulder=2`；NULL=3只供未来条件丢弃，无阶段标签或阶段配额。
- 每条保存源50Hz轨迹、IK参考、实际执行NPZ、实际`(T,36)`CSV、qvel、力矩与JSON血缘。
- 导出实际执行的未经归一化`(N,10,59)`、stride5，保留clip、主体、帧区间、腰/骨盆/躯干角变化与clip-balanced权重。肩部起转前仅合法中立窗口为stand，起转后为肩部条件，跨边界剔除。
- 归一化只输出train-only诊断文件，不替换任何正式Prior统计；不能直接使用旧IK-v2加载器的硬编码身份和幅度配对假设。
- 人体视频是动作期望参考，不具有真实接触力标注。骨盆协同退让不等价于纯相对转腰；后续混合Prior须重新选配比、统一统计、校准并通过在线Score验收。
