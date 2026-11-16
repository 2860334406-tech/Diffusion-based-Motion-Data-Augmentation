# EXP-035：多来源肩部 Conditional Prior 与 PPO

本实验从零训练三条件 Prior，再从零执行 A→B1→B2。原数据、历史 Prior 和 EXP-033 的失败策略不覆盖、不 warm start。

## 数据与配额

数据来自 EXP-032 的 2400 条 IK v2、EXP-034 的 26 条合格个人二次重定向轨迹，以及 EXP-028 中重新通过当前完整碰撞 FK 检查的静止窗口。旧库不加入肩部、前进或后退正例；其 FK 检查不冒称动力学可执行性验证。

| 条件 | IK | 个人 | LAFAN | G1 Moves | AMASS-G1 | 每 epoch |
|---|---:|---:|---:|---:|---:|---:|
| stand |168|168|112|112|112|672|
| left_shoulder |672|672|0|0|0|1344|
| right_shoulder |672|672|0|0|0|1344|

IK 肩部先按11个5°幅度区间均衡；个人按 clip 均衡；旧库按来源、主体、原始文件及窗口抽样。左右镜像不是新的独立来源。保持原主体/参数族划分；个人 P01/P02/P03 分别属于 train/validation/test，禁止为扩大个人训练量移动主体。G1 Moves 没有留出静止样本，报告 N/A，不宣称其泛化验收已通过。

未归一化的50Hz、10帧、59维特征统一为最后一帧 heading；joint顺序和四元数约定必须一致。训练集按实际条件、来源、clip和幅度概率计算精确加权q01/q99。以分位数中心为中心设置最小完整跨度：位置0.02m，6D旋转0.05，关节0.10rad，线速度0.20m/s，角速度0.40rad/s。训练/采样/GSI/评分共用统计，不裁剪输入或抬高奖励。统计身份随checkpoint保存。

## Prior 与评分

schema 为 stand=0、left=1、right=2、NULL=3；NULL不是静止。随机初始化128维、4 heads、2 layers；DDPM50 steps，10 noise samples，batch1024；AdamW 3e-4、wd1e-4、clip1；10%条件丢弃，无EMA，guidance1。正式预算10000 epochs，独立1epoch smoke不进入正式权重；每100epoch恢复点、每1000epoch候选，模型仅保存本地。

维持V3公式、waist yaw第21维10倍权重、t=(8,15,22)，正确/反向条件共用噪声。每候选使用混合训练集4096样本/条件重新冻结校准。验证条件总体按来源和clip加权：正确>静止/反向及正确条件>反向条件各≥95%，median margin≥0.5，正确reward p05≥0.05；静止左右偏好35%–65%，mean|margin|≤0.2。幅度分区只作诊断。

通过候选按0.2/0.4/0.4加权验证损失选模。锁模后一次test，64配对噪声；4096生成窗口/条件，有限、静止速度p95≤0.15m/s、安全GSI接受率≥50%、左右方向率≥80%，评分门槛保持。独立样本按原始clip报告。

发布前在实际训练环境执行参考站立及64env、2iteration scratch PPO检查；各≥1024个支持条件下的非终止窗口，SMP中位数≥0.01、精确零分率≤1%，数值、更新及checkpoint有限。失败则停止，不重复盲测选模，不自动放宽门槛或扩预算。旧EXP-033的零奖励可作为负向回归检查，其权重绝不进入本实验。

## PPO 与验收

独立任务 `Smp-Force-Unloading-Conditional-ShoulderMixedV1-G1`，Actor/Critic输入128/992，全部学习状态随机初始化。4096env、seed42、std0.10、初始lr1e-4、自适应调度、每1000轮保存。相同新Prior用于GSI和Score，GSI上限0.15m/s；静止奖励exp(-10(vx²+vy²))，仍乘以原SMP，未加保底/显式转腰奖励/condition one-hot。

| 阶段 | 轮数 | 速度 | 力上限 | ramp | 静止权重 |
|---|---:|---:|---:|---:|---:|
| A |3000|无接触|—|—|0.9|
| A extension |首次失败追加2000|无接触|—|—|0.9|
| B1 |4000|0.01m/s|0.05mg|1s|0.1|
| B2 |4000|0.03m/s|0.10mg|0.5s|0.1|

肩部左右各50%，球半径0.10m、质量5kg、表面间隙0.02m、从正前向后推，12s单事件、等待1–3s。保持3帧接触确认和当前条件历史掩码；意外胸背接触清零。导轨行程仅遥测。

A评估1024episodes：速度p95≤0.30m/s、安全失败率≤5%、位移p50≤0.25m、超时完成率≥95%。失败并集包含自碰撞、关节限位以及全部非超时/非紧急力安全终止。扩展后仍失败停止。A通过后左右各128episodes接触smoke，正确接触覆盖≥80%才开始B1。

B1/B2各对stand/左/右评估1024episodes，左右要求正确接触覆盖≥80%、接触后方向正确≥80%、event success≥20%、紧急力率分别≤10%/20%。2秒响应须目标肩世界后移及扣除root平移后的后移均≥0.015m，躯干yaw符号对应侧别。骨盆协同转动允许，waist/pelvis单独报告，不套用旧骨盆上限和固定waist峰值。B1质量失败警告后继续B2，不训练C。

连续20个24步rollout的受支持窗口SMP精确零分比例≥99%即停止诊断；NaN/Inf、身份变化、GPU/进程故障停止并保留恢复状态。后续阶段恢复紧邻阶段完整学习状态，新的阶段重置仿真环境。

## 运行与产物

仅使用物理GPU0，UUID `GPU-eb0fc39f-6914-9e1d-65af-570524e8e061`，进程内cuda:0，EGL设备0；启动检查UUID和占用。产物及缓存保存在持久盘，通过 `logs/reproduction/g1_shoulder_mixed_prior_ppo_v1/` 访问。W&B仅配置、标量，模型不上传。

入口：`.venv/bin/python -m scripts.run_g1_shoulder_mixed_prior_ppo_suite`；明确恢复使用`--resume`，已有身份不一致拒绝复用。`--prepare-only`仅数据/存储预检，`--prior-only`执行完整Prior流程后结束。实验锁防止重复驱动。

通过全部Prior门槛后发布 `datasets/pretrain_ckpt/pretrained_g1_shoulder_mixed_conditional_v1.pt`。manifest保留SHA、父项、实际预算、W&B与恢复历史；最终stand/左右肩各600控制帧视频。本实验不保证混合数据必然解决奖励排序或PPO收敛，门槛失败按实记录。
