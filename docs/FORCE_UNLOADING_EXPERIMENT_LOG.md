# 导轨球形推头卸力任务实验记录

本文件为追加式实验日志。已有实验条目不回写；更正或复现实验使用新的实验 ID，并引用原条目。

## EXP-000：设计可行性与仓库接口核查

- 开始时间（UTC）：2026-11-07T16:21:30Z
- 结束时间（UTC）：2026-11-07T16:25:00Z
- 状态：完成
- 目的：确认设计文档能映射到现有 SMP/mjlab 架构，固定第一阶段可实现的机械、接触、观察、奖励与安全接口。
- 假设：mjlab 1.6 的一维 slide joint、mocap 固定基座和 `ContactSensor` 足以实现可受机器人反作用力影响的推头，无需修改仿真引擎。
- 主要方法：检查 `ManagerBasedRlEnv` 的 manager 执行顺序、G1 geom/body 命名、SMP feature buffer、`task_smp_product`、ContactSensor 的多接触槽及子步力历史；把已批准的固定参数映射为 `RailPusherCfg`。
- 配置：MuJoCo timestep 0.005 s，decimation 4；球半径 0.175 m，质量 5 kg，刚度 700 N/m，临界阻尼约 118.32 N·s/m，参考速度 0.05 m/s，最大驱动力 0.15 mg，最大行程 0.375 m。
- 命令：只读源码检查（`rg`、`sed`）以及 `pretrained_loco.pt` 元数据检查。
- 结果：确认 ContactSensor 可直接输出 `found/force/pos/normal` 和 `(B,N,4,3)` 子步力历史；确认 actor 基础维度 96、critic 基础 10 帧维度 960；确认 locomotion prior 输入为 `(10,59)`。
- 产物：[`FORCE_UNLOADING_RAIL_PUSHER_DESIGN.md`](FORCE_UNLOADING_RAIL_PUSHER_DESIGN.md)。
- 结论：基线可在仓库内以新任务和 class-based event term 实现，不需要 PhysX contact report 或跟踪未来推头轨迹。
- 下一步：实现任务，运行纯数学测试与无先验 CUDA 物理 smoke test。

## EXP-001：基线实现与 CUDA 物理 Smoke Test

- 开始时间（UTC）：2026-11-07T16:25:00Z
- 结束时间（UTC）：2026-11-07T16:39:59Z
- 状态：完成（机械/接触 smoke）；PPO smoke 另立条目
- 目的：验证导轨约束、三种推力方向、接触法向力采集、观察维度、数值稳定性与 locomotion prior 输入契约。
- 假设：零动作保持策略虽然不代表训练后行为，但足以验证推头动力学会与 G1 发生接触，并且机器人反力会显著减慢推头。
- 主要方法：注册 `Smp-Force-Unloading-G1`；物理 smoke 中移除扩散初始化和奖励，仅保留同一推头实体、控制器、传感器、观察、终止和指标；6 个 CUDA 环境运行 300 个控制步，三目标等概率采样。
- 配置：seed 42；A100-SXM4-80GB；6 env；300 steps；胸口/左肩/右肩概率均为 1/3；等待时间在 smoke 中设为 0 s；正式基线仍为胸口单方向、等待 1–3 s。
- 命令：
  - `uv run python -m unittest tests/test_force_unloading_math.py -v`
  - `MUJOCO_GL=egl uv run scripts/smoke_force_unloading.py --num-envs 6 --steps 300 --device cuda --seed 42`
- 结果：4/4 单元测试通过；actor shape `(6,128)`；critic shape `(6,992)`；SMP window/feature 为 `(10,59)`；采样覆盖目标 `[0,1,2]`；最大导轨横向误差 `1.862645149230957e-09 m`；最大接触力 `56.2855224609375 N`；接触期间最小推头速度约 `1.53519e-05 m/s`（低于 0.05 m/s 参考速度）；actor/critic 全部有限，无 NaN/Inf。
- 产物：`src/smp/rl/tasks/force_unloading/`、`scripts/smoke_force_unloading.py`、`tests/test_force_unloading_math.py`。
- 结论：机械与数据链路满足第一阶段 smoke 验收；机器人反作用力能把推头速度降到近零，导轨横向约束和无旋转结构成立。
- 下一步：加载 `pretrained_loco.pt`，运行 64 环境、2 次 PPO iteration 的 TensorBoard smoke，并记录检查点、奖励和损失是否有限。

## EXP-002：SMP + PPO 两次迭代 Smoke Test

- 开始时间（UTC）：2026-11-07T16:45:21Z
- 结束时间（UTC）：2026-11-07T16:47:14Z
- 状态：完成
- 目的：验证正式 locomotion prior、SMP 乘法奖励、PPO actor/critic、TensorBoard 日志和检查点保存链路。
- 假设：缩小 GSI pool 只影响 smoke 启动成本，不改变待验证的模型形状、奖励计算和 PPO 更新路径。
- 主要方法：使用 64 个并行环境和 seed 42，加载 `pretrained_loco.pt`；GSI pool/batch 临时缩为 64 并关闭 `torch.compile`；运行 2 个 PPO iteration，每轮 24 steps/env、保存一次模型；启用 mjlab NaN guard。首次运行发现 capture-point 指标在无足底接触时发生 `Inf×0` 日志 NaN，修复为 `torch.where` 后完整重跑。
- 配置：actor `(128→512→256→128→29)`；critic `(992→512→256→128→1)`；TensorBoard logger；`save_interval=1`；`max_iterations=2`。
- 命令：`MUJOCO_GL=egl uv run scripts/train.py Smp-Force-Unloading-G1 --env.scene.num-envs 64 --env.seed 42 --env.events.init-smp-state.params.gsi-buffer-size 64 --env.events.init-smp-state.params.gsi-batch-size 64 --env.events.init-smp-state.params.compile-model False --agent.max-iterations 2 --agent.save-interval 1 --agent.logger tensorboard --agent.experiment-name smp_force_unloading_g1 --agent.run-name smoke_seed42_fixed_metrics --log-root logs/reproduction/rsl_rl --enable-nan-guard True`
- 结果：退出码 0；总计 3072 transitions；iteration 1 value loss `0.0025`、surrogate loss `-0.0558`、SMP reward mean `0.1083`、variance `0.0274`；所有观测、训练损失和指标均为有限值。24-step rollout 仅覆盖 0.48 s，短于 1–3 s 等待窗口，因此该 smoke 未触发推头接触，`peak_force` 和 `event_success` 为 0，符合预期。
- 产物：`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-07_16-46-33_smoke_seed42_fixed_metrics/`，含 `model_0.pt`、`model_1.pt`、TensorBoard events、`params/env.yaml`、`params/agent.yaml` 和 git diff 快照。
- 结论：任务注册、先验加载、GSI、SMP reward、PPO 更新、NaN guard、日志和检查点链路均通过；两次迭代不用于判断策略质量或收敛。
- 下一步：正式训练前先决定是否过滤 GSI 中少量初始关节越界/自碰撞样本；随后按正式规模训练并使用独立长回放评估卸力成功率。

## EXP-003：GSI 初始安全状态诊断

- 开始时间（UTC）：2026-11-07T16:46:20Z
- 结束时间（UTC）：2026-11-07T16:46:32Z
- 状态：完成（发现正式训练前待处理项）
- 目的：量化 PPO smoke 中 `joint_hard_limit` 和 `self_collision` 的早期重置来源。
- 假设：早期重置来自 locomotion prior 的少量 GSI 样本，而不是导轨推头或 PPO 数值故障。
- 主要方法：64 环境、seed 42、GSI pool/batch 64；加载先验并仅执行 reset，比较每个关节与 hard limits，同时读取已有 `self_collision` sensor。
- 配置：与 EXP-002 相同的 GSI smoke 配置，不执行 PPO step。
- 命令：使用 `g1_force_unloading_smp_env_cfg()` 构造 64-env CUDA 环境，reset 后统计 joint-limit excess 与 self-collision 数量。
- 结果：3/64 个环境存在大于 0 的关节越界，最大越界 `0.0171166062 rad`；其中 1/64 超过 `0.01 rad`；4/64 个环境初始 self-collision。未发现 NaN/Inf。
- 产物：诊断输出记录在本条目，无额外模型产物。
- 结论：问题来自 GSI 初始样本与硬安全终止条件的边界不一致；机械推头 smoke 和 PPO 数值链路仍成立，但正式训练会因无效初始样本损失采样效率。
- 下一步：优先在 GSI pool 生成/抽样阶段做安全拒绝采样，并保持 hard-limit/self-collision 终止严格；不建议仅为降低重置率放宽安全阈值。

## EXP-004：GSI 安全拒绝采样与刷新状态保护

- 开始时间（UTC）：2026-11-07T17:02:00Z
- 结束时间（UTC）：2026-11-07T17:14:26Z
- 状态：完成
- 目的：消除 EXP-003 发现的 GSI 初始关节越界和自碰撞样本，并保证周期 GSI pool 刷新不会扰动正在训练的环境。
- 假设：使用物理关节限位做第一层张量过滤，再通过不推进时间的 MuJoCo `forward` 检查 `self_collision` sensor，可以低成本筛出安全窗口；筛选前后保存并恢复 `qpos/qvel` 与 SMP 历史即可使周期刷新对活动 episode 保持状态中性。
- 主要方法：在通用 GSI 生成器中加入可选拒绝采样；过滤 NaN/Inf、最后一帧关节位置超限和自碰撞；按最大采样倍率限制失败重试并输出接受率；初始化与 `gsi_refresh` 共用同一筛选路径。仅在 `Smp-Force-Unloading-G1` 启用，关节限位内缩 `1e-4 rad`，使用既有 `self_collision` sensor；碰撞筛选临时复用仿真 world，并在 `finally` 中恢复完整 `qpos/qvel` 和七个 SMP buffer 张量。
- 配置：A100-SXM4-80GB；seed 42；首次验证 64 env、GSI pool/batch 64；刷新验证 32 env、GSI pool/batch 32；关闭 `torch.compile` 以缩短诊断启动时间。
- 命令：`uv run ruff check src/smp/rl/events.py src/smp/rl/tasks/force_unloading/force_unloading_env_cfg.py tests/test_gsi_safety.py`；`uv run python -m unittest tests/test_gsi_safety.py tests/test_force_unloading_math.py -v`；另使用 `ManagerBasedRlEnv` 运行 64-env 初始池 reset 诊断和 32-env 周期刷新状态保持诊断。
- 结果：Ruff 通过；7/7 单元测试通过。初始池从 128 个候选中保留 64 个，候选接受率 `92.19%`，拒绝关节超限 6 个、自碰撞 4 个、非有限值 0 个；4096 次 reset 中关节越界 0、自碰撞 0、最大关节越界量 `0 rad`。周期刷新从 64 个候选中保留 32 个，候选接受率 `89.06%`，拒绝关节超限 3 个、自碰撞 4 个；刷新前后 `qpos`、`qvel` 和全部 SMP 历史缓冲区逐位一致；刷新后的池在 2048 次 reset 中关节越界 0、自碰撞 0。
- 产物：`src/smp/rl/events.py`、`src/smp/rl/tasks/force_unloading/force_unloading_env_cfg.py`、`tests/test_gsi_safety.py`。
- 结论：EXP-003 的 GSI 初始安全问题已解决；安全筛选同时覆盖首次建池和周期刷新，未通过候选不会进入卸力任务的 GSI pool，周期筛选不会改写活动环境的物理状态或 SMP 历史。
- 下一步：运行 64 env、每环境至少 200 rollout steps 的接触覆盖 PPO smoke test，再决定是否启动胸前单方向正式训练。


## EXP-005：胸前接触覆盖与首次正式运行诊断

- 开始时间（UTC）：2026-11-07T17:20:45Z
- 结束时间（UTC）：2026-11-07T17:37:40Z
- 状态：完成（诊断运行主动停止）
- 目的：验证 12 s episode 中胸前推头能够进入接触阶段，并判断从随机扩展 actor 直接开始 30000 轮训练是否具备有效学习信号。
- 主要方法：先以 4096 环境、seed 42、30,000 iteration 和每 1,000 iteration 保存启动 W&B 正式运行；观察早期 episode 后，使用 512 环境、300 iteration 的接触覆盖 smoke 分离检查 GSI locomotion、推头事件和终止原因。
- 配置：固定胸口目标；等待 1–3 s；球半径 0.175 m；参考速度 0.05 m/s；紧急力阈值 0.30 mg；初始 actor 为新增 32 维接触历史后的随机网络。
- 结果：首次正式运行在 iteration 38 主动停止；平均 episode 约 0.22 s，主要由原始 GSI 中不安全/高速状态及随机 actor 导致，无法覆盖推头事件。接触 smoke 的 episode 长度提高到约 11–15 steps，但仍未形成可用于卸力学习的稳定接触。W&B 诊断 run：`yhwnl2nu`。
- 产物：`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-07_17-20-45_stage1_chest_seed42_30k/` 及后续 contact-coverage smoke 日志。
- 结论：不能从随机扩展策略直接进行长训练；需要保留已训练 Forward actor 的本体感觉映射，并进一步限制 GSI 初始移动速度。
- 下一步：构造与 Forward 策略函数等价的 128 维 warm start，并验证低速 GSI 拒绝采样。

## EXP-006：Forward Warm Start 与低速 GSI 基线选择

- 开始时间（UTC）：2026-11-07T17:51:00Z
- 结束时间（UTC）：2026-11-07T18:22:17Z
- 状态：完成
- 目的：在不完整重训扩散先验的前提下，为胸前推力 PPO 提供可持续 rollout 的 locomotion 初始化，并降低随机 GSI 高速状态造成的早期失稳。
- 主要方法：从 Forward `model_23000.pt` 映射 actor 的 96 维本体感觉权重到新 128 维 actor，新增 32 个接触输入列置零；把速度命令对第一层的影响折叠进 bias；critic 重新初始化。比较 0.50、0.25 和 0.00 m/s 三种命令折叠；固定动作标准差 0.10、优化器学习率 `1e-4`。GSI 生成增加最后一帧 root 平面速度不超过 1.2 m/s 的过滤，并保留 joint-limit/self-collision 过滤；紧急力回合的 task reward 置零，避免策略利用撞击后的瞬时力下降。
- 配置：4096 env；seed 42；GSI pool/batch 1024；刷新 256；最大采样倍率 20；每个候选窗口尺寸 `(10,59)`。
- 结果：全规模 GSI 从 5,120 个候选保留 1,024 个，安全候选接受率 23.77%；拒绝 joint limit 287、root speed 3,393、self collision 223、nonfinite 0。三种 warm start 都消除了主要自碰撞并保持数值有限，但固定胸前障碍与 locomotion 的相对接近速度仍使验证峰值力约为 3.0 mg，紧急力失败率接近 100%；因此该问题作为 PPO 第一阶段需要学习的初始失败模式保留，不修改接触后的固定导轨力学。
- 产物：`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-07_18-20-00_forward23000_speed000_std010_lowspeed_gsi_lr1e4_warmstart/model_0.pt`；三个 full-scale validation 目录；新增 GSI 单元测试。
- 结论：选用 0.00 m/s 命令折叠、std 0.10、lr `1e-4` 的 warm start 作为正式运行起点；已知初始策略尚不能卸力，正式训练重点观察紧急力率是否下降。
- 下一步：以 W&B 在线标量日志启动 30,000 iteration，并每 1,000 iteration 保存本地策略。

## EXP-007：第一阶段胸前推力 30,000 轮正式训练

- 开始时间（UTC）：2026-11-07T18:22:17Z
- 结束时间（UTC）：2026-11-08T03:00:30Z
- 状态：停止，失败基线（产物保留）
- 目的：仅训练第一阶段胸前单方向卸力策略，评估 PPO 能否从高冲击初始策略逐步学习降低首次接触速度、峰值力和紧急终止比例。
- 主要方法：从 EXP-006 的零速命令 Forward warm start 恢复 actor/critic/normalizer/optimizer；使用 4096 个并行环境、24 steps/env、seed 42 和 PPO；通过 W&B 记录训练指标；每 1,000 iteration 保存一次本地检查点。自动模型上传关闭，避免未经单独确认把策略文件上传到远端。
- 配置：`max_iterations=30000`；`save_interval=1000`；动作 std 0.10；初始学习率 `1e-4`、adaptive KL schedule；胸口目标概率 1.0；A100-SXM4-80GB；`MUJOCO_GL=egl`；NaN guard 开启。
- 命令：`MUJOCO_GL=egl uv run scripts/train.py Smp-Force-Unloading-G1 --env.scene.num-envs 4096 --env.seed 42 --agent.resume True --agent.load-run 2026-11-07_18-20-00_forward23000_speed000_std010_lowspeed_gsi_lr1e4_warmstart --agent.load-checkpoint model_0.pt --agent.max-iterations 30000 --agent.save-interval 1000 --agent.logger wandb --agent.wandb-project smp --agent.experiment-name smp_force_unloading_g1 --agent.run-name stage1_chest_seed42_30k_lowspeed_warmstart --agent.actor.distribution-cfg.init-std 0.10 --agent.algorithm.learning-rate 0.0001 --agent.upload-model False --log-root logs/reproduction/rsl_rl --enable-nan-guard True`
- 结果：iteration 20 时峰值力约 3.02 mg、紧急力失败率约 99.86%、event success 0；最终有效日志点 iteration 19,485，最新完整千轮 checkpoint 为 `model_19000.pt`。停止前 100 轮均值为紧急力失败率 `98.9094%`、峰值力 `3.4030 mg`、event success `0%`、SMP reward mean `0.014771`、mean reward `0.015009`、episode length `104.96` steps；NaN 失败率 0。
- W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/4ikmyq96>
- 本地产物：`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-07_18-22-17_stage1_chest_seed42_30k_lowspeed_warmstart/`；配置快照确认 30,000 iteration、每 1,000 iteration 保存、模型不上载。
- 验收：未通过。iteration 15,000 replay 与最终标量均表明策略持续向前行走撞击推头，event success 为 0；本 run 已在最新完整千轮 checkpoint 后正常停止，并作为 locomotion prior 失败对照封存。


## EXP-008：iteration 15,000 中期评估与无头 Replay

- 开始时间（UTC）：2026-11-08T01:13:00Z
- 结束时间（UTC）：2026-11-08T01:21:12Z
- 状态：完成；EXP-007 正式训练继续运行
- 目的：检查 30,000 轮正式训练的中期收敛情况，并通过实际 replay 判断奖励上升是否对应胸前卸力能力。
- 主要方法：解析完整 TensorBoard 标量，对每个 1,000 iteration 附近计算最近 100 轮均值；选取最新完整阶段检查点 `model_15000.pt`，使用 seed 42、1 个环境、确定性推理和官方 `VideoRecorder`，通过 EGL 无头执行 600 个控制步，录制 1280×720、50 FPS、12 s 视频；按橙色推头像素定位接触帧并抽取连续帧检查终止行为。
- 中期曲线：iteration 14,900–14,999 的 100 轮均值为紧急力失败率 98.58%、峰值力 3.37 mg、event success 0、平均 episode 104.34 steps、平均奖励 0.01481、SMP reward mean 0.01464；全程 NaN/Inf 终止为 0。iteration 15,725 的即时值仍为紧急力失败率 98.69%、峰值力 3.39 mg、event success 0。
- Replay 结果：机器人持续向前行走，未观察到主动后撤、转腰或屈膝卸力。第一次推头在 frame 92 出现，frame 92–97 与胸前发生快速重叠，frame 98 因紧急力终止后消失并重置；约 frame 280 和 511 重复相同模式。12 s 视频共出现 3 个短暂推头接触片段，没有成功卸力事件。
- 产物：`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-07_18-22-17_stage1_chest_seed42_30k_lowspeed_warmstart/videos/headless_model_15000/rl-video-step-0.mp4`；同目录下 `contact_sheet.jpg`、`contact_event_sheet.jpg` 和 `first_impact_sequence.jpg`。
- 结论：当前奖励上升主要来自 SMP motion-prior reward 提高，不代表卸力任务收敛；策略仍以行走姿态高速撞击固定推头，峰值力较训练初期没有改善，第一阶段在 iteration 15,000 尚未达到验收要求。
- 下一步：保留正式训练至 30,000 iteration，并在最终检查点重复定量评估与 replay；若最终仍无 event success，应调整接合课程或任务奖励后重新训练，而不是把本 run 作为成功策略。


## EXP-009：LAFAN1 G1 站立与低速片段筛选

- 开始时间（UTC）：2026-11-08T01:52:00Z
- 结束时间（UTC）：2026-11-08T02:24:05Z
- 状态：完成（运动学筛选；候选尚未进行视觉和动力学验收）
- 目的：判断 LAFAN1 G1 retargeting 数据中是否存在可用于胸前卸力 prior 的安静站立、原地踏步和低速重心调整数据。
- 数据范围：从仓库 README 指定的 Hugging Face 数据集 `lvhaidong/LAFAN1_Retargeting_Dataset` 临时下载完整 `g1/` 目录；共 40 个 CSV、264,705 帧、30 FPS、约 147.06 min。文件级类别只有 dance、fallAndGetUp、fight、fightAndSports、jumps、run、sprint 和 walk，没有显式 stand、idle、step-in-place 或 balance 文件。临时数据位于 `/tmp/smp-lafan-g1-analysis.kvGP1r/`，未写入正式 `datasets/`。
- 主要方法：使用 90 帧（3 s）窗口、15 帧步长，计算 root 平面平均速度、净位移、轨迹半径、root 高度及变化、双腿关节速度 RMS、膝关节活动幅度和 root yaw 变化。站立候选要求直立、3 s 小位移和低腿部活动；原地动态候选要求直立、净位移小于 0.08–0.10 m、双腿持续活动，并限制 yaw 变化以排除明显转圈。
- 站立结果：发现 121 个重叠的 3 s 安静站立窗口，合并为 23 段。强候选包括 `dance2_subject4.csv` 0.50–7.00 s、`dance2_subject2.csv` 0.50–6.00 s、`dance2_subject3.csv` 0.50–7.50 s、`dance2_subject5.csv` 0.50–4.00 s、`dance1_subject1.csv` 0.50–3.50 s；`sprint1_subject4.csv` 中还存在 55–65 s、93.5–100 s、135.5–147.5 s、173.5–181.5 s 等动作间静止段。最静止窗口的平均 root 平面速度约 0.0023 m/s。
- 原地踏步/重心调整结果：加入 yaw 过滤后得到 38 个 3 s 候选窗口。最适合优先回放的片段包括：`walk4_subject1.csv` 1.00–4.00 s（净位移 0.014 m、平均速度 0.038 m/s、yaw 约 8.1°）；`fightAndSports1_subject4.csv` 74.50–77.50 s（0.043 m、0.078 m/s、yaw 约 2.2°）；`dance2_subject2.csv` 35.50–38.50 s（0.031 m、0.072 m/s）；`sprint1_subject4.csv` 55.00–58.00 s（0.037 m、0.046 m/s）；`jumps1_subject5.csv` 214.50–217.50 s（0.009 m、0.046 m/s）；`walk3_subject2.csv` 58.00–61.00 s（0.029 m、0.068 m/s、yaw 约 2.1°）。
- 现有本地数据结论：`/root/gentle-humanoid-training/dataset/lafan` 仅由 allowlist 中的 walk1/2/3 片段构成，不是完整 LAFAN；SMP 仓库的 `pretrained_lafan_run.pt` 按 README 也是 LAFAN run subset。因此上述新发现的站立/低速片段不应假定已包含在现有 prior 中。
- 限制：LAFAN 文件只有粗粒度动作标签，数值候选可能是舞蹈过渡、起跑准备、落地缓冲或动作间 reset pose；该 retargeting 数据只保证运动学约束，不保证 G1 动力学可执行。必须先视觉回放，再通过 MuJoCo 关节限位、自碰撞、足底接触和无控制跟踪 smoke 验收。
- 结论：完整 LAFAN G1 数据内确实存在足够长的安静站立片段，也存在一批低位移、双腿活动的原地踏步/重心调整候选；可用于构建新的低速 prior 数据集，但不能直接使用完整动作文件，需按时间段裁剪和筛选。
- 下一步：优先回放上述候选，确定语义；将通过验收的窗口裁剪到独立 CSV，补充镜像和适量重复采样，再使用仓库现有 `csv_to_npz.py` 转换并进行低速 diffusion prior smoke test。

## EXP-010：停止 Locomotion Prior 失败基线

- 开始时间（UTC）：2026-11-08T02:59:57Z
- 结束时间（UTC）：2026-11-08T03:00:30Z
- 状态：停止，失败基线（产物保留）
- 目的：在低速 prior 重训前，于最新完整千轮检查点停止 EXP-007，避免继续消耗预算，同时完整保留可比较的失败对照。
- 主要方法：正常中断训练，等待 W&B/TensorBoard 刷新；保留全部 `model_*000.pt`、既有 replay 和 W&B run；解析停止前 100 个有效 iteration 的标量均值。
- 结果：最终有效日志点 iteration 19,485，最新完整 checkpoint 为 `model_19000.pt`。停止前 100 轮均值：紧急力失败率 `98.9094%`、峰值力 `3.4030 mg`、event success `0%`、SMP reward mean `0.014771`、mean reward `0.015009`、episode length `104.96` steps；NaN 失败率 0。replay 已确认策略持续向前走并撞击推头。
- W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/4ikmyq96>
- 产物：`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-07_18-22-17_stage1_chest_seed42_30k_lowspeed_warmstart/model_19000.pt` 及同目录全部旧产物。
- 结论：EXP-007 作为 locomotion prior 失败基线封存，不进入更强推力阶段。

## EXP-011：LAFAN 低速平衡数据集视觉验收与转换

- 开始时间（UTC）：2026-11-08T03:00:30Z
- 结束时间（UTC）：2026-11-08T03:20:15Z
- 状态：完成
- 目的：构建站立与低位移动态各占一半、可复现且符合 README 契约的低速 prior 数据集。
- 主要方法：复制 40 个 G1 CSV；按半开区间裁剪严格 90 帧；拒绝 T-pose、明显转身和不合格姿态；按 G1 左右关节约定生成矢状面镜像，并用 Forward 原始/镜像对逐值验证；生成无头视频和审查指标。
- 配置：6 个站立、6 个动态原片段，每段 3 s、30 FPS；各增加镜像。站立与动态均为 18 s 原始数据、36 s 镜像后数据；转换为 50 FPS、窗口 `(10,59)`。
- 命令：`uv run scripts/curate_lafan_low_speed.py ...`；`MUJOCO_GL=egl uv run scripts/review_lafan_low_speed.py ...`；`MUJOCO_GL=egl uv run scripts/csv_to_npz.py --input-dir datasets/csv/lafan_low_speed_balance --output-dir datasets/npz/lafan_low_speed_balance`。
- 结果：24/24 CSV 为 `(90,36)`；24/24 NPZ 为 `(140,10,59)`，共 3,360 windows，站立/动态各 1,680；全部有限、无 joint-limit violation。13 个相关测试通过。完整 LAFAN `norm_stats.npz` 未覆盖，SHA-256 为 `c5aac10f152f8d2b6f65ecf4bd1ddd027a25c89a6c2fa72bd599e93ef0064628`。
- 产物：`docs/LAFAN_LOW_SPEED_BALANCE_MANIFEST.csv`、`datasets/csv/lafan_low_speed_balance/`、`datasets/npz/lafan_low_speed_balance/`、`logs/reproduction/data/lafan_low_speed_review_final/`。
- 结论：类别与窗口严格 1:1，满足正式训练条件。

## EXP-012：LAFAN 低速 Prior 从零训练与验收

- 开始时间（UTC）：2026-11-08T03:20:15Z
- 结束时间（UTC）：2026-11-08T03:40:10Z
- 状态：完成，通过全部门槛
- 目的：从零训练 128 维、2 层、无 EMA 的 10,000 epoch 低速 DDPM prior，并验证速度分布和 GSI 接受率。
- 主要方法：先做 1 epoch 无 W&B smoke；随后 seed 42、train split 1.0 正式训练；W&B 仅同步标量/配置，`upload_model=False`；固化最终权重，并对候选/locomotion prior 各采样 4,096 窗口和构建 1,024 状态 GSI 池。
- 配置：`d_model=128`、2 layers、10 noise samples、50 timesteps、batch 1024、lr `3e-4`、10,000 epoch、保存间隔 5,000。
- 命令：`uv run scripts/pretrain.py --data-dir datasets/npz/lafan_low_speed_balance --num-layers 2 --no-use-ema --save-interval 5000 --num-epochs 10000 --train-split 1.0 --d-model 128 --seed 42 --name lafan_low_speed_balance_seed42 --log-dir logs/reproduction/pretrain`；`MUJOCO_GL=egl uv run scripts/validate_low_speed_prior.py`。
- 结果：smoke loss `0.911987`，正式最终 loss `0.041301`。候选速度 mean/p95/max 为 `0.02740/0.07531/0.18110 m/s`；locomotion p95 为 `2.99238 m/s`，降低 `97.4834%`；全部有限。GSI 安全候选 1,189/1,792，接受率 `66.3504%`，保留 1,024。稳定 checkpoint SHA-256：`5574dc5720a356937f6a9bc01b77f3d5e8456970d59e59c59b05e2ca6a92d0b7`。
- W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/mrdp496o>（0 media、0 artifact）。
- 产物：`logs/reproduction/pretrain/lafan_low_speed_balance_seed42/20261108_032204/pretrained.pt`、`datasets/pretrain_ckpt/pretrained_lafan_low_speed_balance.pt`、`logs/reproduction/data/low_speed_prior_validation.json`。
- 结论：全部 prior 门槛通过，可以进入 PPO 课程训练。

## EXP-013：低速 Prior 胸推 PPO Smoke Test

- 开始时间（UTC）：2026-11-08T03:41:48Z
- 结束时间（UTC）：2026-11-08T03:41:54Z
- 状态：完成
- 目的：验证新 prior、GSI、安全重置、128/992 维 warm start、PPO 更新、指标和 checkpoint 链路。
- 主要方法：64 环境、2 iteration、TensorBoard；从 EXP-006 `model_0.pt` 恢复；GSI pool/batch 64、关闭 prior compile、启用 NaN guard。
- 结果：GSI 接受率 `64.84%`；actor/critic 输入 `128/992`；两次 PPO 更新及 `model_0.pt`、`model_1.pt` 保存成功。iteration 1 value loss `0.0001`、surrogate loss `-0.0315`、std `0.10`；全部有限，无 NaN。
- 产物：`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-08_03-41-48_low_speed_prior_smoke_seed42/`。
- 结论：端到端链路通过；两轮 smoke 不用于策略质量判断。

## EXP-014：课程阶段 A 站稳训练、扩展与门槛停止

- 开始时间（UTC）：2026-11-08T03:42:42Z
- 结束时间（UTC）：2026-11-08T06:10:59Z
- 状态：停止，阶段 A 门槛失败；未进入 B1/B2/C
- 目的：在完全无推头接触的条件下，使用低速 prior 将 EXP-006 warm start 调整为稳定、低速且低位移的站立策略；通过阶段门槛后才允许接触课程。
- 主要方法：使用 4,096 个环境、seed 42、动作标准差 0.10、学习率 `1e-4`，从 EXP-006 `model_0.pt` 恢复完整 actor、critic、normalizer 和 optimizer；显式加载 `pretrained_lafan_low_speed_balance.pt`，把等待时间设为 `(1e9,1e9)` 禁用接触。先训练 3,000 iteration，再以 `model_2999.pt` 恢复完整状态执行计划允许的最大 2,000 iteration 扩展。两个 run 均每 1,000 iteration 保存本地 checkpoint，W&B 仅记录标量和配置，`upload_model=False`。
- 配置：GSI 最大平面速度 `0.5 m/s`；12 s episode；NaN guard 开启；阶段 A 门槛为事件前 root 平面速度 p95 ≤`0.30 m/s`、非足部触地/跌倒合计 ≤`5%`、12 s 位移中位数 ≤`0.25 m`。
- 首轮结果：最终 checkpoint `model_2999.pt`。独立确定性评估 1,024 episodes：速度 mean/p95/max 为 `0.1894/0.4424/1.1600 m/s`，位移 p50 为 `0.02821 m`，非力学失败率 `0%`，joint-limit failure `1.3672%`，NaN 0。速度 p95 超限，首次门槛失败。
- 扩展结果：最终 checkpoint `model_4998.pt`。独立确定性评估 1,024 episodes：速度 mean/p95/max 为 `0.2115/0.4951/1.6800 m/s`，位移 p50 为 `0.02991 m`，非力学失败率 `0%`，joint-limit failure `0.0977%`，NaN 0；99.1211% episode 达到 timeout。位移与非力学失败门槛通过，但速度 p95 仍超限且比首次评估更高，因此扩展后门槛仍失败。
- W&B：首轮 <https://wandb.ai/qc13659041465-ustc/smp/runs/5r79qrb6>；扩展 <https://wandb.ai/qc13659041465-ustc/smp/runs/e1mxnb7d>。两个 run 的本地文件清单仅含配置、标量 summary、metadata 和文本输出，无模型、视频或 artifact 上传。
- 产物：`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-08_03-42-42_curriculum_a_stand_low_speed_prior_seed42/model_2999.pt`；`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-08_05-11-31_curriculum_a_extension_low_speed_prior_seed42/model_4998.pt`；`logs/reproduction/evaluation/curriculum_a_model_2999.json`；`logs/reproduction/evaluation/curriculum_a_extension_model_4998.json`。
- 结论：低速 prior 明显降低了典型位移并消除了持续向前行走，但策略速度分布仍有长尾，未满足进入接触课程的安全前提。按照预定规则，在用尽 2,000 轮扩展额度后停止整个 PPO 课程；不启动轻柔接触、中等接触或标称胸推，也不录制受力 replay。
- 下一步：若另行批准新实验，应先修改阶段 A 的目标或奖励以直接惩罚 root 速度长尾，并重新设立独立预算；不得从本次失败 checkpoint 直接进入更强推力阶段。



## EXP-015：LAFAN 低速 Prior 从零策略胸前推力课程

- 开始时间（UTC）：2026-11-08T17:11:20Z
- 结束时间（UTC）：进行中
- 状态：阶段 A 运行中
- 目的：不加载任何 Forward 行走策略，从随机初始化 Actor、Critic、观测归一化器和优化器开始，使用低速平衡 prior 完成 A→B1→B2→C 胸前推力课程。
- 主要方法：增加可配置 stationary reward 权重和可恢复课程驱动器；阶段 A 使用 force-drop/stationary=`0.1/0.9` 且禁用接触，训练 3,000 轮，门槛失败时追加 2,000 轮；B1/B2/C 分别使用保守递进力学参数训练 4,000/4,000/19,000 轮。各阶段恢复紧邻前一阶段完整 PPO 状态，每 1,000 轮保存；门槛失败记录告警但继续，NaN/Inf、checkpoint 损坏或进程失败才硬停止。
- Prior：`datasets/pretrain_ckpt/pretrained_lafan_low_speed_balance.pt`，SHA-256 `5574dc5720a356937f6a9bc01b77f3d5e8456970d59e59c59b05e2ca6a92d0b7`；GSI 最大 root 平面速度 `0.5 m/s`。
- 验证：Ruff 通过；奖励、课程和 GSI 共 15/15 单元测试通过。64 环境、2 iteration 的 scratch smoke 使用 `resume=false`，Actor/Critic 输入为 `128/992`，GSI 接受率 `64.84%`，两轮 PPO 和 checkpoint 全部有限、无 NaN/Inf；8 回合评估及 10 帧 EGL 视频链路通过。
- 正式配置：A100-SXM4-80GB；4,096 env；seed 42；std 0.10；初始 lr `1e-4`；W&B 标量在线记录，`upload_model=false`；阶段 A run 明确不含 `load-run` 或 `load-checkpoint`。
- 阶段 A W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/5w9r1mj0>
- 产物：`logs/reproduction/curriculum/lafan_low_speed_scratch_seed42/curriculum_state.json`；`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-08_17-12-28_lafan_scratch_curriculum_scratch_smoke_seed42/`；`logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-08_17-15-24_lafan_scratch_curriculum_a_seed42/`。
- 当前结果：阶段 A iteration 1 已开始，无 NaN/Inf；初期随机策略自碰撞和关节限位重置较多，属于从零训练起始分布，需以 3,000 轮后独立 1,024 episode 评估为阶段结论。
- 后续：驱动器将在 A 结束后自动评估并决定是否追加 2,000 轮，然后继续 B1、B2、C；最终 C 评估同时录制 600 控制帧无头视频。

## EXP-016：移除导轨行程失败终止

- 开始时间（UTC）：2026-11-09T03:38:47Z
- 结束时间（UTC）：2026-11-09T03:38:47Z
- 状态：实现完成，待新进程生效
- 目的：允许胸前或背后受力时通过持续、受控后移卸力，不再把推头到达 `0.375 m` 物理行程上限直接判为 episode 失败。
- 主要方法：从任务终止项中删除 `pusher_travel`；保留 slide joint 的物理限位；将 `failure_travel` 失败指标改为中性的 `travel_limit_reached` 遥测；同步单元测试和设计文档。
- 影响范围：到达行程上限不再提前重置，也不再因普通终止而损失剩余回报；原有低力 `event_success` 判据暂不改变。修改只对新建环境生效，正在运行且已加载旧配置的 EXP-015 C 进程仍使用旧定义。

## EXP-017：人工停止 C 并按新行程口径统一评估

- 开始时间（UTC）：2026-11-09T03:45:00Z
- 结束时间（UTC）：2026-11-09T03:54:16Z
- 状态：完成；C 训练按用户要求停止，四阶段复评完成
- 目的：终止尚未完成的 C 阶段训练，并在“导轨到限只记录、不终止”的统一环境定义下评价 A 扩展、B1、B2 和 C 最新策略。
- 停止点：C 最后日志 iteration `21195/31996`，约完成本阶段 `43.1%`；最后完整 checkpoint 为 `model_21000.pt`。先发送 SIGINT，训练完成当前调用并进入 W&B 收尾但未及时退出，随后 SIGTERM 结束子进程；所有训练与课程驱动进程均已退出，已有 checkpoint 和 W&B run `rm9nurmi` 保留。
- 评估方法：各阶段使用 256 env、seed 42、确定性策略独立运行 1,024 episodes；A 使用 `model_4998.pt`，B1 使用 `model_8997.pt`，B2 使用 `model_12996.pt`，C 使用 `model_21000.pt`；全部使用 LAFAN 低速 prior 和对应阶段力学参数。
- A 扩展：接触前速度 p95 `0.5052 m/s`，位移 p50 `0.0310 m`，非力学失败 `0.1953%`，关节失败 `0.9766%`，自碰撞终止 `4.8828%`；仅速度门槛失败，说明位置保持尚可但速度长尾未解决。
- B1：严格低力成功率 `0%`，紧急力失败 `0%`，峰值力 p95 `1.1464 mg`，自碰撞终止 `25.9766%`；未学会卸力，且短时冲击与自碰撞尾部显著。
- B2：严格低力成功率 `0.3906%`，紧急力失败 `0%`，峰值力 p95 `0.2047 mg`，关节失败 `6.0547%`；相较 B1 接触力尾部和自碰撞改善，但低力成功仍基本没有。
- C：严格低力成功率 `30.4688%`，导轨到限率 `83.7891%`，紧急力/非力学/关节/NaN 失败均 `0%`，工作区失败 `0.0977%`；峰值力 mean/p95/max 为 `0.1282/0.1721/0.2777 mg`，纵向位移 mean/p95 为 `0.2910/0.3406 m`，足滑 p95 `1.9926 m/s`。严格 C 门槛因成功率低于 `50%` 未通过，但按胸推“安全后退到限也成功”的语义，至少 `83.7%` episode 可视为有效退让。
- 训练/评估差异：最后 100 轮随机训练 rollout 的 B2/C `event_success` 分别为 `60.31%/71.24%`，确定性评估仅 `0.39%/30.47%`，说明动作采样噪声对成功有显著贡献，Actor 均值策略鲁棒性不足。
- 结论：A→B1→B2→C 课程确实在 C 阶段学出了安全后退卸力，但没有稳定达到原始低力完成标准；B1/B2 的正式质量门槛失败，C 的新语义表现较好但仍存在足滑、接触前速度长尾和确定性退化。
- 产物：`logs/reproduction/evaluation/no_travel_termination_a_extension_model4998.json`、`no_travel_termination_b1_model8997.json`、`no_travel_termination_b2_model12996.json`、`no_travel_termination_c_model21000.json`；C W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/rm9nurmi>。


## EXP-018：GSI 0.15 与陡峭静止奖励胸推课程（50/50 Prior）

- 开始时间（UTC）：2026-11-09T08:14:00Z
- 结束时间（UTC）：进行中
- 状态：A 与条件扩展完成（速度门槛均失败）；B1 在输出断管中断后已从最新完整检查点恢复运行
- 目的：针对阶段 A 初始速度和速度长尾过大，将 GSI root 平面速度上限由 `0.5` 降至 `0.15 m/s`，并将静止奖励改为 `exp(-10·(vx²+vy²))`，隔离检验初始化过滤和奖励曲率的效果。
- 主要方法：保留现有 50/50 `pretrained_lafan_low_speed_balance.pt`；Actor、Critic、normalizer 和 optimizer 从零初始化；按 A 3000（失败扩展 2000）→B1 4000→B2 4000→C 19000 的原课程训练，4096 env、seed 42、std 0.10、lr `1e-4`，每 1000 轮保存。
- 产物目录：`logs/reproduction/curriculum/gsi015_decay10_static50_chest_seed42/`；W&B 和阶段评估将在运行后补充。
- Smoke：64 env、2 iteration、TensorBoard，Actor/Critic 输入 `128/992`，`resume=false`，GSI 接受率 `63.57%`；PPO 更新和 `model_0.pt`/`model_1.pt` 均有限，无 NaN/Inf。
- 阶段 A：正式 run 于 `2026-11-09T08:47:09Z` 启动，4096 env、3000 iteration、每 1000 轮保存；W&B <https://wandb.ai/qc13659041465-ustc/smp/runs/rqeze6ny>。
- 阶段 A 评估：`model_2999.pt` 的 1,024 episode 确定性评估速度 p95 `0.43056 m/s`，未通过 `0.30 m/s` 门槛；非力学失败率 `0.1953%`、位移 p50 `0.02816 m` 均通过。首次自动评估因 `target_probabilities` 元组 CLI 传参错误中断，修复后直接复用 checkpoint 完成评估，未重训 A。
- A 扩展：按条件规则从 `model_2999.pt` 恢复 Actor、Critic、normalizer 和 optimizer，追加 2,000 轮；终点 `model_4998.pt`。1,024 episode 确定性评估速度 p95 `0.51741 m/s`，仍未通过 `0.30 m/s` 门槛；非力学失败率 `0.09766%`、位移 p50 `0.02677 m` 均通过，NaN 为 0。W&B <https://wandb.ai/qc13659041465-ustc/smp/runs/r5ibgwrv>。
- B1 首次运行：开始时间 `2026-11-09T12:40:05Z`；从 `model_4998.pt` 恢复完整 PPO 状态，原预算 4,000 轮。训练正常推进到日志 iteration `8669/8998`，全程未见 NaN/Inf；但 `2026-11-09T14:28:31Z` 课程驱动器向已关闭的父终端写输出时触发 `[Errno 32] Broken pipe`，连带结束训练。最新完整周期检查点为 `model_8000.pt`，不是 CUDA、GPU、checkpoint 或数值故障。
- B1 恢复：恢复时间 `2026-11-09T15:19:36Z`。驱动器增加断管容错，并只对明确的 Broken-pipe 中断扫描最高有效 checkpoint；校验 `model_8000.pt` 的 Actor、Critic 和 optimizer 张量全部有限后，以独立 run 补训 998 轮，保持原终点 `model_8997.pt` 和课程总预算不变。恢复 run 使用 4,096 env、相同 seed/力学/PPO 配置，W&B <https://wandb.ai/qc13659041465-ustc/smp/runs/001y78p6>；启动后 GSI 接受率 `63.57%`，已确认 PPO 更新且 `nan_or_inf=0`。


## EXP-019：80% 静止 Prior 与胸推课程

- 开始时间（UTC）：2026-11-09T08:14:00Z
- 结束时间（UTC）：进行中
- 状态：80/20 prior、A/A 扩展、B1/B2/C 与阶段评估全部完成
- 目的：在与 EXP-018 相同的 GSI、奖励和 PPO 配置下，仅把 prior 训练见到的静止窗口比例提高到 80%，比较 prior 数据配比对阶段 A 和最终胸推能力的影响。
- 主要方法：每个 epoch 保持 3360 个样本，按 seed 42 精确抽取 2688 个静止窗口和 672 个动态窗口；从零训练 10,000 epoch、128 维、2 层、无 EMA 的 DDPM；保存为独立 `pretrained_lafan_low_speed_static80.pt`，通过有限性、速度和 GSI 验收后执行完整从零 PPO 课程。
- 采样规范：`docs/LAFAN_LOW_SPEED_STATIC80_SAMPLING.json`；课程目录：`logs/reproduction/curriculum/gsi015_decay10_static80_chest_seed42/`。
- Prior 结果：1 epoch smoke loss `0.928617`；正式训练最终 loss `0.037343`。4,096 个 DDPM 样本全部有限，速度 mean/p95/max 为 `0.02074/0.04881/0.14421 m/s`；旧 50/50 prior 的对应 p95 为 `0.07531 m/s`，新 prior 降低 `35.18%`。GSI 接受率 `64.45%`，保留 1,024 个窗口，无 root-speed 拒绝，满足门槛。
- Prior 产物：`datasets/pretrain_ckpt/pretrained_lafan_low_speed_static80.pt`，SHA-256 `2f9682f9d1e3e4a5298576f637736345e69fe9fea510992d4b9f7afd5f430dec`；旁车元数据 `datasets/pretrain_ckpt/pretrained_lafan_low_speed_static80.metadata.json`；验证报告 `logs/reproduction/data/low_speed_static80_prior_validation.json`。
- Prior W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/zasfr208>；仅同步标量和配置，未上传模型。


## EXP-020：80% 静止 Prior 左右肩正前方卸力课程

- 开始时间（UTC）：2026-11-09T08:14:00Z
- 结束时间（UTC）：进行中
- 状态：B1/B2 完成；C 于 iteration 16202 收到外部 SIGTERM，最后完整 checkpoint 为 `model_16000.pt`，旧产物保留为慢速肩推对照
- 目的：训练机器人应对来自正前方、作用于左肩或右肩的向后推力，避免沿用原控制器的侧向肩推定义。
- 主要方法：左右肩按 episode 各 50%；球头对准所选肩部并沿机器人 heading 局部 `-x` 推进；复用 EXP-019 实际终止的无接触 A checkpoint，随后独立训练 B1/B2/C 共 27000 轮。每阶段分别固定左肩、右肩各评估 1024 episodes，C 阶段分别录制回放。
- 血缘与产物：父 manifest 固定为 `logs/reproduction/curriculum/gsi015_decay10_static80_chest_seed42/curriculum_state.json`；本实验目录为 `logs/reproduction/curriculum/gsi015_decay10_static80_shoulder_seed42/`。


## EXP-021：双 A100 并行完成三组卸力实验

- 开始时间（UTC）：2026-11-09T17:40:06Z
- 结束时间（UTC）：进行中
- 状态：双卡 smoke 通过；实验 1 与实验 2 已并行正式运行
- 目的：保持三组正式实验的 PPO 语义和训练量不变，仅通过实验级双卡并行缩短总墙钟时间；实验 3 在实验 2 的实际终止 A checkpoint 可用后占用首张空闲卡。
- 主要方法：课程驱动器增加物理 `gpu_id` 隔离并在 manifest 记录 GPU index、UUID 和型号；总控以事件方式维护 PID、GPU、父 checkpoint、阶段依赖和失败状态。实验 1/2 分别绑定 GPU 0/1；实验 3 只接受通过门槛的 A 或完成的条件 A 扩展，先跑左右肩接触 smoke，再开始 B1/B2/C。SIGINT 会转发到两个独立进程组并等待 W&B/checkpoint 收尾，`exit code -2/SIGINT` 仅在存在有效且领先父 checkpoint 的阶段模型时允许恢复。
- 单元验证：Ruff 通过；课程恢复与双卡调度相关 14/14 个 unittest 通过，覆盖 GPU 隔离、实验 1/2 独立命令、实验 3 的 A 依赖、首张空闲 GPU 和 `model_11000.pt → model_12996.pt` 的 1,997 轮恢复计算。
- 双卡 smoke：GPU 0 使用 50/50 prior、GPU 1 使用 static80 prior，同时运行 64 env、2 iteration；两边均为 24 steps/env、5 PPO epochs、4 minibatches，保存 `model_0.pt/model_1.pt`，TensorBoard 与 checkpoint 全部有限。产物位于 `logs/reproduction/curriculum/dual_gpu_smoke_20261109/`。
- GPU：GPU 0 为 `GPU-110dd33b-d093-af41-5fbe-3d5f3b2e2378`；GPU 1 为 `GPU-4169da3c-33af-015e-5902-355f8e02a6c9`，均为 A100-SXM4-80GB。
- 正式启动：总控 PID `11889`；实验 1 curriculum PID `11894`，从 B2 `model_11000.pt` 恢复 1,997 轮到固定终点 `model_12996.pt`；实验 2 curriculum PID `11895`，使用 static80 prior，从随机 Actor/Critic/normalizer/optimizer 开始 A 3,000 轮。两组正式参数快照均确认 4,096 env、24 steps/env、5 PPO epochs、4 minibatches、每 1,000 轮保存。
- 当前正式 run：实验 1 B2 恢复目录为 `logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-09_17-41-39_gsi015_decay10_static50_chest_b2_resume_from_11000_seed42/`；实验 2 A 目录为 `logs/reproduction/rsl_rl/smp_force_unloading_g1/2026-11-09_17-41-54_gsi015_decay10_static80_chest_a_seed42/`。两卡启动后利用率约 71%/77%，显存约 4.8/4.9 GiB，`nan_or_inf=0`。
- 总控产物：`logs/reproduction/curriculum/gsi015_static80_shoulder_suite_state.json`；分进程日志位于 `logs/reproduction/curriculum/gsi015_static80_shoulder_suite_logs/`。两个 8,192 环境短测目录仅作为性能基准，不写入三组正式 manifest 或结果比较。

- W&B：实验 1 B2 恢复 run <https://wandb.ai/qc13659041465-ustc/smp/runs/5l9urmyx>；实验 2 A run <https://wandb.ai/qc13659041465-ustc/smp/runs/2tgzwmvs>。

## EXP-022：缩小球头的胸前推手速度梯度

- 开始时间（UTC）：2026-11-10T13:34:58Z
- 结束时间（UTC）：进行中
- 状态：实现、单元测试、真实接触 smoke 和完整 CLI dry-run 均通过；GPU0 正式训练运行中（B1）
- 目的：解决原标称 `0.05 m/s` 推手速度过慢的问题，观察更快纵向胸推下策略的卸力、后退、峰值力、足滑和事件成功率变化。
- 主要方法：使用 80% 静止 prior `pretrained_lafan_low_speed_static80.pt`、GSI root 平面速度上限 `0.15 m/s` 和 `exp(-10·(vx²+vy²))` 静止奖励；从 EXP-019 的无接触 A 扩展 `model_4998.pt` 分叉，不加载旧接触策略。B1/B2/C 推手速度依次为 `0.05/0.10/0.20 m/s`，训练 `4000/4000/19000` 轮；对应力上限与 ramp 仍为 `0.05/0.10/0.15 mg` 和 `1.0/0.5/0.25 s`，以保持原课程安全约束。
- 球头：半径由 `0.175 m` 减到 `0.10 m`；球体积为旧球头的 `18.659%`（减少 `81.341%`）。质量仍为 `5 kg`，仅隔离几何尺寸和速度梯度，不同时改变质量定义。
- 计算资源：4,096 env、seed 42、std 0.10、lr `1e-4`、每 1,000 轮保存；仅使用物理 GPU0，胸推完成后才允许肩推开始。
- Smoke：64 env、0.20 m/s、半径 0.10 m；实际检测到胸前接触，最大接触力 `243.865 N`，导轨横向误差 `1.16e-10 m`，Actor/Critic `128/992`，全部有限。
- 正式训练：`2026-11-10T13:36:55Z` 从 EXP-019 A 扩展 `model_4998.pt` 启动 B1；GSI 安全池接受率 `62.16%`，PPO 已进入更新且 `nan_or_inf=0`。W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/lyyz6e68>。
- 产物：`logs/reproduction/curriculum/gsi015_decay10_static80_r010_v005_010_020_chest_seed42/`；评估目录 `logs/reproduction/evaluation/gsi015_static80_r010_speed_gradient_gpu0/`。


## EXP-023：缩小球头的左右肩推手速度梯度

- 开始时间（UTC）：2026-11-10T13:34:58Z
- 结束时间（UTC）：排队中
- 状态：实现、单元测试、真实左右肩接触 smoke 和完整 CLI dry-run 均通过；肩推 smoke 已完成，正式课程严格等待 EXP-022 释放 GPU0
- 目的：在左右肩各 50% 且球头从机器人正前方向后推进的条件下，验证相同高速梯度的双侧卸力能力，避免一侧结果掩盖另一侧。
- 主要方法：与 EXP-022 使用相同 80% prior、GSI 0.15、陡峭静止奖励、半径 0.10 m、`0.05/0.10/0.20 m/s` 梯度和 B1/B2/C 预算；从同一无接触 A 扩展 `model_4998.pt` 独立分叉。每阶段固定左肩和右肩各评估 1,024 episodes，最终分别录制回放。
- 计算资源：仅物理 GPU0；总控严格串行，不能与胸推并发。
- Smoke：64 env 覆盖左肩和右肩，0.20 m/s 下实际检测到接触，最大接触力 `665.661 N`，导轨横向误差 `7.48e-9 m`，Actor/Critic `128/992`，全部有限。正式课程保留阶段力上限和紧急力终止。
- 产物：`logs/reproduction/curriculum/gsi015_decay10_static80_r010_v005_010_020_shoulders_seed42/`；总控状态 `logs/reproduction/curriculum/gsi015_decay10_static80_r010_speed_gradient_gpu0_suite_state.json`。

## EXP-024：GPU1 五条件扩散 Prior 与统一卸力策略

- 开始时间（UTC）：2026-11-10T16:56:00Z（持久化总控启动；此前完成实现、数据筛选和 smoke）。
- 状态：10,000 epoch Prior 正式训练于 2026-11-10T17:18:37Z 完成；逐类 4096 样本验收于 17:19:00Z 未通过左右肩门槛，总控状态 blocked_prior_gate。PPO（含 64 env/2 iteration 策略 smoke）没有启动，未发布为正式 SMP prior；候选权重、全部日志和回放均保留。不是 CUDA/NaN/终端断管中断。
- 目的：用同一个条件扩散 Prior 表达无接触静止/腰回中、左肩正向腰 yaw、右肩镜像负向腰 yaw、胸口后退、背部前移，在严格 Prior 验收后训练覆盖四部位的统一卸力 PPO。
- GPU 隔离：所有 GPU 子进程使用物理 GPU1（UUID GPU-4169da3c-33af-015e-5902-355f8e02a6c9），进程内 cuda:0；GPU0 的 EXP-022/023 不停止、不切卡。
- 数据：完整 LAFAN G1 CSV；subject 1–4 训练、subject 5 验证；原片段及镜像严格同划分，标签是动作意图而非真实力标注。筛选平移速度、腰相对骨盆 yaw、骨盆转向和关节运动，使用 MuJoCo FK 检查关节限位、自碰撞、足部支撑及地面穿透；歧义窗口剔除。训练五类窗口数依次为 2646/478/478/552/1986（共 6140），验证为 426/98/98/78/248（共 948）。NPZ 保留 (N,10,59)、condition_id、帧区间和筛选指标；windows.csv 和 manifest 记录源 SHA、镜像及划分。训练集独立归一化，不覆盖旧数据。
- Prior：每 epoch 按 seed 42 类内有放回抽样，每类 672，共 3360；10% 条件丢弃使用独立 NULL ID 5。类别 embedding 与 timestep embedding 相加后进入现有 adaLN，参考 [DiT 类别条件设计](https://arxiv.org/html/2212.09748v2)，保持旧无条件模型接口和 checkpoint 兼容。128 维、2 层、50 diffusion steps、10 noise samples、batch 1024、lr 3e-4、无 EMA、10,000 epochs，从零初始化；checkpoint 保存 optimizer、RNG、采样 epoch、数据 SHA、字典及归一化。
- 条件运行：注册独立 Smp-Force-Unloading-Conditional-G1；GSI 只采样条件 0，上限 0.15 m/s；运行中实际接触力选主导部位，连续 3 控制帧超过 0.0075 mg 激活、低于 0.005 mg 解除。跨条件 10 帧历史使用 NULL；条件×扩散时间步的误差归一化随 PPO 保存、恢复。Actor/Critic 不新增未来目标信息，维度仍 128/992。
- 机械接入：球半径 0.10 m、质量 5 kg，按 capsule 碰撞几何设置 2 cm 初始表面间隙；肩部瞄准上端点，修正默认弯肘姿态下瞄准中心会与前臂/手腕初始相交的问题，仅影响新任务。导轨在事件开始时固定 heading，胸/肩沿 -x、背沿 +x；行程上限只作遥测。接触分类利用实际分区力的物理子步历史，避免末子步接触解除漏检，不使用预定目标代替传感器分类。
- Smoke：1 epoch loss 0.934406、验证 macro loss 0.887693；64 env B1 机械 smoke 固定机器人姿态以隔离几何（不是策略能力测试），四部位均真实接触且正确分类，首次激活约 2.54–2.86 s；实际球半径 0.10 m，导轨横向误差约 7.45e-9 m，无 NaN/Inf。正式策略的 64 env/2 iteration smoke 排在 Prior 验收后。49 项现有及新增单元测试通过，包含完整 PPO/SMP 状态恢复、Prior SHA 拒绝、四部位独立评估和真实 CLI 解析。
- PPO：A 3000（首次门槛失败追加 2000）→B1 4000→B2 4000→C 19000；A 的 Actor/Critic/normalizer/optimizer 全部从零，后续仅恢复紧邻前阶段完整 checkpoint。4096 env、seed 42、std 0.10、lr 1e-4，每 1000 轮保存；GSI 0.15、静止函数 exp(-10·(vx²+vy²))；A 静止权重 0.9，后续 0.1，四部位各 25%。接触阶段分别评估无接触及四固定部位各 1024 episodes，独立输出方向、腰旋转、骨盆转向、位移、足滑、恢复静止和最终回放。
- 验收约束：每条件独立采样 4096 窗口并用配对噪声比较。Prior 必须通过有限性、静止速度、GSI 接受率和逐条件方向门槛才允许启动 PPO；不自动增加正式 Prior 预算。PPO 质量门槛失败仅警告（A 条件扩展除外），数值、checkpoint、进程或 GPU 异常停止并保留现场。
- 产物与恢复：独立目录 logs/reproduction/conditional_contact_v1/；总控 manifest.json、driver.log，prior/latest.pt（100 epoch 周期及首 epoch）、1000 epoch 快照、最终 prior/pretrained.pt；通过门槛后才以 SHA 校验方式发布 datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v1.pt。总控使用独立会话和文件日志，防止终端断管；保留进程、阶段和恢复记录。历史 Prior、策略、结论不覆盖。
- W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/l9hk1gas>；仅同步标量和配置，模型保留本地。
- 中期诊断（epoch 3000，非正式验收）：GPU1 每类 64 个样本；静止窗口最大速度 p95 为 0.0353 m/s，胸/背平均 vx 为 -0.1550/+0.1727 m/s；左右肩方向达标率 67.19%/70.31%，尚未达 80%。独立安全 GSI 接受率 98.59%，五类 EGL 动作预览生成成功。原始训练/验证肩部窗口均 100% 符合固定方向判据，提示早期生成分布仍有不足，而非标签索引/镜像错误；保留门槛，等待 10,000 epoch 正式验收，不据此提前开启 PPO。诊断产物：logs/reproduction/conditional_contact_v1/prior_gsi_preview_diagnostic_epoch3000/。
- 正式 Prior 结果：最终 train loss 0.114421，subject 5 验证 macro loss 0.166689；所有 20,480 个生成窗口有限。静止窗口最大平面速度 p95 0.026113 m/s（通过），安全 GSI 接受率 98.59375%（通过）；胸口平均 vx -0.145396 m/s、方向成功率 99.73145%（通过）；背部平均 vx +0.179684 m/s、方向成功率 98.21777%（通过）；左右肩完整方向判据成功率 66.62598% / 68.38379%（均未达 80%）。
- 失败定位：左右肩腰 yaw 符号正确率 90.89355% / 96.14258%，但进入或保持有效腰姿比例仅 68.96973% / 70.36133%，再叠加骨盆转向约束后得到最终通过率。肩部训练数据的平均末帧腰 yaw 幅度为 0.07117 rad、窗口转角幅度 0.01989 rad；最终生成分别为左 0.04565/0.01505 rad、右 0.05607/0.01356 rad。原始肩部训练及 subject 5 验证窗口全部符合固定判据，故当前证据指向生成动作强度和保持分布不足，而非标签交换或采样条件对齐错误；不放宽判据、不增加 guidance scale。
- 无新增训练的复核：验证集最优 checkpoint 为 epoch 5050（macro loss 0.152346）。独立按每类 4096 样本复核后，左右肩仅 61.10840% / 64.35547%，仍未通过；因此不能仅用 early stopping checkpoint 解决。报告位于 prior_validation_best_epoch5050/。原始去噪误差的跨条件比较还出现 back 条件偏好，作为额外风险保留；不能把未经条件归一化的该矩阵当作动作分类准确率或 PPO 能力。
- 最终候选产物：logs/reproduction/conditional_contact_v1/prior/pretrained.pt，SHA-256 380e6fd3591cbde4f049f88ec87f549732fd98bf311fa3d7a0badb11813feebb；正式验收 prior_validation/report.json，五类 prior_validation/*_fk_samples_5x_slow.mp4（FK 动作预览，不是受力策略回放）。未创建 datasets/pretrain_ckpt/pretrained_lafan_contact_conditional_v1.pt；没有 PPO checkpoint 或受力策略视频。验收标量已同步到上述 W&B run summary，没有上传模型。
- 后续所需：建议单独审查并增强肩部“转入/转后保持”数据的幅度和类别内配比，使用新数据版本/新输出目录重训、保留本次失败结果；这将超出本轮已经用完的 10,000 epoch 正式 Prior 预算，尚未执行。磁盘仅余约 0.75 GiB，长程 PPO 前还需用户指定扩容或归档位置；未删除历史产物。GPU1 已空闲，GPU0 原实验继续。


## EXP-025：GPU1 条件 Prior 第二版——增强肩部转入/保持数据

- 准备时间（UTC）：2026-11-11T02:08:45Z；总控启动 02:10:00Z，正式 Prior 启动 02:11:22Z。状态：数据、57 项单元测试、Prior/四部位机械 smoke 通过；GPU1 正式训练运行中，尚未进行正式条件验收或启动 PPO。
- 授权与目的：用户已批准额外一轮 10,000 epoch 正式 Prior 预算；针对 EXP-024 左右肩生成幅度/保持不足，增强肩部可辨识动作数据，不降低验收门槛。Prior 与后续 A 策略仍各自从零初始化，不从 EXP-024 权重继续训练。
- 数据方法：只从 EXP-024 已通过 FK/关节限位/地面穿透/支撑/自碰撞检查的原始窗口取子集，不放大关节角、不生成虚构标签。保留源文件、源 SHA-256、父 manifest/shard SHA、父窗口 index、50 Hz 帧区间、镜像、subject 划分及 phase_id。subject 1–4 训练、subject 5 验证不变；静止/胸口/背部窗口全部保留，重新计算训练集归一化。
- 肩部筛选：统一为对应方向有符号腰 yaw（相对骨盆），转入要求末帧 ≥ 0.08 rad、窗口增量 ≥ 0.02 rad、首帧 ≥ 0；保持要求末帧 ≥ 0.12 rad、全窗口最小值 ≥ 0.10 rad、增量绝对值 ≤ 0.01 rad。两类均要求全窗口不超过 0.35 rad；原始平移速度/骨盆转向/安全约束不变。左右镜像严格交换标签并保持相同筛选规则。
- 数据数量与局限：训练五类为 2646/89/89/552/1986（共 5362），验证为 426/16/16/78/248（共 784）。每侧训练含 60 个转入、29 个保持；验证含 9 个转入、7 个保持。肩部独立窗口数量明显减少且含重叠，存在泛化/过拟合风险；不会把 3360 次抽样视作 3360 个独立样本，也不宣称扩充了真实动作覆盖。
- 采样与模型：seed 42，每 epoch 五类各 672，共 3360；左右肩各在转入/保持池内均匀有放回抽取 336/336，其余三类均匀有放回。模型和损失不改：d_model=128、2 layers、50 DDPM steps、10 noise samples、batch 1024、lr 3e-4、10% 独立 NULL 条件丢弃、无 EMA、guidance scale 1；正式预算固定 10,000 epochs。checkpoint 增加可审计 sampling 配置并保存完整 optimizer/RNG/sampler epoch，恢复时核验数据哈希与采样轮数。
- 存储处理：系统盘约余 258 MB，未删除或搬移历史数据。新产物实际写入用户私有 GPFS 目录 /personal_fast_storage/smp_experiments/conditional_contact_v2_20261111/，通过 logs/reproduction/conditional_contact_v2/ 符号链接访问；数据写入验证成功。挂载盘显示约 484 TB 空闲是共享文件系统容量，不代表个人配额。新实验 W&B/Matplotlib/Warp/CUDA/Triton/Inductor 缓存及临时目录均指向该独立目录；未改变 GPU0 的原存储路径，系统盘剩余空间仍是旧实验风险。
- GPU 隔离与恢复：仅物理 GPU1，UUID GPU-4169da3c-33af-015e-5902-355f8e02a6c9，进程内 cuda:0；启动前检查 GPU1 空闲与 UUID。使用独立会话、文件日志和排他锁；异常不自动重置训练预算，不停止其他 GPU 进程。GPU0 EXP-022/023 继续运行。
- 验收与 PPO：沿用 EXP-024 的 4096 样本/条件验收、静止速度 p95 ≤ 0.15 m/s、安全 GSI ≥ 50%、逐肩方向通过率 ≥ 80%、胸/背速度及方向门槛。Prior 全部通过后才发布独立 pretrained_lafan_contact_conditional_v2.pt 并启动统一卸力 PPO；A 3000（失败追加 2000）、B1 4000、B2 4000、C 19000，4096 env、std 0.10、lr 1e-4、save 1000、GSI 0.15、静止尺度 10、四部位各 25% 均不变。模型仅本地保存；W&B 同步标量与配置。
- 验证：57/57 unittest、Ruff 通过；新增覆盖 20% 五类与肩部 50/50 精确配比、采样断点复现、缺失 phase 拒绝、镜像/近中立筛除、原始窗口与 subject 血缘、GPU/缓存隔离及存储禁止回退至系统盘。1 epoch scratch smoke 的 train loss 为 0.934979、validation macro loss 为 0.889869，checkpoint 有限。64 env B1 机械 smoke 四部位均真实接触且正确分类，首次条件激活 2.54–2.86 s，Actor/Critic 128/992，导轨横向误差 7.45e-9 m；该 smoke 固定机器人姿态，仅验证接触几何而非策略能力。
- 正式运行证据：总控 PID 407999、Prior 子进程 PID 408462；启动日志明确 Scratch=True、completed_epochs=0，正式与 smoke 使用不同目录。已完成至少 175/10,000 epochs，train loss 0.221814、validation macro loss 0.254745，尚不能据此宣称方向验收通过。GPU1 利用率约 83%；GPU0 原任务利用率约 76%，未中止或修改。
- W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/yjw7i9mw>；在线同步标量及配置，无模型上传。数据 manifest SHA-256：4e4a8226e40b43c3c00447c745ce75eea2ed6d065c55366650218e96f44ff57d。
- 独立产物：logs/reproduction/conditional_contact_v2/{manifest.json,driver.log,data/,prior_smoke/,prior/,prior_validation/,curriculum/,evaluation/,rsl_rl/}；EXP-024 失败结论、候选权重和全部历史产物保持不变。




## EXP-026：Conditional 肩部对比 Score 与 B1/B2 专项课程

- 开始时间（UTC）：2026-11-12T08:26:17Z。
- 当前状态：实现和单元测试完成；冻结校准的静止偏置硬门槛失败，状态为 `blocked_calibration`。未迁移正式 A checkpoint，未启动 PPO smoke、B1 或 B2，也未创建 W&B run。
- 目的：冻结 Conditional v2 Prior，从已通过 A 门槛的 `model_2999.pt` 分叉，只训练左右肩 B1/B2；不增加显式转腰奖励或 condition one-hot，检验同噪声镜像条件对比 SMP score 与肩部专项课程能否驱动腰部退让。
- 固定身份：Prior `pretrained_lafan_contact_conditional_v2.pt`，SHA-256 `ade23a0bf027addbd3a210ae82ec3d979b2d38c6e16c978e459f930d1e004734`；父 A checkpoint SHA-256 `b2b6dc8f4fae3a93ce0028f3a594bceb09494590bb4b1d4f29ee24355006c3fb`；物理 GPU0 UUID `GPU-d0c5535e-c4e1-8b72-e6f1-ae82f0b75771`，启动检查时无其他计算进程。
- Score 实现：59 维权重默认 1，仅特征索引 21（`waist_yaw_joint`）为 10；固定扩散时间步 `(8,15,22)`。正确肩条件和镜像肩条件使用完全相同的 `x_t` 与噪声；score 使用冻结条件/时间步均值、`exp(-ReLU(z-1))` base 和 `sigmoid(4·margin)` contrast，输出限制在 `[0,1]`。无接触只用 base。
- 条件切换：保留力 EMA、开启/解除阈值和连续 3 帧 debounce；第 3 帧确认后立即切换到实际肩条件。混合历史仅使用 `history == 当前条件` 的帧，旧 Conditional 任务仍保持 NULL 分支和完整 10 帧稳定等待。
- 校准方法：训练集按 seed 42 对五个条件各有放回抽取 4,096 窗口，生成冻结参考均值；数据 manifest SHA-256 `4e4a8226e40b43c3c00447c745ce75eea2ed6d065c55366650218e96f44ff57d`。校准 artifact SHA-256 `e02673cb357a4a0a7b0302c62f47d9f865c2bfbad5b78a62df1141c23352891d`。
- 肩部验证结果：subject 5 左肩/右肩正确条件优于镜像条件的比例均为 `100%`；median margin 分别为 `1.03918` 和 `1.00324`，均通过 `≥0.5` 门槛；全部数值有限。
- 静止偏置结果：静止窗口左肩偏好率 `59.4924%`，通过 35%–65% 门槛；但 `mean(|margin|)=0.355576`，未通过 `≤0.2` 门槛。追加 32 组噪声诊断后，对每个静止窗口先做噪声平均的 `mean(|margin|)` 仍约 `0.354885`，确认是 v2 Prior 的系统性左右肩能量偏置，不是单次噪声方差；特征索引 21 已核对为 waist yaw。
- 验证：V3 与旧 Conditional 专项测试共 14/14 通过，覆盖同噪声配对、镜像映射、腰 yaw 10× 权重、时间掩码、3 帧切换、旧任务 NULL 行为、score 边界和逐张量 checkpoint 迁移；仓库全量回归测试 63/63 通过。Ruff 和 Python 语法检查通过；独立任务注册、runner、物理参数及校准路径可加载。
- 硬停止说明：既定计划要求校准失败时停止，因此未用失败校准继续 PPO，也未覆盖原 Prior、父 checkpoint、旧任务或历史结果。若后续决定修正 v2 Prior 的静止左右对称性，或明确重定义/调整静止偏置门槛，应作为新的用户决策记录后再恢复总控。
- 产物：`logs/reproduction/conditional_contact_shoulder_score_v3/manifest.json`、`calibration/report.json`、`calibration/shoulder_score_calibration.pt` 和 `logs/unit_tests.log`。逻辑目录映射到 `/personal_fast_storage/smp_experiments/conditional_contact_shoulder_score_v3_20261112/`。

### EXP-026 补充诊断：依次判断 Score 修正与 Prior 重训必要性

- 时间与目的：2026-11-12 UTC，09:15:17 建立候选协议，约 09:18–09:19 采集策略轨迹。冻结 v2 Prior，先验证校准/base 修正，再检验留出动作与真实策略，避免仅因静止偏置门槛失败而直接重训。仅既定物理 GPU0，无网络训练或模型上传。
- 内部校准/选择：subject 1–3 校准、subject 4 选参；六个候选为均值/95 分位参考各搭配原指数、0.1 倍指数衰减、倒数型 base。特征权重和对比项不变，镜像/重叠窗口按主体分组。内部六个均通过诊断，选定 `train123_p95_exp_0p1` 后锁定参数。1–4 已用于训练 Prior，此划分只隔离 Score 校准；subject 5 先前已审阅，不能称为完全盲测，本轮未据其结果再调参。
- 外部测试：subject 5 每肩 16 窗口（进入 9、保持 7）及静止 426 窗口，64 组同噪声；六个保留 base 的候选均未满足预先记录的诊断判据（并非正式课程新门槛）。选定方案正确动作高于静止为左/右 `59.7938% / 84.5368%`，进入转腰为 `50.4630% / 78.9242%`；正确分数均值虽恢复到约 `0.05387 / 0.05563`，排序仍不可靠。
- 纯对比对照：不纳入正式候选。去掉 base 后 subject 5 正确动作高于静止为 `97.2805% / 98.8474%`，表明方向信息仍存在，但不足以证明可直接用于 PPO。
- 真实轨迹：A `model_2999.pt` 无接触及左右 B1、旧 `model_17000.pt` 左右 B2，各 64 环境、600 控制帧、确定性策略。每 5 帧采样，排除 reset 后不足 10 帧的混合 GSI 窗口；共 `37,789` 窗口，评分 `10,416` 个（保留全部实际接触，固定 seed 子采样无接触）。新 Score 只作回放后评分，没有控制策略或更新网络。
- 真实误奖励：实际肩部接触、腰部相对首次接触变化绝对值 <0.02 rad、未终止且无紧急力的 38/13 个左/右窗口，按实际条件历史掩码计算的纯对比奖励均值为 `0.97915 / 0.75282`（16 次噪声）。这些不是独立 episode 或完整安全认证。实际轨迹没有满足 v2 进入/保持幅度要求的肩部正例，不能宣称真实策略正负奖励排序已验收。
- 接触覆盖：旧策略左肩 B2 的 64 个完成 episode 中首次实际肩部/胸部接触为 23/7，右肩为 7/16，其余无确认接触。旧指标的 4/14 个方向成功全部来自胸部，肩部为 0；未将胸部退让计为肩部转腰。后续需核对运动中的肩部命中几何。
- 判断：本轮限定的校准/base 修正不足以可靠解决问题，纯对比方案也有真实误奖励风险。建议补充中立→转腰→保持的肩部姿态覆盖，准备独立的新 Prior 训练/验证，而非简单增加 epoch 或放宽门槛；这不是证明所有 Score-only 设计都失败。真实未响应轨迹可作诊断负例，不应误标为肩部正例。新训练预算另行明确，本轮没有启动重训/PPO。
- 实现限制：V3 环境入口 `compute_features(normalize=True)` 与 buffer 无参数接口不兼容，正式 smoke 前仍需修复。此次使用纯评分函数和旧任务采集，不受该入口问题影响；未修改生产奖励实现。
- 保持与产物：原 Prior、校准 SHA、旧 checkpoint、正式门槛及 `blocked_calibration` 状态保持不变。独立产物和复现脚本为 `logs/reproduction/conditional_contact_shoulder_score_v3/diagnostics/score_decision_20261112T091517Z/`，包含 `decision_summary.json`、逐候选报告、真实轨迹、`policy_score_report.json` 和 SHA-256 清单。

## EXP-027：肩部 80%＋静止 20% 独立三条件 Prior——数据审查阻断

- 数据审查时间（UTC）：2026-11-12T10:56:38Z 至 10:56:56Z。当前状态：`blocked_data_gate`，不是训练中断或 GPU 故障。正式 Prior 已完成 `0/10000 epochs`；未执行单 epoch smoke、未创建 W&B run、未发布新 Prior、未启动 PPO。
- 目的与授权：按用户确认的方案重新挖掘完整 LAFAN，准备新的静止/左肩/右肩 Conditional Prior；左右肩合计 80%、静止 20%，不含胸口后退或背部前移，不继承 v2 权重。用户选择先重挖现有 LAFAN、不合成动作，允许小幅调整步，并批准独立 10,000 epoch 预算和三划分。
- 数据方法：扫描 40 个原始 CSV，50 Hz、10 帧窗口、stride 5；原始和镜像分别执行筛选与 MuJoCo FK/关节限位/自碰撞/支撑/穿透检查。subject 1–3 为训练、4 为验证、5 为测试；原片段与镜像不跨划分，独立计算训练集 1%/99% 归一化统计。静止额外要求腰 yaw 窗口增量绝对值不超过 0.005 rad。
- 肩部阶段：早期起转为首帧有符号 yaw ≥0、末帧 <0.08 rad、增量 ≥0.01 rad；明确转入为首帧 ≥0、末帧 ≥0.08 rad、增量 ≥0.02 rad；保持为末帧 ≥0.12 rad、窗口最小值 ≥0.10 rad、增量绝对值 ≤0.01 rad。全窗口有符号 yaw ≤0.35 rad，root 平面速度 ≤0.20 m/s，骨盆 yaw 变化 ≤0.05 rad。
- 上下文与连接：完整 1 秒位置上下文采用 51 个 50 Hz 端点，中心取最近的较早帧；净平移 ≤0.10 m、累计路程 ≤0.15 m，边界不足剔除。早期起转必须在随后 1 秒内连接到同源、同方向且通过物理检查的明确转入/保持窗口，连接段有符号 yaw 保持在 [0,0.35]。这些约束没有为增加数量而放宽。
- 抽样实现与测试：新增独立三条件 schema，静止/左肩/右肩 ID 为 0/1/2，NULL 为 3；旧五条件 schema 未改。每 epoch 配额固定 672/1344/1344，左右肩每侧早期/明确转入/保持为 336/336/672，类别与阶段内部均匀有放回；采样器保存 seed/样本数/epoch 并可精确恢复。此处是已实现的抽样配置，不表示已运行训练。

| 划分 | 静止窗口 | 每侧早期起转 | 每侧明确转入 | 每侧保持 | 每侧肩部总数 |
|---|---:|---:|---:|---:|---:|
| 训练 subject 1–3 | 1504 | 8 | 27 | 13 | 48 |
| 验证 subject 4 | 882 | 0 | 8 | 15 | 23 |
| 测试 subject 5 | 338 | 0 | 11 | 10 | 21 |

- 阻断原因一：验证与测试的左右肩早期起转均为空，共四个阶段覆盖门槛失败。逐级审查显示：验证每侧早期候选 71 → 上下文后 40 → FK 后 26 → 连接检查后 0；测试为 78 → 46 → 41 → 0。说明存在短时起转碎片，但没有满足本轮完整约束的“起转→明确转入/保持”连接，不能将其直接当作合格早期肩部正例。
- 阻断原因二：按原始文件与帧区间去重、镜像不算新来源、重叠窗口不重复计数，训练主体的非重叠肩部窗口从同主体旧 v2 的 47 个降至 38 个。虽有 10 个不与旧肩部窗口重叠的新窗口，但不足以弥补严格筛选后的覆盖减少，未满足“实际增加非重叠肩部片段”的门槛。每个训练肩部阶段来自至少两个原始文件的门槛通过。
- 验证结果：19 项本轮新增测试及历史回归合计 82/82 通过；Ruff 检查通过。独立复核全部 shard/源文件 SHA、有限性、条件与阶段、subject 划分、训练集归一化、早期连接目标、门槛重算、80/20 精确配比及采样恢复。旧无条件/五条件和 PPO 相关测试保持通过；没有将这些 CPU 测试称为新 Prior GPU smoke 或新模型验收。
- GPU 与存储：指定物理 GPU0 UUID `GPU-d0c5535e-c4e1-8b72-e6f1-ae82f0b75771` 校验通过，检查时无其他计算进程；本次数据审查全部在 CPU 上执行。产物实际位于 `/personal_fast_storage/smp_experiments/conditional_shoulder80_prior_v1_20261112/`，以 `logs/reproduction/conditional_shoulder80_prior_v1/` 链接访问。持久盘 1 MiB 写入并 fsync 测试通过；系统未提供 quota/mmlsquota/lfs 工具，因此共享剩余空间不视为个人配额保证。
- 产物：根 `manifest.json` 保存批准预算、GPU、存储、历史身份及未启动的训练状态；`data/manifest.json`、`data/audit.json`、`data/windows.csv`、三划分 NPZ 与 `norm_stats.npz` 保存候选数据和逐级筛除证据。数据 manifest SHA-256 为 `7188e78d79d70dae7001fb69aa8c3fd357de220a01c258ed2dd348450b5252d9`；测试日志为 `unit_tests_verified.log`。
- 实现范围：新增 `scripts/curate_conditional_shoulder80.py`、`scripts/run_conditional_shoulder80_data_review.py`、`src/smp/pretrain/shoulder80.py` 以及两个测试文件。本轮按批准的停止条件停在数据审查，后续三条件模型加载/训练/选模/验收/发布流水线尚未实施或执行；不要把数据阶段完成误记为完整训练计划完成。
- 历史保持：原 v2 Prior SHA-256 `ade23a0bf027addbd3a210ae82ec3d979b2d38c6e16c978e459f930d1e004734` 和数据 manifest SHA-256 `4e4a8226e40b43c3c00447c745ce75eea2ed6d065c55366650218e96f44ff57d` 复核未变；旧 checkpoint、任务、EXP-026 manifest、校准与结论均未改写。
- 后续边界：继续正式训练前需要用户决定是否补充新的真实动作来源，或批准调整连接/上下文等筛选约束并建立新数据版本。当前不擅自扩大数据来源、不合成动作、不删除早期阶段、不放宽门槛，也不通过重复抽样冒充新增动作。新 Prior 和 PPO 的有效性均未在本轮得到验证。

## EXP-028：G1 Moves 与 AMASS-G1 扩充肩部动作——原数值门槛不变

- 时间（UTC）：2026-11-12 11:29:08 冻结获取计划，11:50:06 完成扩展数据筛选，随后完成产物复核。目的：按用户新授权引入真实动作来源，补齐 EXP-027 缺失的早期起转和来源多样性；本轮仅审查数据，不训练 Prior/PPO，不创建 W&B run。
- 来源：G1 Moves 固定 revision `895d064b2385725e6aabf540461c219afc3e0916`；AMASS-G1 固定 revision `275c4c4a028cadf5241c10a613c2f11913531ad2`。共 758 个动作文件：G1 Moves 7 个；AMASS 的 ACCAD 202、MPI_HDM05 173、CMU 300、MPI_mosh 76 个。另保存 20 个来源/许可/格式说明文件。仅 CSV/数值 NPY，不加载第三方 pickle，不执行下载的回放代码，不下载模型或视频。AMASS 是本轮选定子集，不是全量结论。
- 获取与划分：先按 seed 42 的路径哈希及受试者轮询选文件，再按 `sha256('42:'+来源受试者ID) mod 5` 固定 train/validation/test 为 0–2/3/4；未按筛选结果移动受试者。122 个外部受试者/表演者分组分配为 68/32/22。ACCAD 同一人的不同动作目录归为一组，G1 Moves 同一表演者所有片段归为一组；无法可靠解析主体的 126 个候选路径不纳入。LAFAN 原 subject 1–3/4/5 划分不变，外部受试者不伪装成 LAFAN subject 4/5。
- 格式：G1 Moves 按 60 Hz；AMASS 按文件名及发布方回放规则读取帧率，并固定加回 root 高度 `0.793 m`。此为坐标表示转换，不是逐片段贴地拟合。均按原时间尺度插值到 50 Hz，生成 10 帧、stride 5、59 维特征；未慢放、合成、放大腰角、裁剪关节角或修改模型限位。
- 门槛：直接调用 EXP-027 原 `classify_windows / screen_frames / link_early_windows`，早期/明确转入/保持定义、速度、1 秒上下文、1 秒连接和 FK 安全门槛全部不变。原筛选脚本 SHA-256 为 `d6d131a80bcc1370cd01713b80210569837b2e228251d53a9c780c052a35bdd4`。扩展版按新增来源的主体分组检查相同的覆盖/多样性门槛；原 LAFAN 单独数据仍维持 EXP-027 的阻断结果。
- 恢复记录：首次审查在第 405 个源文件发现 HTTP 下载截短；全量只读检查另发现第 476 个文件同样截短。远端原文件正常，按发布方 LFS SHA-256/大小重新获取这两份文件，并补齐整个批次的 Git blob/LFS 身份校验。778/778 文件最终通过；坏副本和首次部分产物分别保留于 `transport_recovery/`、`review_attempt01_transport_truncation/`，未删除历史产物。修复了下载器的短读/Content-Length 检查；未通过修改动作或放宽筛选来绕过错误。
- G1 Moves 结论：`M_Move5`（转腰出拳）、`M_Move19`（慢速套路）、`M_Move3`（马步）最终肩部窗口均为 0。第一段主要没有通过上下文与完整连接，后两段的肩部候选被当前 G1 踝 roll 限位筛除；未因动作名字或预览效果直接纳入。补查的 `B_StretchDance / B_SpiralDance / B_WiggleDance` 分别获得每侧 39/13/12 个肩部窗口，共每侧 64 个，来自 3 个原始文件、同一表演者，均在预先固定的训练划分。
- AMASS-G1 结论：获得每侧 2555 个肩部窗口，来自 200 个原始文件、47 个来源内主体；另有合格静止窗口。两个新来源合计新增肩部原始文件 203 个、来源内主体 48 个。镜像不算新来源，非重叠窗口也不等同于独立受试者或独立实验。

| 扩展数据划分 | 静止窗口 | 每侧早期起转 | 每侧明确转入 | 每侧保持 | 每侧肩部合计 |
|---|---:|---:|---:|---:|---:|
| train | 3660 | 143 | 612 | 477 | 1232 |
| validation | 1898 | 101 | 340 | 149 | 590 |
| test | 482 | 34 | 325 | 530 | 889 |

- 表内包含各划分的原始与镜像窗口，不把这些计数解释为独立动作数量。按原始文件/帧区间去掉镜像重复及重叠，训练肩部非重叠窗口为 784（旧 v2 同训练主体基线 47，EXP-027 严格 LAFAN 为 38），其中 756 个不与旧 v2 肩部区间重叠。验证/测试为 377/539。扩展数据门槛全部通过，状态 `data_gate_passed_not_trained`。
- 抽样与归一化：仅汇总训练主体计算新的 1%/99% 归一化，不覆盖原统计。保留每 epoch 3360、静止/左/右 672/1344/1344、每肩早期/转入/保持 336/336/672 的精确采样配置；测试了配比及恢复复现，未实际训练。
- 验证：45 项旧 Conditional 回归与 20 项新适配/短读/产物测试共 65 项通过；Ruff 通过。新测试独立重算 gate、归一化、逐 shard 哈希/有限性/标签、早期连接目标、受试者/镜像划分、原 Prior 身份和采样状态。全部为 CPU 工作，没有占用 GPU 或干预现有训练。
- 产物：逻辑目录 `logs/reproduction/external_shoulder_data_review_v1/`，实际位于 `/personal_fast_storage/smp_experiments/external_shoulder_data_review_v1_20261112/`。`plan.json` 固定来源与划分，`publisher_integrity.json` 固定远端身份，`review/manifest.json`、`windows.csv`、NPZ、`norm_stats.npz`、`audit.json` 保存新数据与逐级证据；`source_summary.json` 及 `plots/` 保存逐来源汇总和 7 张腰/骨盆/速度诊断图（不是机器人回放视频）。新 manifest SHA-256：`d5616d77a051266bde3b075534ce434d6159b1a1dd790ab0a56057d63f29eb05`。
- 后续边界：数据审查通过不代表新 Prior 采样、SMP 最终奖励排序或 PPO 受推转腰已通过。原 80/20 三条件训练预算、采样/评分门槛不变，但本轮没有启动训练、重新选模型或发布 checkpoint；EXP-026、EXP-027、原 v2 Prior 与旧数据/策略保持原状。新数据按来源许可使用并保留归属信息；跨数据库未公开的受试者别名不能声称已完全排除。

## EXP-029：扩展真实动作数据上的三条件 Shoulder80 Prior 从零训练

- 时间（UTC）：2026-11-12 13:26:27 启动；13:26:39 单 epoch smoke 通过后启动正式训练。目的：在 EXP-028 已补齐早期起转与来源多样性的真实动作上，训练独立的肩部 80%＋静止 20% Prior，检验完整 V3 SMP 分数能否区分正确转腰、静止与反向转腰。本轮不启动 PPO。
- 数据：复核 EXP-028 manifest SHA-256 `d5616d77a051266bde3b075534ce434d6159b1a1dd790ab0a56057d63f29eb05`，所有数据门槛通过；筛选阈值未放宽。通过合并 `windows.csv` 加载原 LAFAN 和 G1 Moves / AMASS-G1，而非仅扫描 external-only shard 目录。train / validation / test 的静止、左、右数量分别为 `(3660,1232,1232)`、`(1898,590,590)`、`(482,889,889)`；沿用原主体划分与训练集归一化，按 shard 哈希及逐窗口来源校验。
- 配比：每 epoch 3360 个窗口，静止/左肩/右肩为 672/1344/1344；每侧早期/明确转入/保持为 336/336/672，类别/阶段内均匀有放回。10% 条件丢弃发生在精确配比之后；只有 stand / left_shoulder / right_shoulder，NULL ID=3，不包含胸背前后移动类别。
- 模型与预算：独立随机初始化，seed 42，10,000 正式 epochs，batch 1024，d_model 128，4 heads、2 layers、50 DDPM steps、每窗口 10 组噪声；AdamW lr 3e-4、weight decay 1e-4、grad clip 1.0，无 EMA。沿用仓库原 DDPM epsilon-prediction L1 训练损失；腰部 10 倍权重只用于 V3 评分，不额外添加监督或对比训练损失。正式初始 29 个状态张量与 smoke 的未训练初始张量逐项一致，未加载 smoke 训练权重或旧 v2 Prior。
- GPU 与存储：启动前核对物理 GPU0 UUID `GPU-d0c5535e-c4e1-8b72-e6f1-ae82f0b75771` 且无其它计算进程；进程内 cuda:0，EGL=0。产物逻辑目录 `logs/reproduction/conditional_shoulder80_prior_v1_expanded/`，实际位于 `/personal_fast_storage/smp_experiments/conditional_shoulder80_prior_v1_expanded_20261112/`，缓存和 W&B 本地文件同盘。完成 1 MiB fsync 写入探针及共享空间检查；个人配额工具不可用，不将共享剩余空间当成个人配额保证。
- 验证：72 项相关 CPU 回归通过，包括数据/标签/镜像/划分与精确采样、三/五条件及无条件兼容、NULL 区分、同噪声/时间掩码评分、恢复后下一步 loss/模型张量/采样一致。GPU 单 epoch smoke 完成 4 次优化更新，loss 0.9325713600，梯度、参数、DDPM 四分支及对比分数均有限；smoke 不计入正式预算，不以随机模型采样效果作为动作验收。
- 选模：每 1000 epochs 在固定 validation 主体上重新以训练集每条件 4096 个样本校准 μ，保留原 waist_yaw 10×、t=(8,15,22)、R_base×R_contrast 公式及门槛。64 组配对噪声，正确/镜像使用相同 x_t；正确动作、匹配静止与反向镜像动作同噪声比较。按每窗口 64 组噪声的平均最终奖励计算动作排序，按原始源文件 block bootstrap 报不确定性；静止偏置仍检查逐噪声 mean(abs(margin))。门槛通过的候选按 0.2/0.4/0.4 验证 L1 损失选择，不使用 test 选模；门槛失败不追加预算或放宽阈值。
- 保存与恢复：每 100 epochs 原子更新 `prior/latest.pt` 及 SHA sidecar；每 1000 epochs 保留独立 checkpoint。包含模型、优化器、Python/NumPy/Torch/CUDA RNG、sampler epoch、优化步数、归一化、schema、数据/代码身份和候选状态。进程使用独立 session 与驱动器排他锁；异常保留最后完整检查点，必须显式 `--resume`，不自动重置预算或抢占 GPU。
- W&B：`https://wandb.ai/qc13659041465-ustc/smp/runs/pwhlugzc`，仅配置和标量，无模型上传。驱动器 PID 142594，正式训练 PID 142808（启动时记录；后续恢复以 manifest 为准）。
- 启动阶段实测：正式第 1 epoch loss 0.93257，第 100 epoch 0.24185，第 375 epoch 0.18132；第 500 epoch 恢复 checkpoint 已存在，未见 NaN/Inf 或进程错误。该记录是启动验证快照，实时进度请读 `prior/run.json`、`prior/metrics.jsonl` 与 `prior.log`。
- 阶段边界：数据门槛和 smoke 已通过，正式训练进行中，不代表 Prior 动作或最终奖励排序已通过。训练器会保留完整预算产物；无通过候选则记录失败。有候选时锁定为 `candidate_locked_pending_test`，独立最终测试、1/3/5 帧过渡诊断、历史无转腰轨迹复查、旧 v2 对比与三类回放仍待训练后验收，不自动发布模型或启动 PPO。EXP-026/027/028、旧 Prior 和 PPO 产物不改写。
- 首次阶段验证补记：第 1000 epoch 的 checkpoint、独立校准、64 组配对评分 NPZ 和 report 已保存，验证加权 L1 loss 0.178068。静止偏好 64.34%、mean(abs(margin)) 0.197994 暂时通过原偏置门槛，但左右肩逐阶段 correct>stand 仅 76.24%–86.58%，未达到 95%；两侧 hold 的正确奖励 p05 约 0.01676/0.01749，也未达到 0.05，另有 margin 门槛失败。因此本候选未入选，不称为 Prior 验收通过。质量失败已写入 W&B/JSON，正式训练按原预算继续，随后确认已完成至少 1200 epochs 且进程正常；没有扩大预算或启动 PPO。
- 最终状态（UTC 2026-11-12 13:52:17）：10,000 epochs、40,000 次优化更新正常完成，末轮训练 L1 loss `0.095801`，无 NaN/Inf、GPU 或 checkpoint 故障。10 个千轮 checkpoint、校准与验证报告的 SHA-256 身份全部复核通过；总控状态为 `failed_no_score_passing_candidate`，`selected=null`。
- 正式验收结论：**未达到训练目标**。10 个候选均未通过固定的 28 项 validation Score gate，因此按预注册协议不使用 test 反复选模、不执行正式 subject 5 终测、不发布 `pretrained_lafan_shoulder80_conditional_v1.pt`，也不启动 PPO。训练完成只证明优化链路运行正常，不代表 Prior 行为合格。
- 最接近但仍失败的第 2000 epoch 候选通过 `17/28` 项：左右肩各阶段中，correct>reverse 的最低比例为 `95.30%`，说明镜像方向信息已部分学到；但 correct>stand 仅为 `73.27%–84.56%`，低于 `95%`，正确奖励最低 p05 为 `0.02384 < 0.05`，静止 `mean(|margin|)=0.20273 > 0.2`。第 3000 epoch 的加权 validation L1 loss 最低，为 `0.162646`，但同样仅通过 `17/28` 项，不能按 loss 单独选用。
- 过拟合证据：训练 loss 从第 3000 epoch 的 `0.13551` 继续下降到第 10000 epoch 的 `0.09580`，而 validation loss 从最低 `0.16265` 上升到 `0.20714`；通过项由 `17/28` 降至 `12/28`。第 10000 epoch 的各阶段 correct>stand 只剩 `25.50%–52.48%`，最低正确奖励 p05 约 `1.75e-10`，继续增加 epoch 没有改善目标排序。
- Score 分解诊断：早期 checkpoint 的 contrast 分支对明确转入/保持及镜像方向已有较强区分，但早期起转仍不稳定；完整 `R_base × R_contrast` 中，`R_base` 对静止动作过于宽容，并随训练后期对留出集正确动作显著降分。纯 contrast 对照只用于定位原因，未替代正式完整奖励门槛。
- 数据域诊断（非 gate）：第 2000 epoch 在 AMASS-G1 validation 上 correct>stand 约为左/右 `79.2%/80.8%`，仍未达到 95%；LAFAN 留出片段约为 `4.3%/0%`，显示明显跨来源泛化问题。G1 Moves 合格肩部片段只在训练划分，不能提供独立 validation 结论。训练 sampler 精确平衡条件与阶段，但不平衡来源，AMASS-G1 在肩部样本中占主导。
- 产物边界：全部失败候选、校准、配对评分和 W&B 标量保留在 `logs/reproduction/conditional_shoulder80_prior_v1_expanded/`；W&B run 为 `pwhlugzc`。失败 checkpoint 可用于离线诊断，但不得称为正式 Prior 或作为已经验收的 PPO prior。下一轮应优先修正训练目标与最终 Score 排序的一致性、来源平衡和早期起转区分，而不是简单追加相同训练轮数。
- 数据可视化复查（2026-11-13）：以 seed 42 对实际存储窗口做 train/test × 左/右肩 × early/strong/hold 的确定性分层抽查，共 12 条，并生成 FK 10 倍慢放、姿态总览和运动曲线。12 条均通过原数值筛选，但人工审阅发现 walk/fight/jump/dance、双臂平举及交谈手势等明显语义污染；没有一条在同一连续片段内完整展示“中立站立→对应方向起转→原地保持”。未发现明显特征列错位、镜像符号颠倒或 FK 解码错误，问题集中在以局部腰 yaw 数值代替肩推退让动作语义。产物位于 `data_visualization_seed42_v1/`，完整结论见 `docs/SHOULDER80_DATA_VISUAL_AUDIT_20261113.md`；本次不自动修改数据或重训模型。

## EXP-030：失败 Shoulder80 Prior 明确绕过后的肩部 PPO B1/B2

- 时间与目的（UTC）：2026-11-12 16:13 首次 smoke，16:29 双先验 smoke，16:30 首次正式 B1，16:41 从有效 checkpoint 恢复。目的：按用户“先直接进入 PPO 训练”的明确指示，从已通过无接触 A 的策略分叉，只训练左右肩 B1/B2，检验未通过正式验收但镜像方向信息较强的 Shoulder80 epoch-2000 候选能否在 PPO 中产生可用信号。
- 验收边界：reward Prior 为 `conditional_shoulder80_prior_v1_expanded/prior/checkpoint_02000.pt`，SHA-256 `73d9b0409066650525c3bb619f8440af947c2a96f3697f8774ed23d859c90cbe`；其正式 gate 仅通过 17/28，仍是失败候选，不称为已发布或已验收 Prior。本实验由用户显式覆盖阻断门槛，manifest 固化 `prior_gate_passed=false` 与 11 项失败检查。
- 双先验设计：首次 smoke 发现 Shoulder80 静止采样与旧 A 策略不兼容，76.6% 环境约 0.23 秒内触发关节硬限位。因此 GSI 单独改用 Conditional v2 的 stand 分布（SHA-256 `ade23a0bf027addbd3a210ae82ec3d979b2d38c6e16c978e459f930d1e004734`），仅 SMP reward 使用 Shoulder80；两套身份随 PPO checkpoint 保存并严格核验，GSI root 平面速度上限仍为 0.15 m/s。
- 父项与迁移：父 A `model_2999.pt` SHA-256 `b2b6dc8f4fae3a93ce0028f3a594bceb09494590bb4b1d4f29ee24355006c3fb`。新迁移 checkpoint 不覆盖父项，Actor、Critic、观测归一化器、optimizer、iteration 和环境计数逐张量保持；丢弃旧在线 SMP normalizer，写入 reward/GSI Prior、冻结校准及覆盖身份。
- 双先验 smoke：64 env、2 PPO iteration 数值有限，Actor/Critic 维度 128/992；安全 GSI 接受率约 95.9%。固定左肩的实际接触/分类为 100%/100%，固定右肩为 95.3%/95.3%；接触前 root 平面速度均值约 0.07 m/s。旧 A 在接触后仍有左/右 100%/98.4% 关节限位终止，作为 B1 初始质量问题记录，不解释为策略已会转腰。
- 首次正式中断：原肩部 score 对实际传感器检测到的意外胸/背接触直接抛出 `non-shoulder condition`，B1 在保存完整 `model_3000.pt` 后停止；无 NaN/Inf 或 GPU 故障。修复为：不支持的接触 ID 用 stand 分支完成有限推理、最终肩部奖励强制为 0，并记录 `smp_unexpected_condition`，防止把胸背响应冒充肩部正例。
- 恢复：`model_3000.pt` 通过完整性与双 Prior 身份检查，manifest 标记为经诊断的可恢复代码中断；剩余预算固定为 3999 iteration，目标 checkpoint 仍为 `model_6998.pt`。相关 score/recovery 回归测试 14/14 通过；全套本轮聚焦测试 23/23 通过，Ruff 通过。
- 正式运行：物理 GPU0 UUID `GPU-d0c5535e-c4e1-8b72-e6f1-ae82f0b75771`，4096 env、seed 42、std 0.10、lr 1e-4、每 1000 轮保存；B1 为 0.01 m/s、0.05 mg、ramp 1.0 s，B2 将从 B1 完整恢复并使用 0.03 m/s、0.10 mg、ramp 0.5 s。恢复 run 已越过原故障点且 `smp_unexpected_condition` 实际非零、无再次异常。
- W&B：首次中断 run `https://wandb.ai/qc13659041465-ustc/smp/runs/m35ie8xu`；恢复 run `https://wandb.ai/qc13659041465-ustc/smp/runs/6d56d0ng`。仅同步配置与标量，不上传模型。
- 完成状态：B1 于 2026-11-12 19:22 UTC 完成至 `model_6998.pt`，B2 从该完整 checkpoint 恢复并于 22:01 UTC 完成至 `model_10997.pt`；最终 checkpoint SHA-256 为 `a3eef9b89d57d5898c3c76c911c460c2f418ba02dcb023c9e4d76cc45173a8c7`。两阶段均无 NaN/Inf；质量门槛失败按预设策略记录，不再继续 C。B1 W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/6d56d0ng>；B2 W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/vy2gl15g>。
- 终点评估修正（2026-11-13）：发现旧遥测虽注册 `signed_waist_response_peak_rad` / `pelvis_yaw_abs_peak_rad`，控制器却未累计更新，历史峰值恒为 0；同时旧“given contact”方向率混入了被分类为胸口的接触。修复仅影响评估遥测，不改变训练策略或奖励；15 项条件接触测试与语法检查通过。最终报告只在“正确分类为目标肩”的 episode 内统计方向率和响应峰值，左右肩各重新评估 1,024 个确定性 episode。

| B2 正式指标 | 左肩 | 右肩 | 门槛 |
|---|---:|---:|---:|
| event success | 14.26% | 1.66% | ≥ 20% |
| 任意确认接触覆盖率 | 13.48% | 0.98% | ≥ 80% |
| 正确目标肩接触率 | 6.54%（67/1024） | 0%（0/1024） | ≥ 80% |
| 正确目标肩接触后的方向正确率 | 2.99% | 无样本 | ≥ 80% |
| 2 秒正确方向 waist-yaw peak p50 | 0 rad | 无样本 | ≥ 0.08 rad |
| 2 秒正确方向 waist-yaw peak p95 | 0.01993 rad | 无样本 | 仅诊断 |
| pelvis-yaw 绝对峰值 p95 | 0.12088 rad | 无样本 | ≤ 0.175 rad |
| emergency-force 率 | 0% | 0.098% | ≤ 20% |
| 非力学失败率 | 0.488% | 0.098% | 仅诊断 |
| 接触足滑速度 p95 | 1.576 m/s | 1.792 m/s | 仅诊断 |

- 终点结论：B2 左右肩均未通过 gate。左侧即使正确接触，腰部正确方向转动仍很弱；右侧没有一次正确分类的目标肩接触，首先暴露推头轨迹/肩部碰撞分区的严重左右不对称。SMP episode 聚合中的 contrast reward 接近 1 主要由无接触帧占比过高造成，不能解释为肩部条件奖励有效。当前证据不支持“策略已学会肩推后转腰”或“已学会可靠卸力”。
- 回放：按 B2、seed 42、固定左右肩分别生成 4-env 确定性 600 帧（50 Hz、12 s、1280×720）视频。该固定 replay 批次两侧均未形成确认接触，画面中机器人基本维持直立、没有可辨认的定向腰部退让，直观复现了低接触覆盖问题；该小批次仅作视觉核验，不替代 1,024-episode 正式统计。视频位于 `replay/b2_left_seed42/rl-video-step-0.mp4` 与 `replay/b2_right_seed42/rl-video-step-0.mp4`。
- 最终产物：修正报告为 `evaluation/conditional_shoulder80_override_b2_seed42_{left,right}_corrected_metrics.json`，回放对应的 4-episode JSON 为 `evaluation/replay_b2_{left,right}_seed42.json`。运行状态、恢复血缘、最终评估与回放身份记录在 `logs/reproduction/conditional_shoulder80_ppo_v1_override/manifest.json`；逻辑目录映射到 `/personal_fast_storage/smp_experiments/conditional_shoulder80_ppo_v1_override_20261112/`。失败 Prior 的验收结论保持不变。

## EXP-031：G1 肩部退让全身轨迹 IK 数据生产——先导门槛阻断

- 时间与目的（UTC）：2026-11-13 05:15–06:15。目的：建立不混入 LAFAN、G1 Moves 或 AMASS-G1 肩部正例的合成数据生产链，生成“中立站立→起转→明确转入→保持”的 G1 左右肩退让动作；本轮只允许完成数据生成与验收，不训练 Prior/PPO。
- 实现：新增 Mink 1.4.0 条件依赖和锁文件；新增确定性参数/配额生成器、浮动基座全身差分 IK、双足硬约束、COM/姿态/速度/碰撞限制、原 G1 位置执行器下的无外力 MuJoCo 回放、逐轨迹硬门槛、断点 manifest、CSV/NPZ/JSON 与 SHA-256。动力学回放使用 `0.005 s` timestep、`0.02 s` 控制周期、地面摩擦 0.6、无 root 锁定；训练候选只取实际执行轨迹，不取 IK 参考。
- 数据接口：实现实际轨迹到 `(N,10,59)` 的 stride-5 汇总器、阶段标签、逐窗口血缘和仅训练 split 统计；与仓库原转换器数值对照最大误差 `1.8e-7`。修复 `GVHMR_GMR_G1_SHOULDER_DATA_REQUIREMENTS.md` 缺失的第 8 节阶段门槛，并新增 `G1_SHOULDER_IK_DATA_PRODUCTION.md`（当前 SHA-256 `a4d9fbf41a241baacc5b4695596117b7765118c2d2a7f9f031cecc93c359fd47`）。
- 测试：聚焦测试 10/10 通过，覆盖最小加加速度、精确配额、左右 family/split、确定性 retry、NumPy 指标 JSON 回归、阶段分类、关节/xyzw/59 维契约，以及此前最困难的 `0.26 rad / 0.5 s` 浮动基座动力学轨迹；Ruff 通过。
- 恢复历史：首次先导在 165 个 slot 时发现肩部专属 `numpy.bool_` 无法写入 JSON，保留为 `pilot_attempt01_serialization_error/`；修复后完整尝试发现最快起转跟踪滞后，保留为 `pilot_attempt02_tracker_lag/`。随后只调整高层 waist-yaw 跟踪并加入 `0.002 rad` 的视觉中立预稳定，未改变原执行器增益、参数网格或验收阈值；当前源码哈希与最终 manifest 逐项一致。
- 最终先导结果：要求 240 条，接受 228 条、失败 12 条，状态 `blocked`；48 条 stand 全通过，肩部 180/192 条通过，96 个参数单元中 90 个得到完整左右对。所有 228 条接受样本均为 attempt 0；失败的 12 个 slot 在两次确定性候选中均失败，未用重复抽样补数。
- 阻断项：12 个失败全部属于 `yaw=0.26 rad、entry=0.7 s` 的 6 个站距×膝屈曲单元及其左右侧；24 个失败 attempt 唯一失败检查均为 `phase_windows`，实际和 IK 参考都只有 1 个 early 窗口，而门槛要求至少 2 个。参考轨迹的代表性计数为 `[stand,early,strong,hold]=[9,1,5,17]`，实际执行代表性计数为 `[10,1,6,15]`。继续按窗口阈值改变控制相位会变成针对筛选器造数据，因此未放宽 early 门槛、未更改 stride、未替换参数单元。
- 其余物理证据：接受集左右脚接触覆盖率均为 100%；root 平面位移最大 `0.001765 m`，足滑最大约 `4.10e-5 m`，全关节跟踪 RMSE 最大 `0.00595 rad`，waist RMSE 最大 `0.03180 rad`，力矩 p99 最大占上限 `18.68%`，饱和比例为 0，保持段 root 速度 p95 最大 `0.000414 m/s`、waist 速度 p95 最大 `0.01689 rad/s`。阻断是阶段覆盖，不是跌倒、接触、限位、力矩或数值故障。
- 产物：逻辑目录 `logs/reproduction/g1_shoulder_ik_v1/` 映射至 `/personal_fast_storage/smp_experiments/g1_shoulder_ik_v1_20261113/`，约 130 MiB。最终先导 manifest SHA-256 为 `a43b813c86ca9d72e9173c981d3fd20d15594c5d75fec67d84324442203b039d`；`gate_report.json` SHA-256 为 `53bf9e7f69caead0c2be9b444b6c54da78b5c5624db781d2bb800928081bd97f`，固定全部失败 slot、两次候选、参考/实际阶段计数和接受指标分位数。
- 决策：严格执行先导硬停止。未把 228 条部分集发布为可训练数据，未生成 `dataset/manifest.json` 或归一化统计，未把自动门槛失败伪装成视觉验收；因此没有进行 60 条正式人工回放检查、没有启动 2400 条正式集，也没有启动 Prior/PPO。若后续继续，必须先由新计划明确解决“纯最小加加速度离散采样与至少 2 个 early 窗口”的规格矛盾，再建立新的数据身份。

- 终点任务重验收（UTC 2026-11-13 06:54）：用户确认转腰幅度和速度不作为固定验收量，动作改为整条轨迹的任务完成判定。目标位置定义为目标肩相对 pelvis、表达在逐帧 pelvis heading 坐标系中的局部 `x` 后移；规定后移 `0.015 m`，到位允许 `0.001 m` 容差，必须在各 clip 自己的 `0.5/0.7/0.9/1.1 s` 截止时间内首次进入目标区。终点仍须实际达到 `0.015 m`；最后 1 秒后移始终不少于 `0.014 m`，肩部三维局部位置相对保持段中位位置的最大偏差不超过 `0.002 m`。原 root、pelvis、双足接触/足滑、限位、碰撞、跟踪、力矩和速度门槛全部保留；取消 `early/strong/hold` 窗口数量、固定 waist-yaw 幅度和固定速度曲线 gate。
- 重验收方法：不改写原 blocked `manifest.json` 或逐轨迹 metadata。新脚本逐文件复核 execution/reference/CSV SHA-256，复用原冻结物理指标，并从实际执行 qpos 重新计算 pelvis-heading 肩部终点、首次到位时间和保持偏差；结果写入派生 `manifest.endpoint_v2.json` 和 `endpoint_revalidation_v2.json`。旧 manifest SHA-256 仍为 `a43b813c86ca9d72e9173c981d3fd20d15594c5d75fec67d84324442203b039d`。
- 重验收结果：**自动门槛 240/240 通过，失败 0**；全部选择 attempt 0。此前仅因阶段窗口失败的 12 个 slot 在轨迹不变、物理门槛不变的情况下通过终点任务验收。左/右肩首次到位时间最大均为 `0.86 s`，相对各自截止时间的最小裕量均为 `0.02 s`；截止时后移最小为 `0.014812/0.014945 m`，终点后移最小为 `0.018885/0.018910 m`，最后 1 秒后移最小为 `0.018826/0.018879 m`。保持段肩部位置最大偏差全局最大 `0.000378 m`，显著低于 `0.002 m`；stand 终点 waist yaw 最大绝对值 `0.000915 rad`。全 240 条的左右脚接触覆盖率均为 100%，root 位移最大 `0.001849 m`、pelvis yaw 变化最大 `0.000518 rad`、足滑最大 `4.10e-5 m`、全关节/waist 跟踪 RMSE 最大 `0.005949/0.031802 rad`、力矩 p99 最大为上限的 `18.68%`、饱和率为 0、电机速度比最大 `2.28%`；保持段 root/waist 速度 p95 最大 `0.000434 m/s` / `0.016890 rad/s`。
- 实现与测试：生成器未来运行改用相同终点/时间/保持 gate，修复 stand waist 检查此前因乘零符号而恒真的问题；59 维汇总器只保留 `stand/left/right` 条件，跨条件边界窗口剔除，不再输出阶段标签。聚焦测试 `9/9` 通过，并实际覆盖原阻断组合 `yaw=0.26 rad、entry=0.7 s`；Ruff 通过。生产规范升级为 v1.1，SHA-256 `4f8c6179ebcafdc5ea1e10c3b438dea742b524e78a925803cc16dffb1fe033cb`；GMR 输出要求 SHA-256 `26917033e45ada7d9692f5416d46e7abb356c76c944c1bc835aaddcf3bc4a254`。
- 新产物：派生 manifest SHA-256 `78876ecff48ff43b38062edbcc73c95a6016ad496b9b4a9a235ef82bb17a3392`；重验收报告 SHA-256 `7f24d551f7378c22efce626b2bddb9efda8460ebf0d6da6c7d4c0c0f7c0c9cd7`。该结论仅表示 pilot 自动终点门槛通过；60 条分层人工回放尚未执行/签署，正式 2400 条生成、Prior 和 PPO 仍未启动。
- 10 条分层 replay（UTC 2026-11-13 07:54）：从通过终点重验收的派生 manifest（SHA-256 `78876ecff48ff43b38062edbcc73c95a6016ad496b9b4a9a235ef82bb17a3392`）确定性抽取 stand/left/right=`2/4/4`，肩部样本覆盖参数网格首、中、末区域。使用 EGL 以 50 Hz 输出正面＋侧面拼接视频，单视频 `960×360`、150–182 帧、3.00–3.64 s；10 个 MP4 均由 imageio-ffmpeg 完整解码且数值有限。产物在 `pilot/replay_sample_10/`，包含 10 个视频、逐条首/中/末关键帧、2 张 contact sheet、`selection.json` 和 `preview.json`；selection SHA-256 为 `358f15e78a30521e0cd6be672f6c48419efd24a9e29b4255c6db47cec0614190`。该抽查显式标记 `formal_visual_gate=false`，不替代正式 60 条人工视觉验收。
- 60° 左肩幅度诊断（UTC 2026-11-13 08:07）：按用户要求在 pilot/formal 配额之外独立生成 `π/3≈1.047 rad` 左肩退让轨迹，不混入已验收 240 条数据。首个 2 s 转入候选实际达到 `59.9965°`，其余门槛均通过，但 waist 跟踪 RMSE `0.05659 rad > 0.04`，因此保留为 rejected 且不用于回放结论。第二个候选保持相同终点，改为 1 s 站立＋4 s 转入＋2 s 保持；实际终点 `59.9965°`、waist 跟踪 RMSE `0.03423 rad`、肩部后移 `0.12273 m`，全部物理门槛通过。合格 manifest SHA-256 为 `71f707ddf335f9dcf43eb2196caf205dc3d001f4b80cc4fd4f3895f71f5fb2e0`。双视角 replay 为 50 Hz、350 帧、7 s、`960×360`，完整解码验证通过，视频 SHA-256 `8097aaaea5d43f76b30338109074ada590a0e05cbcf927f90711e278afb9be63`，位于 `diagnostic_left_60deg/replay/left_shoulder_60deg.mp4`。该轨迹仅用于幅度视觉诊断；是否把约 60° 样本纳入正式数据分布需另行决定。

## EXP-032：G1 连续随机转腰幅度正式数据集 v2

- 时间（UTC）：边界 smoke、pilot 与正式生成于 2026-11-13 进行；pilot 为 09:11:02–09:11:40，formal 为 09:22:43–09:28:47；完整性与视觉终验于 2026-11-14 完成。
- 目的：新建不覆盖 v1 的 `g1_shoulder_ik_v2`，生成 2400 条 `stand/left/right=480/960/960` 的纯净 G1 全身 IK＋无外力动力学轨迹。肩部目标幅度不再使用离散网格，而在 `15°–70°` 连续随机覆盖；本实验不训练 Prior 或 PPO。
- 抽样：seed 42；六种站姿各含 160 个左右配对 family。每个 `(stage, split, 站姿)` 独立执行 randomized Latin-hypercube sampling；平均转腰速度独立采样于 `12–18°/s`，与幅度随机解耦，完成帧数严格为 `ceil(幅度/速度×50)`。左右共享幅度、速度、完成时间与 split，但分别求解和回放，未复制或镜像关节数组。
- 配额：每个站姿的正式配额为 train/validation/test=`112/24/24` 对；最终 split 为 train `336/672/672`、validation `72/144/144`、test `72/144/144`，无 family 泄漏。stand 轨迹的目标幅度、目标速度和转入时间均为 0。
- 动力学：沿用 G1 完整碰撞和位置执行器、`0.005 s` 物理步长、`0.02 s` 控制周期、50 Hz 输出、摩擦 0.6、双脚固定和 free root。正式冻结 waist-yaw 跟踪增益为 180，原 G1 执行器参数不变。
- 边界 smoke：左右肩 `15°/42.5°/70°` 共六条在冻结配置下全部通过。精确 15° 候选仅因浮点跟踪低 `0.00040°` 触发严格比较，范围比较加入 `0.001°` 数值容差；动力学门槛未放宽。
- Pilot：重新生成的 240 条为 `48/96/96`，240/240 自动通过；96 个肩部 family 均有左右样本。实际幅度左肩 `15.2338/42.3803/69.9332°`（min/p50/max），右肩 `15.2337/42.3802/69.9345°`；11 个 5° 区间全部非空。60 条分层回放按 `stand/left/right=12/24/24` 审查，每侧四个幅度区间各 6 条；60 个 MP4 均以 50 Hz 完整解码，未见方向错误、迈步、整体转身或明显抖动。pilot manifest SHA-256 为 `654107ef26be64ccebf3fd290f547c882facf3980c9363de805e58c6382fd59f`。
- 正式结果：2400/2400 条合格，最终失败 slot 为 0。正式 manifest SHA-256 为 `a2ed23e37ac772c4a1379f85f54bbf9a764320264580a33bc23e684be6a70d2a`，gate report SHA-256 为 `123898d3b802e2aa47722d089cb92ff888978b1880403694492416c94b7fcba9`。
- Retry：仅 test 站姿 c02 的同一约 `69.871°`、`17.970°/s` 左右配对在 attempt 0 因 waist RMSE `0.040128 rad` 略超 `0.04` 被拒；attempt 1 保持幅度、速度、完成时间和 split 不变，只改变允许的 arm/COM/保持变体，RMSE 降至约 `0.03932 rad` 后通过。拒绝尝试和接受尝试均保留，未换更容易的角度补位。
- 幅度覆盖：train 左/右实际范围约为 `15.011–69.991/69.992°`，validation 为 `15.073–69.678°`，test 为 `15.110–69.868/69.870°`；六个 side/split 组均覆盖全部 11 个 5° 区间，且满足 min≤16°、max≥69°。train/validation/test 的幅度—速度相关系数绝对值均不超过约 `0.057`，符合独立采样预期。
- 物理结果：2400 条左右脚接触覆盖率均为 100%；root 位移最大 `0.00857 m`，足滑最大约 `3.57e-5 m`，全关节 RMSE 最大 `0.00738 rad`，waist RMSE 最大 `0.03950 rad`，终点幅度误差最大 `4.54e-5 rad`。力矩 p99 最大为上限的 `18.684%`、饱和率 0、电机速度比最大 `1.811%`；保持段 root/waist 速度 p95 最大 `0.000269 m/s` / `0.00908 rad/s`。
- 训练数据：实际执行轨迹转换为 50 Hz、stride 5 的 `(N,10,59)`。train/validation/test 窗口数为 `80178/17162/17108`；train 条件计数为 `21840/29169/29169`。归一化 `q01/q99` 仅由 train 重算，独立复核最大误差为 0。每个窗口记录 clip ID、目标/实际幅度和目标速度；`clip_balanced_weight` 对每个源 clip 求和为 1，避免长时大幅轨迹按窗口数量主导后续 Prior。
- 窗口标签说明：肩部 clip 的初始中立窗口被标为 condition 0，但其幅度/速度字段记录源 clip 后续肩部目标；480 条独立 stand clip 的目标幅度、速度和完成时间已逐条确认均为 0。这不是 stand 轨迹被设置了转腰目标。
- 正式视觉终验：60 条按 train/validation/test 各 20 条抽取；每个 split 为 4 条 stand、左右肩各 8 条，并且每侧四个幅度区间各 2 条。全部视频完整解码；总览、高幅样本和两条约 `69.87°` retry 的完整时序均通过检查，无方向错误、迈步、骨盆代偿转身或明显抖动。
- 完整性：全量复核 2400 条接受轨迹和 2 次拒绝尝试的 7206 个 reference NPZ、execution NPZ 与 CSV 哈希，错配为 0；960 个左右 family 高层身份一致；正式数据含 960 个连续唯一目标幅度。`integrity_report.json` 最终 `passed=true`。
- 测试：Ruff check、Ruff format check、Python 编译检查均通过；v2 聚焦测试 `15/15` 通过，覆盖 seed 确定性、不同 seed 差异、连续性、Latin-hypercube 分层、左右配对、split 配额、50 Hz 对齐、retry 身份、恢复一致性、边界 smoke 和回放分层选择。
- 产物：逻辑目录 `logs/reproduction/g1_shoulder_ik_v2/` 映射至 `/personal_fast_storage/smp_experiments/g1_shoulder_ik_v2_20261113/`；包含独立边界 smoke、失败诊断、pilot、formal、CSV/NPZ、train-only 统计、分布图、报告、60 条 pilot 回放和 60 条 formal 回放。v1 pilot、60° 诊断和 EXP-031 全部保持不变。
- 发布校验：`checksums.sha256` 覆盖 9754 个正式产物且 `sha256sum -c` 全部通过，清单 SHA-256 为 `0e8cf2d2a1f49e914b6ef64fa50f2040d8248191546a3f213204209cb933caa0`；`release_manifest.json` SHA-256 为 `fd31b04d0f7d149b71b8befb104c4a395863de1f4ac09f284c18328bdc332a6a`。
- 全实验校验：顶层 `checksums.sha256` 进一步覆盖 boundary smoke、保留诊断、pilot、formal、报告和回放共 11921 个文件，`sha256sum -c` 全部通过；清单 SHA-256 为 `6f1cae96c4c20398e180076d3b0ed627eb1b38921421d29c3326987ff3ba6475`，顶层 `release_manifest.json` 明确记录 `v1_overwritten=false`、`prior_or_ppo_started=false`。

## EXP-033：G1 IK v2 三条件 Prior 与肩部专项 PPO

- 启动时间与目的（UTC）：2026-11-14 07:14。仅使用 EXP-032 已验收的 2400 条纯 IK v2 轨迹，从零训练 `stand/left_shoulder/right_shoulder` 三条件 Conditional Prior；Prior 通过冻结 Score 与盲测后，才允许从零执行 `A→B1→B2` 肩部专项 PPO。不混入 LAFAN、G1 Moves、AMASS-G1，不复用旧 Prior 或 A checkpoint，不训练 C。
- 数据与方法：数据 manifest SHA-256 为 `d9ad8ef6d8d8b9680709d113556a5efd50ae4812c1138b6fb832dd20f7ec486e`，train/validation/test 窗口数为 `80178/17162/17108`。每 epoch 严格抽样 `672/1344/1344`，先 clip 后窗口，肩部按 11 个 5° 区间分层。Prior 为随机初始化 `d_model=128`、4 heads、2 layers、50 DDPM steps，batch 1024，10 组噪声，AdamW `3e-4`，10% condition dropout，无 EMA，预算 10000 epochs。
- Score 与停止条件：每 1000 epochs 用训练集重新校准，`t=(8,15,22)`，waist-yaw 特征 21 权重 10，correct/opposite 共用噪声，分数为 `R_base×R_contrast`。只在 validation 整体门槛通过的候选中按 clip-balanced `0.2/0.4/0.4` loss 选模，test 仅用一次；无候选或 test 失败时保留产物并禁止 PPO。
- 预检：指定物理 GPU0 UUID `GPU-eb0fc39f-6914-9e1d-65af-570524e8e061`，A100-SXM4-80GB，启动时空闲；W&B 已鉴权，不记录 key。新实现与历史兼容回归 114/114 通过，Ruff 和语法检查通过。单 epoch smoke 完成 4 次更新，loss `0.9329274081`，梯度、三条件/NULL DDPM 和对比 Score 全部有限，实际抽样为 `672/1344/1344`；smoke 权重不进入正式预算。
- 存储与当前状态：逻辑目录 `logs/reproduction/g1_shoulder_ik_v2_prior_ppo_v1/` 映射至 `/personal_fast_storage/smp_experiments/g1_shoulder_ik_v2_prior_ppo_v1_20261114T071401Z/`，已通过 fsync 写入探针，启动时可用空间约 413.9 TB。当前仅 smoke 通过；正式 Prior、盲测、发布与 PPO 结果将续记本条目。
- Prior 训练与发布（UTC 08:04）：完整训练 10000 epochs / 40000 optimizer steps，无恢复、NaN/Inf 或 GPU 故障。epoch 4000/5000/6000/9000 通过 validation 全部 Score 门槛；按通过候选的最低 clip-balanced loss 锁定 epoch 9000（loss `0.06411257`），checkpoint SHA-256 `d9e50f10db13bb720a79e4992b79f6b5119aa327066c6d1cbec8929ce01542a2`。Prior W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/92s0q7qj>，未上传模型。
- Prior test：锁模后首次使用 test split，最终 Score 门槛通过；每条件 4096 个生成窗口全部有限，stand 窗口最大平面速度 p95 为 `2.88484e-5 m/s`，左/右肩方向率为 `81.8848%/82.7637%`，GSI 安全接受率 `100%`。Prior 已发布为 `datasets/pretrain_ckpt/pretrained_g1_shoulder_ik_v2_conditional_v1.pt`；冻结校准 SHA-256 `978ebaa9e2ca9fbd19d6e9b2822f3c3b18d0d31f29a51e73d2a35646475d7d34`，PPO 因此获得授权。
- PPO 当前状态：64-env、2-iteration scratch smoke 通过，`resume=false`，Actor/Critic 输入已硬校验为 `128/992`，checkpoint 和更新有限。A 已从随机 Actor/Critic/normalizer/optimizer 启动，W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/ymippsm8>。诊断显示纯 IK 数据若干特征归一化范围极窄，导致仿真轨迹的 normalized energy 约 `1e9`、`R_base` 数值下溢为 0，初期平均训练奖励为 0。该现象不属于既定 NaN/Inf 等硬停项，未擅自修改 Score；将由 A 的 1024-episode 正式门槛判定是否追加 2000 轮及后续是否停止。

## EXP-034：个人 GVHMR/GMR 动作二次受约束重定向

- 时间与目的（北京时间）：2026-11-15。将用户采集并经 GVHMR/GMR 得到的 27 条 G1 轨迹重新处理为可用于后续混合 Prior 的物理合格动作；本轮只生产与验收数据，不训练 Prior/PPO。
- 用户确认：双脚原地六自由度固定；允许腰、髋和骨盆优化分配转角；不设置额外 pelvis yaw 上限或固定 waist 幅度。全部 27 条处理后，每名参与者的 stand/left/right 至少各 1 条合格才允许发布可训练子集。
- 输入身份：`datasets/G1-GVHMR-GMR/manifest.csv` SHA-256 `b284fb32ce05f5d8c5f8c1cbb6754c84daeec81037ddbfc53197971c0e07831a`；P01/P02/P03 每名各 3 条 stand、左肩、右肩。固定按主体划分 train/validation/test，源视频与 NPZ/CSV/元数据逐项校验。
- 方法：当前 SMP G1 模型、Mink 1.4.0；原动作作为软参考，双足位姿与内缩 1cm 的 COM 支撑凸包作为硬约束，完整非相邻碰撞几何含手腕/手掌与髋部；每 50Hz 输出帧执行 4 个 0.005s IK 子步。使用原执行器、自由 root、摩擦 0.6、无外力回放，预稳定与额外末帧保持各 1 秒均不进入训练数据。MuJoCo 警告、自动重置与非有限状态均拒绝候选。
- 验收变化：肩部标记固定于躯干，使用初始 heading 计算世界及扣除 pelvis 平移后的退让量，两者终点均 ≥0.015m；最后1秒保持偏差 ≤0.005m，root 平面位移 ≤0.10m。仍保留接触、足滑、关节/电机、碰撞、姿态与跟踪误差门槛；转角只受关节和物理约束，不把骨盆协同运动冒称纯相对转腰。
- 产物与流程：独立目录 `logs/reproduction/g1_personal_retarget_v1/` 映射到持久盘时间戳目录；记录最多6个确定性站姿候选与拒绝原因，冻结代码/数据/配置身份并支持恢复。先9条 smoke，再处理27条；OSMesa CPU 生成四视图原始/执行对照回放、角度曲线和关键帧。只有具名视觉复核与全主体条件覆盖均通过后才导出实际执行的未归一化 `(N,10,59)`。
- 归一化边界：仅输出 train-only 诊断统计与窄范围报告，不替换任何现有 Prior 统计。后续混合 Prior 的配比、统一归一化、校准和在线 Score 验收另行开展。
- 当前状态：流水线与聚焦测试已实现；首批 P01 stand/左右肩的 CPU IK＋动力学候选均通过。九条 smoke、全量处理、回放复核和最终接受数量续记本条。
- 首轮 smoke 诊断：8/9 源 clip 通过，P02 右肩 take001 的6种站姿均因 `hold_pelvis_speed` 和 `hold_shoulder_stability` 被拒；骨盆保持速度约0.133rad/s、肩部保持偏差约21mm，无碰撞。复查源曲线发现最后约0.5秒开始回中，源骨盆速度p95约0.223rad/s。该轮遵守smoke门槛停止，未启动full。补充统一的末段稳定软目标：从源末2秒中的前1秒取中位姿态，五次过渡后保持最后1秒；不改变源文件、实际状态、轨迹时长或验收阈值。后续重新建立配置与代码身份，保留首轮全部拒绝证据。
- 冻结配置与全量物理结果（北京时间02:07起）：修正软目标后的9/9 smoke均于attempt0通过，随后处理全部27条，共32个候选，26条接受（stand/left/right=9/9/8）、1条最终拒绝。P01/P02/P03的条件计数分别为`3/3/3`、`3/3/2`、`3/3/3`，主体条件覆盖门槛通过。唯一拒绝为`P02_session01_right_shoulder_take003`，六种站姿均只因root倾角`20.4818°–20.4929°>20°`失败，未放宽门槛或替换源视频补数。
- 物理指标：接受样本左右脚接触率均为100%，无自碰撞或非脚部触地；最大足部漂移`0.000635834m`，最大root平移`0.0355741m`；全关节/waist跟踪RMSE最大`0.0129491/0.0178128rad`。力矩p99最大为上限的`18.7655%`，饱和帧率为0。左肩世界后退`0.08843–0.13044m`，右肩`0.11668–0.14803m`；保持位置偏差最大分别`0.0000584/0.0002307m`。该结果允许明显骨盆转动，按初始heading评估，不沿用EXP-032的骨盆0.05rad门槛。
- 恢复与完整性：真实执行一次full恢复，27条记录、32个候选与所有已完成metadata逐项一致，没有重新生成或重置尝试预算。独立审计核验160份候选metadata/源50Hz/参考/实际/CSV产物身份，并保存冻结实现快照。当前持久目录为`/personal_fast_storage/smp_experiments/g1_personal_retarget_v1_20261114T180721Z/`；首轮失败smoke保留为`logs/reproduction/g1_personal_retarget_v1_smoke_unsettled_20261115/`。
- 回放与复核完成：26个原GMR正面/侧面＋实际执行正面/侧面的完整网格视频，50fps、共17586帧（单条631–856帧），均完成全视频解码核验；全部26条的多时点关键帧由agent逐条复核，结合全帧物理指标未见方向错误、迈步或保持失稳。具名记录在`replay/review.json`，明确方法是“agent关键帧复核＋全轨迹指标”，不冒称人工逐帧观片。CPU渲染曾调整并行度与Mesa线程并恢复部分任务，最终完整结束，没有重算物理轨迹或占用训练GPU。
- 独立重验收：从已保存实际NPZ逐帧重建FK与原生MuJoCo接触，26/26通过；重新计算的足部漂移与生产指标最大差异仅`6.582e-8m`。重新求解P01左肩take001的参考轨迹，root/四元数/29关节三组float32数组与已保存版本逐元素完全一致。证据分别为`kinematic_reaudit.json`与`reference_determinism_audit.json`。
- 数据发布：26条实际执行轨迹导出3451个未归一化`(10,59)`窗口，train/validation/test分别1131/1095/1225；条件计数依次为`628/252/251`、`603/293/199`、`632/276/317`。clip-balanced权重逐clip求和为1，跨条件边界剔除，主体划分无泄漏。train-only诊断的59维数值跨度均不低于`1e-3`（最小约`0.001392804`），但该诊断不代表已经通过在线SMP验收，未替换正式归一化。
- 最终身份与验证：实验identity SHA-256 `0df1c7eed37956c2276564e2f97100630ed7c780c169a4cb3b154cf3677d73bb`；数据manifest SHA-256 `616964e09b235db14e69a68db68d0d2224080020fbc78f9c514cb9f1c08157bf`。发布清单`release_manifest.json`为complete，`checksums.sha256`覆盖347份产物且逐项复核通过，清单SHA-256 `dc0f4c9de4be11e23f9bff01eb07950e597d7c926b3b4553f36554432907cfcb`。非缓存产物约53.2MB；29/29项新旧回归、Ruff、编译和补丁空白检查通过。完整结果见`ASSESSMENT.md`、`assessment.json`与`accepted_metrics.csv`；本轮未训练Prior/PPO、未覆盖原个人数据或历史模型。

## EXP-035：多来源肩部 Conditional Prior 与 PPO

- 时间与目的（UTC）：2026-11-15。按用户确认的配比混合纯 IK v2、个人二次受约束重定向和 LAFAN/G1 Moves/AMASS-G1 静止数据，修复纯 IK 窄归一化导致的在线 SMP 零奖励，再从零训练肩部 A→B1→B2。
- 用户选择：左右肩条件中 IK/个人各50%；stand中 IK/个人/旧库为25%/25%/50%；总体20%stand、40%left、40%right；旧库不加肩部正例。保留原乘积公式并增加在线门槛，不使用奖励保底，不训练C。
- 数据：训练/验证/测试分别84969/20155/18815个未归一化(10,59)窗口；条件数为26128/29421/29420、7181/6534/6440、5794/6490/6531。旧库6040个静止窗口通过当前完整碰撞FK复核，无精确跨划分重复；个人来源仅26条合格轨迹，不把重采样当新来源。原主体/参数族划分保持。
- 方法：统一train-only来源/clip/幅度加权q01/q99与物理量最小跨度；新Prior随机初始化10000epochs，三条件NULL=3；重新校准V3 Score。离线、生成/GSI、实际仿真在线评分全部通过才允许PPO。A3000、失败扩展2000，扩展仍失败停止；B1/B2各4000。腰/骨盆协同允许，按肩部世界及root补偿退让验收；自碰撞/关节限位纳入失败并集。
- 固定GPU：物理GPU0，UUID`GPU-eb0fc39f-6914-9e1d-65af-570524e8e061`；产物位于持久目录`/personal_fast_storage/smp_experiments/g1_shoulder_mixed_prior_ppo_v1_20261115T022658Z/`，逻辑入口`logs/reproduction/g1_shoulder_mixed_prior_ppo_v1/`。W&B配置/标量，不上传模型；历史模型与EXP-033/034保持原样。
- 状态：数据准备完成，新增实现/回归测试及独立Prior smoke进行中。正式训练和阶段验收结果由驱动器续记；详见`docs/G1_SHOULDER_MIXED_PRIOR_PPO_V1.md`。


- 预检完成与启动（UTC 2026-11-15T03:07:42.972149+00:00）：72/72相关回归通过；独立Prior smoke完成4次有限更新、三条件及NULL DDPM/Score有限；实际数据配额与采样恢复精确。旧EXP-033在线负向回归被拒绝（参考与scratch均100%零分）；PPO实测恢复73个学习张量完全一致，下一轮计数与学习率恢复正确。正式独立驱动PID=282713，开始10000epochs新Prior；数据manifest SHA-256 `3504b7222e50e08f15486b52aa77c000d47cd7b618c83e4becea99684af9a801`。首轮schema不兼容smoke已归档修正，不计入正式预算；回归策略不用于正式PPO。

- 正式Prior进度（UTC 2026-11-15T03:14:06.706250+00:00）：已完成1975/10000epochs、7900次更新，训练进程与独立驱动均存活，无NaN/Inf或进程错误。W&B：https://wandb.ai/qc13659041465-ustc/smp/runs/x11zfafi。首个1000epoch候选完成重新校准和验证，weighted validation loss=0.170413425；部分最终奖励排序及p05门槛未通过，继续既定预算，未放宽门槛、未启动PPO。当前状态以`prior/run.json`为准。

- EXP-035 驱动状态更新（UTC 2026-11-15T03:45:28.549023+00:00）：status=blocked_prior_validation；Prior实际10000/10000 epochs，W&B=https://wandb.ai/qc13659041465-ustc/smp/runs/x11zfafi。

- 训练后评估（UTC 2026-11-15T04:34:24.325431+00:00）：Prior正常跑满10000epochs/40000次更新，十个千轮候选均未通过验证，状态blocked_prior_validation；未发布新Prior、未使用test split、未启动PPO。
- 最终Score（按来源与clip均衡）：正确>静止左/右67.293%/60.167%（门槛95%）；正确>反向100%/93.262%；正确条件>反向条件95.366%/98.167%；正确动作clip分数p05为0.0012927/0.0000094006（门槛0.05）。静止左肩偏好13.403%，mean|margin|=0.09139；绝对margin门槛通过，偏好比例未通过。
- 来源差异：最终IK左/右正确>静止98.239%/98.117%，个人P02验证主体仅36.347%/22.217%。个人训练仅P01三条左肩、三条右肩，P02验证三条左肩、两条右肩；不把窗口和噪声重复当独立主体。
- 失败项定位：个人验证动作的contrast项大多接近1，但base项很低。例如P02右肩take001的片段均值contrast≈0.9831、base≈1.162e-5、最终reward≈9.401e-6。该片段正常退让动作仍拿不到有效高分；两项均值的乘积不等同于逐窗口/噪声乘积的均值。
- 过拟合迹象：验证loss最低为epoch2000的0.158028，epoch10000上升至0.188574；最后1000epochs训练loss均值约0.042674。epoch2000仅为诊断参考，没有作为合格Prior选中或发布。
- 补充只读采样诊断：epoch2000/10000各生成三类、每类1024窗口，同初始及后验噪声，全部有限。最终左/右窗末有符号waist yaw≥0.05rad的比例为96.582031%/93.847656%，静止窗口最大平面速度p95=0.011518m/s。该诊断说明条件姿态分布已学到，不等同正式4096样本验收、GSI安全率或接触响应成功。
- 结论：本轮主要瓶颈在个人留出动作的绝对适配评分及跨主体泛化；现有证据支持过拟合与校准尺度收紧的解释，不能据此断言源数据重定向错误。训练未追加、门槛未放宽、PPO未绕过启动。完整数字、CSV、采样NPZ与两张图保存于`logs/reproduction/g1_shoulder_mixed_prior_ppo_v1/assessment/`。

- 故障链定位（UTC 2026-11-15T09:01:54.491803+00:00）：只读抽样每组256窗口、64组配对噪声、t=(8,15,22)，不访问测试集。训练P01的correct归一化能量均值左/右约0.596/0.578；验证P02约37.129/57.621，IK验证约1.438/1.465。P02左肩误差贡献为root旋转26.38%、腿部30.38%、waist roll/pitch11.87%，waist yaw仅0.64%；右肩root旋转51.89%，waist yaw仅0.89%。说明低分主要由全身姿态适配而非腰yaw误差驱动。
- 个人站姿差异（肩部窗口统计）：训练P01左/右root pitch中位数约-4.66°/-8.45°，验证P02约-11.18°/-17.03°；另有髋关节及waist roll差异。P02规范化绝对值总体p95约1.13/1.59，没有发现上次那种大范围输入数值爆炸。当前证据指向跨主体全身姿态分布差异、少量个人clip反复抽样的过拟合迹象，以及训练均值校准后的指数base压分；尚不能据此判定原视频/GVHMR/GMR存在格式或语义错误。细项保存于assessment/feature_error_attribution.json与personal_pose_comparison.json。未修改模型、配比、门槛或启动PPO。

- 修复方向诊断（UTC 2026-11-15T09:14:22.012859+00:00）：在已有验证原始分数上，冻结epoch2000/10000权重，仅诊断性移除base，按原来源/clip权重重算排序。P02左右肩correct>stand均为100%；epoch2000综合IK后为98.299%/94.851%，epoch10000为97.258%/94.287%。说明条件对比已提供方向信号，base是主要抑制项；右肩整体仍未达95%，且该诊断没有验证动作质量/安全性，不能当成正式新Score验收。新公式未部署、原门槛与模型均未修改、PPO未启动。结果保存在assessment/contrast_only_diagnostic.json。


## EXP-036：P01/P02 合并覆盖、完整片段留出的混合 Conditional Prior

- 时间与目的（UTC）：2026-11-15T09:49:39.742029+00:00。按用户授权，将17条合格P01/P02轨迹合并建池，大部分用于训练，再次从零训练Prior；本轮只训练Prior，不启动PPO，也不修改Score公式。
- 划分：seed42，P01全部9条与P02的5条进入训练，共14条（stand/left/right=5/5/4）；P02每类留1条完整视频，共3条验证。P03的9条保留独立检查，本轮训练不对其计算模型指标。来源subject_id保持，合并的是训练池。
- 固定留出片段：P02_session01_stand_take001、P02_session01_left_shoulder_take003、P02_session01_right_shoulder_take002。按condition 0/1/2、排序clip ID、NumPy PCG64 seed42选取，不依赖旧模型分数；同一视频和全部重叠窗口、二次处理版本同划分。验证含义改为“训练已覆盖P02姿态后的新视频”，不宣称未见主体泛化；与EXP-035完整P02验证集不能直接当成同一口径比较。
- 数据与配比：保留EXP-035已验收的IK v2及旧库静止窗口，原始窗口和标签逐项继承；每epoch3360，stand/left/right=672/1344/1344。肩部IK/个人各50%；stand中IK/个人/旧库=25%/25%/50%，旧库LAFAN/G1 Moves/AMASS各112。IK11幅度区间均衡、个人按clip均衡，未强制改变骨盆yaw。统一归一化仅用新训练集按来源/clip权重重算，物理量最小跨度规则保持。
- 固定训练：10000epochs、batch1024、d_model128/4heads/2layers、DDPM50steps、10noise samples、AdamW lr3e-4/wd1e-4、clip1、10%条件丢弃、无EMA、guidance1。先独立1epoch smoke，正式模型重新初始化；每100epoch保存恢复状态，每1000epoch保留checkpoint并用新训练集重算V3校准，验证门槛保持，测试集不用于选模。W&B仅配置与标量，模型本地保存。
- 身份与产物：数据manifest SHA-256 `41c2b9f9c466f8a12d86cb812157958b83ccd990533f416d173aaa053b2902f0`；逻辑目录`logs/reproduction/g1_shoulder_mixed_p01p02_prior_v1/`，实际目录`/personal_fast_storage/smp_experiments/g1_shoulder_mixed_p01p02_prior_v1_20261115T094423Z`。GPU0 UUID固定`GPU-eb0fc39f-6914-9e1d-65af-570524e8e061`，启动空闲检查和存储写入探针通过。旧EXP-035数据/模型保持。
- 预检状态：30/30相关测试通过；完整数据准备已通过，真实数据血缘、采样及P03保留审计进行中。后续实际训练状态、W&B及结果续记本条。

- 实际启动（UTC 2026-11-15T09:51:14.446500+00:00）：30/30相关测试通过；真实数据逐窗口/标签与父集一致、原始视频划分无泄漏、P02左右肩实际被抽样、采样恢复完全一致，P03原始数据保持。独立后台驱动PID=360560，执行1epoch scratch smoke后自动从零训练10000epochs；Score和混合配比保持，不启动PPO。

- 运行确认（UTC 2026-11-15T09:58:15.251640+00:00）：独立1epoch smoke通过，4次更新、三条件/NULL DDPM与Score全部有限，实际条件配额672/1344/1344。正式模型与fresh seed42初始29个参数张量完全一致，与已训练smoke参数不同，确认从零启动。正式Prior已到2000/10000epochs、8000次更新，status=validation_running，W&B=https://wandb.ai/qc13659041465-ustc/smp/runs/nd1a5ilo，无失败记录；独立驱动与训练进程持续存活。本轮仅Prior。

- EXP-036 驱动结果（UTC 2026-11-15T10:29:07.524640+00:00）：status=complete_validation_failed，Prior实际10000/10000epochs，W&B=https://wandb.ai/qc13659041465-ustc/smp/runs/nd1a5ilo；selected_validation_candidate=None。本轮只训练Prior，P03未验收，PPO未启动。

- 完成后评估（UTC 2026-11-15T11:31:31.278433+00:00）：正式训练于09:51:34启动、10:29:07结束，完成10000epochs/40000次更新，训练failure=null；10个候选均未通过全部Score门槛。未发布正式Prior，未测试P03，未启动PPO。评估未修改模型、归一化、校准或Score公式，已释放GPU。
- 验证损失最低的是epoch4000（0.09244687），仅为诊断候选，并非已验收模型。逐肩、来源/clip加权的结果如下；epoch4000仅“correct>stand”两项失败，其余门槛均通过：

  | 验证指标 | 左肩 | 右肩 | 原门槛 |
  |---|---:|---:|---:|
  | 正确动作最终Score高于静止 | 77.94% | 80.11% | ≥95% |
  | 正确动作最终Score高于反向 | 100% | 100% | ≥95% |
  | 正确条件能量优于镜像条件 | 97.28% | 96.80% | ≥95% |
  | median margin | 1.012 | 1.808 | ≥0.5 |
  | 来源加权clip均值Score的p05 | 0.2435 | 0.2469 | ≥0.05 |

  静止左肩偏好率57.21%、mean(abs(margin))=0.1026，均通过。IK来源correct>stand为97.62%/98.18%，留出的个人视频为58.25%/62.04%；主要剩余问题集中在个人视频的正确转腰与静止排序。
- 同口径旧/新对比：旧EXP-035与新EXP-036各取其验证损失最低的诊断候选（epoch2000/4000），在完全相同的两条留出P02肩部视频、相同correct/stand/reverse窗口、64组相同配对噪声上评分，各自使用对应归一化和冻结校准。旧/新验证集整体不同，因此不直接比较两次整体validation loss或整体通过率。

  | 留出个人视频指标 | 旧左肩 | 新左肩 | 旧右肩 | 新右肩 |
  |---|---:|---:|---:|---:|
  | 正确动作平均最终Score | 0.05793 | 0.24487 | 0.00835 | 0.24845 |
  | 静止平均最终Score | 0.22715 | 0.20820 | 0.23790 | 0.17496 |
  | 正确动作Score高于配对静止的窗口比例 | 51.46% | 59.22% | 50.00% | 63.89% |
  | 正确动作Score高于反向 | 100% | 100% | 100% | 100% |

  新模型正确动作base均值0.2465/0.2490、contrast均值0.9948/0.9980；旧模型base为0.0625/0.0087。新模型归一化正确能量z均值2.645/2.605，仍使base显著衰减。说明加入P02改善了分布覆盖和正确动作分数，但还不能保证静止稳定低于正确动作。以上每肩只有1条独立留出视频（103/108个重叠窗口），64次噪声重复不增加独立动作来源数量；不作未见主体泛化或单因素因果结论。
- 采样诊断：epoch4000与epoch10000分别每条件采样1024个窗口，guidance=1；跨条件使用相同初始及DDPM噪声，全部有限。epoch4000左/右末帧有符号waist yaw≥0.05rad的比例97.27%/95.80%，腰yaw中位数0.328/-0.550rad；静止窗口最大平面速度p95=0.01291m/s。epoch10000对应方向比例96.97%/87.70%。这些仅为姿态/数值诊断，不是4096样本正式验收、动力学回放或安全GSI接受率验证。
- 后期退化：epoch10000验证损失升至0.09911964；整体correct>stand降至75.20%/74.57%，clip均值Score的p05降至0.04013/0.02641，静止左肩偏好率19.36%。训练损失仍下降，存在过拟合迹象；不能把最后checkpoint当作自动优于较早候选的模型，也不能靠本次结果断言继续增加epochs可解决问题。
- 身份与复现：epoch4000 checkpoint SHA-256 `4df6197499cda4026839f596d21f89361114d232e239a2ec3101349c8fd75043`，calibration SHA-256 `90894d01a2ea4fecdc59cc7fb07898dab1434928c5b24ccd99b78caa2e6d2139`；epoch10000 checkpoint SHA-256 `60d906535c6dbd490d72bab367b8b352308a3cdbea35f2765c6002e501bfeb94`。已重新核对checkpoint及calibration与训练记录身份一致。
- 产物：`logs/reproduction/g1_shoulder_mixed_p01p02_prior_v1/assessment/assessment.json`、`validation_metrics.csv`、`training_assessment.png`、`sampling_diagnostic.png`、`generated_diagnostic_samples.npz`及评估脚本/日志。结论：P01/P02共同训练有实质改善，左右条件已有方向信号，但最终SMP奖励仍未稳定偏爱正确转腰，全部候选未满足原验收要求；本轮只完成Prior训练与验证，历史产物保持不变。

- 数据回放与Prior分布可视化（北京时间2026-11-16 12:49）：按用户要求，新建`logs/reproduction/g1_shoulder_mixed_p01p02_prior_v1/visualization_20261116/`，不改数据、模型、校准、验收或训练。新增P01/P02各stand/left/right一条，共6条完整原始GMR回放：直接读取原30Hz NPZ，保留全部原帧、关节角与root高度，仅作显示坐标xy/yaw对齐，相机跟随root平面位置。960×480、原始30fps、约12.8–15.0秒，每条已完整解码验证。它们是FK姿态回放，不是新动力学执行。
- 复用26条既有四视角对照视频及转角曲线，逐一核对SHA。对照顺序为GMR正面/侧面、二次受约束重定向后的实际执行轨迹正面/侧面；此处GMR统一重采样50Hz。索引覆盖27条原始片段，P02右肩take003没有合格处理后轨迹，明确标记excluded；不将原始GMR基础筛查结果与二次处理后准入状态混淆。未对P03做Prior评分或选模。
- Prior可视化：复用已保存的epoch4000与10000、每条件1024个未过滤DDPM样本（guidance1，跨条件与epoch同初始/后验噪声）；训练参考每条件2048次按来源/clip权重抽样。绘制三条件腰yaw、root速度、差分关节速度分布，静止root高度/pitch/腰变化，逐关节速度尾部及训练集拟合PCA。PCA拟合使用训练epoch0的3360个20/40/40样本，前两维解释约53.51%方差，仅作投影，不是完整590维分布或验收。
- 新增3段epoch4000条件采样拼图：每条件相同随机索引的8个独立10帧样本，不按质量挑选；原50Hz以5fps播放，10倍慢放并循环3次，视频各30帧/6秒、1280×560，已完整解码验证。不是PPO或连续长动作，也未作动力学仿真、轨迹平滑或关节裁剪。
- 分布观察：静止条件窗口最大root平面速度p95，训练参考/epoch4000/epoch10000为0.05644/0.01291/0.01295m/s；按50Hz角度差分计算的窗口最大关节速度p95为0.85075/3.42571/2.11053rad/s。epoch4000逐关节p95较大的有右肩yaw2.702、左肩yaw2.549、腰yaw2.380rad/s等。说明低root速度不能单独证明全身采样平稳；该统计不是电机指令、GSI接受率或PPO失稳的单因素因果证明。
- 可视化产物：`overview.html`集中浏览6条原始GMR、26条对照与曲线、3条Prior样本及4张分布图；`personal_replay_index.csv`、`prior_distribution_report.json`、`prior_distribution_summary.csv`和逐关节CSV保留数值与来源；相邻保存脚本、随机索引、模型/样本身份及完整SHA清单。主模型仍为未通过原Score门槛的4000epoch候选，10000epoch仅作对照，无新训练或正式验收结论变更。

## EXP-037：未通过离线门槛的 P01/P02 Prior——探索性肩部 PPO

- 时间与目的（UTC）：2026-11-15 15:14。按用户“先启动PPO训练看一下吧”的明确授权，绕过EXP-036未通过的离线Prior奖励排序门槛，观察当前Prior能否在真实肩推PPO中产生有效响应。不宣称Prior已验收，不重训或发布Prior，不修改历史实验结论。
- 冻结输入：采用EXP-036验证损失最低的epoch4000诊断候选，Prior SHA-256 `4df6197499cda4026839f596d21f89361114d232e239a2ec3101349c8fd75043`；校准SHA-256 `90894d01a2ea4fecdc59cc7fb07898dab1434928c5b24ccd99b78caa2e6d2139`。校准逐字节复制到新目录，未重新估计μ或改变数值。数据manifest SHA-256 `41c2b9f9c466f8a12d86cb812157958b83ccd990533f416d173aaa053b2902f0`；Prior仍包含IK、P01/P02个人动作及旧库静止数据。P03未用于选模或本次启动。
- 已知限制：epoch4000整体correct>stand仅77.94%/80.11%，未达95%；仅此左右两项门槛失败。原`R_base × R_contrast`镜像肩对比保持，未增加stand条件对比、显式转腰奖励或condition one-hot，因而本实验不声称解决了离线排序问题。
- PPO初始化与课程：Actor、Critic、观测归一化器和优化器从零初始化，A的resume=false；不加载Forward、旧A或smoke策略。A3000轮；首次站稳失败扩展2000轮，扩展仍失败则停止。A通过后分别做左右肩B1接触smoke，覆盖和分类均须≥80%。B1/B2各4000轮，左右肩各50%；B1质量失败记录后继续B2，B2后停止，不运行C或胸背课程。后续仅恢复紧邻阶段完整PPO checkpoint。
- 固定配置：GPU0 UUID `GPU-eb0fc39f-6914-9e1d-65af-570524e8e061`，进程内cuda:0；4096env、seed42、std0.10、初始lr1e-4、每1000轮保存。GSI由同一Prior的stand条件采样，平面速度上限0.15m/s；静止尺度10；球半径0.10m、质量5kg、表面间隙0.02m，正前方向后推。B1为0.01m/s、0.05mg、ramp1.0s；B2为0.03m/s、0.10mg、ramp0.5s。Actor/Critic维度保持128/992，允许骨盆参与，并分别报告肩部世界退让、root补偿退让、腰/骨盆旋转。
- 评估与记录：64env/2iteration独立从零smoke后正式启动；A评估1024episode，B1/B2分别评估无接触、左肩、右肩各1024episode，最终三类各600帧无头回放。独立W&B阶段run同步配置与标量，模型仅本地。非有限值、身份不一致、GPU/进程故障或持续在线SMP全零触发停止；不因Prior未通过离线门槛而冒充正式验收成功。
- 存储与预检：新目录`logs/reproduction/g1_shoulder_mixed_p01p02_ppo_v1_override/`，实际`/personal_fast_storage/smp_experiments/g1_shoulder_mixed_p01p02_ppo_v1_override_20261115T151432Z/`。GPU0启动前无计算进程，持久盘可写且余量充足；13项相关测试通过、Ruff通过，覆盖从零A、明确B1父checkpoint、原校准逐字节保持及损坏拒绝。脚本为`scripts/run_shoulder_mixed_p01p02_ppo_override.py`，驱动、PPO阶段manifest与代码快照全部隔离。
- 实际运行确认（UTC 2026-11-15T15:20:27.879499+00:00）：后台独立驱动PID426287，smoke进程426501已完成64env/2iteration，checkpoint为`model_1.pt`、Actor/Critic=128/992、全部PPO张量有限。正式A进程PID426772，run于15:17:04创建，已实际更新至iteration56/3000；参数快照确认resume=false、upload_model=false、std0.1、初始lr1e-4、save_interval1000、正确Prior及2cm间隙。正式`model_0.pt`有限，内部Prior/校准身份与EXP-036锁定值一致；GSI安全池1024/1024接受、nonfinite=0，nan_or_inf终止=0。GPU0显存约4.9GiB、利用率73%，无其它实验被终止。
- A阶段W&B：<https://wandb.ai/qc13659041465-ustc/smp/runs/tkns1m8h>。已同步EXP-037/exploratory/prior-validation-failed标签及用户授权绕过、Prior epoch4000、score未修改和scratch初始化配置。启动证据保存在`launch_report.json`、`audit/preflight_checks.json`及`audit/wandb_override_annotation.json`；后续A/A扩展、接触smoke、B1/B2由同一持久驱动按课程和门槛顺序执行。

- EXP-037 驱动状态（UTC 2026-11-15T18:27:00.170077+00:00）：status=blocked_stage_a；Prior仍未通过验收，按用户授权探索PPO。阶段={"smoke": {"status": "complete", "checkpoint": "/personal_fast_storage/smp_experiments/g1_shoulder_mixed_p01p02_ppo_v1_override_20261115T151432Z/ppo/rsl_rl/smp_force_unloading_conditional_shoulder_p01p02_override/2026-11-15_15-15-48_g1_shoulder_p01p02_override_scratch_smoke_seed42/model_1.pt", "checkpoint_iteration": 1, "gate_passed": null, "wandb_url": null}, "a": {"status": "complete", "checkpoint": "/personal_fast_storage/smp_experiments/g1_shoulder_mixed_p01p02_ppo_v1_override_20261115T151432Z/ppo/rsl_rl/smp_force_unloading_conditional_shoulder_p01p02_override/2026-11-15_15-17-04_g1_shoulder_p01p02_override_a_seed42/model_2999.pt", "checkpoint_iteration": 2999, "gate_passed": false, "wandb_url": "https://wandb.ai/qc13659041465-ustc/smp/runs/tkns1m8h"}, "a_extension": {"status": "complete", "checkpoint": "/personal_fast_storage/smp_experiments/g1_shoulder_mixed_p01p02_ppo_v1_override_20261115T151432Z/ppo/rsl_rl/smp_force_unloading_conditional_shoulder_p01p02_override/2026-11-15_17-11-39_g1_shoulder_p01p02_override_a_extension_seed42/model_4999.pt", "checkpoint_iteration": 4999, "gate_passed": false, "wandb_url": "https://wandb.ai/qc13659041465-ustc/smp/runs/ga5w13el"}}。

- 训练后评估（北京时间2026-11-16）：实际只完成A3000＋扩展2000，共5000轮；于北京时间2026-11-16 02:27按“A扩展仍未通过”的预设规则停止，status=blocked_stage_a。日志中的RuntimeError是该质量停止分支，不是NaN/Inf、GPU故障或checkpoint损坏。B1/B2、接触smoke和最终肩推回放均未执行，不把本轮写成已完成整个课程。
- 原评估结果：每阶段各1024个已完成的确定性episode。A/扩展的速度峰值p95分别0.98783/1.07779m/s，安全失败率100%/100%，12秒完成率0%/0%，平均episode时长0.4170/0.4623s；主要触发自碰撞和关节硬限位。原评估异步回收失败环境并在累计1024个完成episode后停止，会优先计入较短episode，因此另做固定初始批次复核；原报告和原驱动停止状态不改写。
- 固定初始批次复核：两checkpoint各1024env，seed42、确定性Actor、与原评估相同的Prior/校准/任务/无接触课程；每环境仅收集首个episode，等待所有初始episode结束，后续回收episode不入统计。两批初始状态SHA一致。复核结果如下：

  | 指标 | A 3000轮 | A扩展后5000轮 | 门槛 |
  |---|---:|---:|---:|
  | episode平面速度峰值的p95 | 1.01176m/s | 1.08252m/s | ≤0.30m/s |
  | 安全失败并集率 | 100% | 100% | ≤5% |
  | 完成12秒站立 | 0% | 0% | ≥95% |
  | 平面位移峰值的p50 | 0.04564m | 0.19856m | ≤0.25m |
  | 平均episode时长 | 0.46166s | 0.50232s | — |
  | 最长episode时长 | 1.08s | 1.24s | — |
  | 自碰撞终止率 | 79.20% | 50.20% | 诊断 |
  | 关节硬限位终止率 | 21.00% | 50.59% | 诊断 |

  两类终止可重叠，不能直接相加；均无NaN/Inf终止。所有固定批次episode都在1.3秒内提前结束，原异步采样方式未改变失败结论。位移项通过不能解释为站稳，因为episode过短。失败是安全条件触发，不等同于全部发生跌倒；本次未观察到非脚触地、低高度或倾斜终止。
- 训练曲线复查：A/扩展最后100轮平均episode return为0.01841/0.01844，平均寿命为18.59/22.44个控制步，即0.372/0.449s；训练同样没有形成12秒站立。在线base均值约0.06187/0.05248，固定批次episode平均SMP约0.25343/0.22346；这些正分数不能证明持续站立。此阶段实际条件始终为stand=0，contrast固定为1，尚未启用肩部镜像对比，不能据本轮断言肩部对比Score已在接触任务中验证失败或成功。
- 身份和产物：两个完整PPO checkpoint均通过有限性检查。A `model_2999.pt` SHA-256 `613be30ccbff1186b9723ebc8f5c2e4dc33507f053f2a4ee97624cd38c791202`；最新扩展 `model_4999.pt` SHA-256 `b8691e18577c361aab8287bdfb6249f6b9a3953fd378bb5b0ec996a1ced6c1aa`。17项冻结代码身份全部一致；未改模型、奖励、门槛或重启训练，P03未使用。扩展W&B为<https://wandb.ai/qc13659041465-ustc/smp/runs/ga5w13el>。
- 汇总位于`logs/reproduction/g1_shoulder_mixed_p01p02_ppo_v1_override/assessment/`：`assessment.json`、`evaluation_comparison.csv`、`training_assessment.png`、两阶段`*_cohort.json`和逐episode NPZ，以及脚本、日志和SHA清单。结论：5000轮探索性PPO未满足基础站稳安全条件，未进入肩推课程；目前没有可验收的肩部卸力/转腰策略。
- A3000单独回放（北京时间2026-11-16 11:19）：按用户要求，使用A的`model_2999.pt`（SHA `613be30ccbff1186b9723ebc8f5c2e4dc33507f053f2a4ee97624cd38c791202`），未使用扩展后的4999模型。GPU0、4env、seed42、确定性Actor、原A无接触配置及原Prior/校准，录制600控制帧。完整解码核验为1280×720、50fps、600帧、12.0秒，无非有限指标，录制结束已关闭环境并释放GPU。视频`ppo/replay/a_3000_seed42/rl-video-step-0.mp4`，SHA `facdb4252a9c3d23a977765c96901e232af4d0d0c711568fa1b4efd881da004b`；相邻保存config、evaluation与manifest。安全终止后正常reset，因此视频包含反复短episode，不能当成连续站立12秒；首4个episode含3次自碰撞、1次关节越限，仅作回放诊断，不替代1024episode评估。
- 前倾原因诊断（北京时间2026-11-16 11:51）：响应用户对IK数据的质疑，仅做诊断，不训练或修改历史产物。新训练集stand窗口中，IK骨盆pitch的p05/p50/p95约0.0895°/0.0909°/0.0979°，个人数据约-0.809°/0.125°/0.871°；IK中立姿态未显示明显初始前倾。stand实际抽样为IK25%＋个人25%＋旧库50%，不能因原始窗口数量多就认为本轮只有IK站立数据。
- GPU同状态短程对照：原PPO控制接口、同一64环境、2秒上限，分别使用GSI状态、清零所有初速度的同姿态、真实IK中立窗口，执行A3000 Actor或固定初始关节目标。六组均未存活至2秒；GSI＋Actor平均0.498s，清零初速度＋Actor为0.500s；IK＋Actor为0.627s，IK＋固定关节目标为0.485s。说明只减小GSI初速度或更换成真实IK中立窗口不能消除失败。此GPU固定目标对照保留原encoder bias、摩擦和COM随机化，不当作纯控制器隔离结论。
- 直接前倾指令：同一批真实IK状态下，Actor第一步腰pitch目标相对当前角平均增加0.137rad（约7.85°），同时改变膝/踝目标；前0.2秒64环境都尚未终止，躯干pitch均值由约1.69°升至7.53°，0.4秒存活子集均值约15.43°。这是当前Actor动作确实将躯干带向前倾的证据，不是原始IK初始姿态已经有相同倾角。
- 原模型控制器隔离：使用同样64个已接受IK中立姿态、同一原MuJoCo G1模型和执行器、初速度全零、摩擦0.6、无随机化/外力、自由root，仅改变控制方式。直接把测得关节角作为固定电机位置目标时64/64在2秒内触发诊断失败，平均0.579s；把原逆动力学算出的初始平衡电机目标固定保持时64/64通过2秒检查；原全身逆动力学反馈控制同样64/64通过。诊断检查使用原生关节限位、生成器穿透门槛、root高度≥0.60m和躯干pitch≤20°，不冒充PPO完整12秒验收。该结果说明这些姿态在适当支撑力矩下可实现，不能简单判为IK运动状态错误。
- 控制接口解释：`src/smp/data/shoulder_ik.py::_track_step`计算重力/接触支撑及姿态反馈力矩，再令电机位置目标为`q_actual + (torque + damping*qvel)/stiffness`。导出的CSV/NPZ保存实际运动状态，不是该电机目标。PPO使用`default_offset + action_scale*action`作为位置目标，Actor须自行学出支撑补偿；SMP状态Prior不直接迁移生成器的平衡控制。这是正常状态Prior与控制器的职责区别，不据此宣称数据格式错误，但原“物理可执行”验收未充分覆盖当前PPO冷启动可控性。
- 结论与局限：目前更明确的证据是Actor尚未学会支撑/平衡，且主动产生前弯指令；已检查的IK姿态可在原控制器下支撑。不能把失败单独归因于IK数据，也不能据短程64状态诊断证明全部2400条数据、混合Prior或奖励均无问题。CPU名义控制器对照不含PPO随机化，2秒检查不等于12秒鲁棒站稳；下一步应先在PPO相同控制接口下验证平衡初始化、支撑控制和站立学习，再决定数据与Prior是否需要调整。本轮未启动新训练。
- 诊断产物：`assessment/standing_diagnostic/diagnosis.json`、`standing_input_statistics.json`、`rollout_diagnostic.json`、`controller_comparison.json`、`standing_diagnosis.png`、逐状态NPZ、脚本/日志和SHA清单；均归入EXP-037，保留EXP-032/036历史验收结论的原始含义。
- 旧LAFAN基线对照与归因修正（北京时间2026-11-16 12:12）：用户指出旧LAFAN能够站立后，重新核对EXP-019的static80 A3000历史报告：1024episode中达到12秒超时的比例97.0703%，平均寿命11.6870s；新A3000对应0%/0.4170s，确有显著退化，不能把旧速度p95未达0.30m/s误说成旧策略不会站立。旧报告的非力学失败并集不含全部关节/自碰撞项，本次不将该旧并集率与新版安全并集直接比较。
- 控制配置逐项对照：旧/新训练快照的关节动作映射、执行器stiffness/damping/effort、机器人默认初态、模拟参数、decimation、Actor/Critic网络结构七项全部一致。旧策略已经在原PD控制下学会支撑，因此前述“IK生成需要支撑力矩补偿”只解释姿态可执行性，不是旧→新退化的根因证据；不据此要求正式策略引入额外逆动力学控制器。
- 交叉推理复核：同一当前无接触任务、同一物理参数/PD/观测接口，冻结旧LAFAN static80 A3000 Actor及新A3000 Actor，各加载自身观测归一化，分别使用固定的旧LAFAN GSI与新条件stand GSI窗口。每组64env、seed42、2秒上限，只做Actor推理，不做PPO恢复/训练，不更改或迁移checkpoint。结果：旧Actor＋旧GSI为62/64存活至2秒（96.875%，截尾平均寿命1.940s）；旧Actor＋新GSI为4/64（6.25%，0.293s）；新Actor＋旧GSI为0/64（0.100s）；新Actor＋新GSI为0/64（0.468s）。四组推头力峰值均为0，无NaN/Inf。说明当前环境仍可执行旧站立策略；新初始化分布明显改变旧策略的可适应性，而新策略也未学到可用站立。旧策略在新初态失效不等于这些状态物理不可支撑，短程交叉也不能单独确定哪项训练变化导致失败。
- 同时变化的训练因素：旧Prior为LAFAN static80无条件模型，新Prior为IK/个人/旧库三条件模型，stand每epoch抽样从80%变为20%；特征归一化由旧Prior统计变为混合训练集分位数及物理跨度下限；GSI由旧无条件采样变为新stand条件采样；SMP由策略轨迹在线均值归一化`exp(-6*normalized_error)`变为训练集冻结μ及waist加权的base。A阶段虽不启用肩部contrast，仍使用这个新的base，故站立学习也受评分改动影响。未做训练期单因素消融，不能把退化唯一归因于IK数据、GSI或固定校准。
- 下一步建议与边界：先锁定已验证的LAFAN GSI及原PD/action/PPO配置，只改变SMP Prior/评分做独立A对照，并检查真实平衡轨迹上的base分布，再考虑改变数据或底层控制。新训练预算未启动。对照报告与图为`assessment/standing_diagnostic/old_new_comparison.json`和`old_new_comparison.png`，交叉推理原始结果为`old_new_cross_probe.json`，相邻保存脚本/日志与更新的SHA清单。
