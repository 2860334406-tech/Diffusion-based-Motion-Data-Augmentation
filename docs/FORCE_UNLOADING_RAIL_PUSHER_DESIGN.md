# 基于 SMP 的导轨球形推头卸力任务设计

## 1. 目标与设计原则

本任务使用“一维导轨上的动态球形推头 + 纵向弹簧阻尼驱动”，训练机器人在胸口或肩膀受到缓慢外力时，通过顺势平移、转腰、屈膝或调整支撑降低实际接触力，同时保持平衡并避免离开工作区域。

任务沿用 /root/MimicKit 里面 Dodgeball 的主要范式：

- 外部事件在 episode 内随机出现；
- policy 持续接收当前外部交互状态，不接收未来轨迹或倒计时；
- 全程使用同一套 task reward，不为提示、加载、卸力和恢复分别切换奖励权重；
- 严重失败时提前终止，从而损失剩余回报；
- 最终奖励由 task reward 与 SMP motion-prior reward 组合。

本方案规定 actor 和 critic 的额外任务相关信息均为：

1. 机器人受到的法向接触力大小；
2. 法向接触力方向；
3. 法向接触力的作用位置。

不向 actor 或 critic 提供推头下一次启动时间、未来参考轨迹、距离加载结束的时间或预先定义的任务阶段倒计时。

> 重要限制：首次接触发生前，接触力大小为 0，方向和位置也没有有效值，因此策略无法在首次接触前预测外力。它能学习的是在微小接触力开始建立后尽快反应，并利用缓慢加载过程提前卸力。如果后续目标变成“接触前主动准备”，至少需要额外提供推头相对位置和速度，或者提供外部提示信号。

## 2. 推头机械模型

### 2.1 一维导轨约束

设导轨方向为单位向量 \(\hat{\mathbf n}\)，球心只能沿导轨移动：

\[
\mathbf p_b(t)=\mathbf p_0+u(t)\hat{\mathbf n}
\]

其中 \(u\) 是球在导轨上的单自由度坐标。导轨约束球心的横向移动；球体旋转进行锁定，以减少滚动和切向摩擦带来的额外变量。

### 2.2 弹簧阻尼驱动

参考坐标沿导轨缓慢前进：

\[
u_{\mathrm{ref},t+1}
=u_{\mathrm{ref},t}+v_{\mathrm{ref}}\Delta t
\]

驱动力为：

\[
F_{\mathrm{drive}}
=\operatorname{clip}\left(
k(u_{\mathrm{ref}}-u)
+c(\dot u_{\mathrm{ref}}-\dot u),
0,
F_{\max}
\right)
\]

球沿导轨的动力学为：

\[
m_b\ddot u
=F_{\mathrm{drive}}
-F_{\mathrm{contact}}
-b\dot u
\]

机器人可以通过接触反力让球减速、停止或短暂后退，因此该推头不同于不会受机器人影响的运动学球体。

### 2.3 建议初始参数

| 参数 | 第一版设置 |
|---|---:|
| 球半径 | 0.15–0.20 m |
| 球质量 | 3–8 kg |
| 参考速度 | 0.03–0.08 m/s |
| 纵向刚度 | 400–1000 N/m |
| 阻尼比 | 0.8–1.2 |
| 阻尼系数 | \(c=2\zeta\sqrt{k m_b}\) |
| 最大驱动力 | \(0.10mg\)–\(0.20mg\) |
| 球—机器人摩擦系数 | 0.10–0.30 |
| 恢复系数 | 0 |
| 最大安全行程 | 0.30–0.45 m |

其中 \(m\) 是机器人总质量，\(m_b\) 是推头球体或滑块总质量。

## 3. Dodgeball 式事件流程

环境不把阶段编号作为 task observation，但内部仍需要一个事件状态机控制推头安全运行。

### 3.1 等待

- 每次事件前随机等待 1–3 s；
- 机器人只获得零接触标志，不知道剩余等待时间；
- 推头撤回到安全位置；
- task reward 公式保持不变；等待期无接触力变化按中性
  `r_drop=0.5` 处理，以维持 SMP 门控的密集学习信号。

### 3.2 缓慢加载

- 随机选择胸口、左肩或右肩；
- 在目标部位附近建立导轨方向；
- `push_direction_mode=target_default` 保留胸口纵向、肩部侧向的原始行为；
- `push_direction_mode=front_to_back` 将胸口和左右肩统一为机器人 heading 局部 `-x`，球头从所选部位正前方纵向向后推进；
- 参考坐标以平滑速度开始前进；
- 推头与机器人接触后，actor 从接触力信息中检测外力；
- 不给 policy 提供推头的参考坐标、参考速度或未来轨迹。

### 3.3 卸力完成与撤回

接触发生后，如果力下降到完成阈值并保持稳定，则判定本次卸力事件完成，推头撤回。撤回期间关闭“力下降奖励”，避免把执行器主动撤回造成的力下降算作机器人贡献。

完成一次事件不立即结束 episode。环境重新采样等待时间和目标部位，类似 Dodgeball 对同一个球进行下一次重新发射。

完整课程可令每个 10–12 s episode 包含 1–3 次推力事件；当前第一阶段基线固定为 12 s、每个 episode 一次事件，完成后进入 `DONE`，不在同一 episode 内重新加载。

## 4. 法向接触信息的构造

### 4.1 合成法向接触力

设球与机器人之间存在若干接触点。第 \(i\) 个接触点的法向、法向力大小和位置分别为 \(\hat{\mathbf n}_i\)、\(F_{n,i}\) 和 \(\mathbf p_i\)。机器人受到的合成法向接触力为：

\[
\mathbf F_n
=\sum_i F_{n,i}\hat{\mathbf n}_i
\]

力大小为：

\[
F_n=\|\mathbf F_n\|_2
\]

力方向为：

\[
\hat{\mathbf f}
=\frac{\mathbf F_n}{F_n+\epsilon}
\]

不要把切向摩擦力加入 \(F_n\)，否则摩擦系数变化会直接改变卸力指标。

### 4.2 等效作用位置

采用法向力加权的接触中心：

\[
\mathbf p_{\mathrm{app}}
=\frac{\sum_i F_{n,i}\mathbf p_i}
{\sum_i F_{n,i}+\epsilon}
\]

这样当接触从肩膀滑向胸口时，作用位置会连续变化，而不是继续停留在预先指定的肩膀点。

### 4.3 局部坐标与归一化

设 \(\mathbf R_h\) 是机器人 heading rotation，\(\mathbf p_r\) 是机器人根节点位置。输入 policy 的方向和位置为：

\[
\hat{\mathbf f}_{\mathrm{local}}
=\mathbf R_h^T\hat{\mathbf f}
\]

\[
\mathbf p_{\mathrm{app,local}}
=\mathbf R_h^T(\mathbf p_{\mathrm{app}}-\mathbf p_r)
\]

力大小按机器人重量归一化：

\[
f_n
=\operatorname{clip}\left(
\frac{F_n}{mg},
0,
f_{\max}
\right)
\]

接触标志为：

\[
c=\mathbb I(F_n>F_{\mathrm{on}})
\]

最终单帧任务观察为：

\[
\mathbf o_{\mathrm{task}}
=\left[
c,
f_n,
\hat{\mathbf f}_{\mathrm{local}},
\mathbf p_{\mathrm{app,local}}
\right]
\]

维度为 8：接触标志 1 维、力大小 1 维、方向 3 维、作用位置 3 维。

无接触时设置：

\[
c=0,\quad
f_n=0,\quad
\hat{\mathbf f}_{\mathrm{local}}=\mathbf 0,\quad
\mathbf p_{\mathrm{app,local}}=\mathbf 0
\]

### 4.4 历史信息

只给当前一帧接触力会形成较强的部分可观测问题，因为 policy 不知道力是在增加还是降低。第一版建议堆叠最近 4 帧相同的任务观察：

\[
\mathbf o_{\mathrm{task}}^{H}
=\left[
\mathbf o_t,
\mathbf o_{t-1},
\mathbf o_{t-2},
\mathbf o_{t-3}
\right]
\]



## 5. Actor 与 Critic 观察

### 5.1 Actor

Actor 保留现有 SMP policy 使用的机器人本体感觉，包括关节位置、关节速度、根节点角速度、projected gravity、上一时刻动作等，然后拼接 4 帧任务观察历史：

\[
\mathbf o_{\mathrm{actor}}
=\left[
\mathbf o_{\mathrm{proprio}},
\mathbf o_{\mathrm{task}}^H
\right]
\]

Actor 不接收：

- 下一次推力事件的触发时间；
- 推头参考位置或参考速度；
- 弹簧伸长量；
- 驱动器内部状态；
- 预先指定的目标身体部位；
- 未来推力大小或方向。

为了面向实机，actor 输入的接触力建议加入：

- 第一阶段基线不加延迟，后续课程加入 1–2 个控制步的随机延迟；
- 2%–5% 的幅值噪声；
- 小幅方向和位置噪声；
- 50–100 ms 的低通滤波。

### 5.2 Critic

Critic 的任务信息与 actor 具有相同语义：法向力大小、方向和作用位置。为了进行 asymmetric actor-critic，critic 可以使用未加噪、无延迟的精确值：

\[
\mathbf o_{\mathrm{critic}}
=\left[
\mathbf o_{\mathrm{critic,base}},
\mathbf o_{\mathrm{task,exact}}^H
\right]
\]

本方案不把下一次触发时间、未来参考轨迹或未来推力提供给 critic，避免 critic 通过 actor 不可能获得的信息过度预测事件。如果现有框架没有独立 privileged critic observation，则 actor 和 critic 直接共享同一份带噪任务观察。

SMP prior 的输入保持不变，只包含机器人运动状态窗口，不拼接接触力、推头或任务观察。

## 6. 接触力滤波

硬接触力存在明显的物理子步噪声。先对一个控制周期内的所有仿真子步求平均，再进行 EMA：

\[
\bar F_t
=\alpha\bar F_{t-1}
+(1-\alpha)F_t
\]

建议：

\[
\alpha=0.70\text{–}0.85
\]

奖励与常规完成判断使用 \(\bar F_t\)，紧急安全终止同时检查未经滤波的原始峰值。

## 7. 统一奖励函数

奖励直接复刻 Dodgeball 的两项式结构：一个主任务奖励加一个小权重的 stationary 奖励。平衡、工作区域、足底滑动和动作变化不再作为密集奖励项；其中安全相关条件放入失败终止，动作自然性由 SMP prior 约束。

### 7.1 推力下降奖励

使用归一化接触力：

\[
\bar f_t=\frac{\bar F_t}{mg}
\]

定义：

\[
r_{\mathrm{drop}}
=m_{\mathrm{drive}}
\frac{1+\tanh\left(
\frac{\bar f_{t-1}-\bar f_t}
{\Delta f_{\mathrm{scale}}}
\right)}{2}
\]

其中：

- 推头前进或保持时，\(m_{\mathrm{drive}}=1\)；
- 推头主动撤回时，\(m_{\mathrm{drive}}=0\)；
- 推力降低时奖励大于 0.5；
- 推力不变时奖励为 0.5；
- 推力升高时奖励小于 0.5，但保持非负。

建议从 \(\Delta f_{\mathrm{scale}}=0.003\text{–}0.01\) 开始调节。

为避免推头撤回造成虚假的正奖励，主动撤回时必须令 \(m_{\mathrm{drive}}=0\)。每次新推力事件开始时，还应将用于差分的上一时刻力初始化为当前力，避免事件切换产生奖励尖峰。

### 7.2 Stationary 奖励

与 Dodgeball 一致，只约束根节点的水平速度：

\[
r_{\mathrm{stationary}}
=\exp\left(-k_v\left\|\mathbf v_{\mathrm{root}}^{xy}\right\|^2\right),
\qquad k_v=1.0
\]

该项在机器人水平速度接近 0 时接近 1，速度增大时下降。它用于抑制机器人通过持续奔跑或大幅平移逃离推头，但不直接限制转腰、侧身、屈膝或短暂顺力移动。由于卸力本身需要身体运动，该项只占较小权重。

实现中由 `stationary_speed_scale` 配置 (k_v)，默认值仍为 1.0；EXP-018～020 的专项课程覆盖为 10.0，并把 GSI root 平面速度上限统一覆盖为 0.15 m/s。

### 7.3 Task reward

采用与 Dodgeball 相同的 0.9/0.1 配比：

\[
r_{\mathrm{task}}
=0.9r_{\mathrm{drop}}
+0.1r_{\mathrm{stationary}}
\]

该形式只有两个任务项：

- 推力下降时，\(r_{\mathrm{drop}}>0.5\)；
- 推力不变时，\(r_{\mathrm{drop}}=0.5\)；
- 推力上升时，\(0\le r_{\mathrm{drop}}<0.5\)；
- stationary 仅提供较弱的原地保持倾向。

高力保持不变和低力保持不变的 \(r_{\mathrm{drop}}\) 都为 0.5。二者通过第 8 节的硬完成阈值区分：只有 \(\bar F_t<F_{\mathrm{done}}\) 才算卸力完成；高力静止还可能最终触发紧急力或 episode 超时。

### 7.4 与 SMP reward 组合

\[
r_{\mathrm{total}}
=r_{\mathrm{task}}\,r_{\mathrm{smp}}
\]

仓库现有下游任务使用 `task_smp_product` 做乘法门控，因此本任务保持同一约定。`r_drop` 必须先映射到 \([0,1]\)，避免负 task reward 与低 SMP reward 相乘形成符号或尺度漏洞；任务配置中不得再叠加独立 SMP 奖励项。

## 8. 事件完成条件

环境内部维护 `contact_latched`。当过滤后的力连续若干步超过很低的接触阈值时置为真：

\[
\bar F_t>F_{\mathrm{on}},
\qquad
F_{\mathrm{on}}\approx0.005mg\text{–}0.01mg
\]

只有已经发生有效接触，并且推头至少运行了最短测试时间或最短测试行程后，才允许判定卸力完成，防止球尚未接触就被当作成功。

卸力完成需要同时满足：

\[
\bar F_t<F_{\mathrm{done}}
\]

\[
\left|
\frac{\bar F_t-\bar F_{t-1}}{\Delta t}
\right|
<\dot F_{\mathrm{done}}
\]

并且机器人未触发第 9 节的任何失败条件，仍处于硬工作区域内。建议：

\[
F_{\mathrm{done}}=0.02mg\text{–}0.04mg
\]

上述条件连续满足 0.3–0.5 s 后，本次事件完成并撤回；当前单事件基线进入 `DONE`，后续多事件课程才重新进入随机等待。

如果球从身体表面完全滑脱，使 \(F=0\)，只要机器人保持平衡且没有逃出工作区域，也视为有效卸力。

## 9. Episode 终止条件

### 9.1 失败提前终止

满足任一条件时立即以失败结束 episode：

1. 非允许身体部位接触地面；
2. 根节点高度低于初始高度的 55%–65%；
3. 躯干倾角超过约 60°；
4. 从推力事件开始时刻起，根节点超出硬工作区，例如纵向超过
   0.60 m 或横向超过 0.40 m；随机等待期不计入该位移，避免 GSI
   locomotion 初态在事件触发前被工作区终止；
5. 原始接触力连续 2–4 个物理子步超过紧急阈值；
6. 关节状态超出硬安全限制；
7. 状态中出现 NaN 或 Inf。

推头达到最大行程不再作为失败终止条件。slide joint 的物理行程上限仍然保留，环境通过 `travel_limit_reached` 单独记录到达上限的事件；只要没有触发其他安全终止，episode 继续运行到卸力完成或时间上限。该遥测本身不改变第 8 节的低力成功判据。

紧急接触力建议初始设为：

\[
F_{\mathrm{emergency}}=0.25mg\text{–}0.35mg
\]

### 9.2 正常结束

Episode 达到 10–12 s 时间上限且没有失败，则正常结束。和 Dodgeball 类似，不要求每次卸力完成后立即终止；主要成功指标是完整 episode 中保持存活和平衡，同时降低接触力。

## 10. 训练与评估指标

除了 episode return，必须单独记录：

- 接触力峰值 \(F_{\mathrm{peak}}/(mg)\)；
- 接触力时间积分 \(\int F\,dt\)；
- 接触后达到 \(F_{\mathrm{done}}\) 的时间；
- 每次事件是否成功卸力；
- episode 内成功事件比例；
- 根节点最大纵向和横向位移；
- capture point 最大越界距离；
- 足底最大滑动速度；
- 跌倒率和紧急力终止率；
- SMP reward 的均值和方差。

最终模型选择不应只根据总奖励，应同时检查低跌倒率、低峰值力、短卸力时间和有限根节点位移。

## 11. 随机化与课程学习

第一阶段：

- 只推胸口；
- 固定导轨方向；
- 低参考速度和低最大驱动力；
- 每个 episode 一次事件；
- 无观测延迟和较小噪声。

第二阶段：

- 加入左肩和右肩；
- 推力方向在目标法线附近随机 ±10°；
- 随机刚度、阻尼、质量和摩擦；
- 每个 episode 1–3 次事件。

第三阶段：

- 推力方向随机 ±20°；
- 增加驱动力和参考速度范围；
- 加入力、方向、位置观测噪声与延迟；
- 加入 10%–20% 无推力 episode；
- 随机地面摩擦和机器人负载。

## 12. 实现建议


当前实现位于 `src/smp/rl/tasks/force_unloading/`，注册任务名为 `Smp-Force-Unloading-G1`，继续使用 mjlab 的 `ManagerBasedRlEnv` 和 manager term，而不是新增独立环境基类。

`RailPusherController` 作为 class-based step event 维护以下职责：

- 构建仅含一个 slide joint 的动态球形推头；固定基座由 mjlab 自动包装为 mocap body，以便按机器人 heading 放置导轨；
- 维护 `WAIT → DRIVE → RETRACT → DONE` 状态机、弹簧阻尼参考和安全行程；
- 聚合并滤波接触法向力，缓存 observation/reward/completion/safety 所需状态；
- manager observations、rewards、terminations 和 metrics 读取同一缓存，避免一控制步内重复推进滤波器。

接触信息直接使用 mjlab 1.6 的 `ContactSensor`：

- primary 为 `rail_pusher/sphere`，secondary 为 G1 `pelvis` 子树；
- 请求 `found/force/pos/normal`，`reduce="none"`，保留 8 个接触槽；
- `history_length=4` 覆盖一个控制周期的 4 个 MuJoCo 物理子步；
- 只取 contact-frame force 的法向分量，并用世界坐标法线和接触点合成等效力与作用位置。

因此不需要 PhysX rigid-contact report 或额外 engine 接口。


SMP 的 `disc_obs` 和 prior 输入保持原样，task observation 只进入 actor/critic 网络。

## 13. 第一版验收标准

开始 PPO/SMP 大规模训练前，先用脚本或零策略验证：

1. 推头球心严格限制在导轨上；
2. 机器人反作用力能够让球减速或后退；
3. 推头主动撤回不会产生正的 `r_drop`；
4. 肩膀转动时作用位置能够移动；
5. 球滑向胸口后接触力不会被漏算；
6. 无接触时任务观察严格为零；
7. 接触力滤波不会引入过长延迟；
8. 高力静止状态的 `r_drop` 为 0，且不能通过 \(F_{\mathrm{done}}\) 完成条件；
9. 摔倒、越界和紧急力能够可靠提前终止；
10. task observation 不会误传入 SMP prior。
