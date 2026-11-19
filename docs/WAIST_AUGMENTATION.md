# 转腰数据增广（Waist Augmentation）

用轨迹级条件扩散模型扩充 G1 人形机器人的转腰训练数据：从少量真实采集 + 程序化锚点出发，生成大量**物理可行**且**风格接近真人**的转腰轨迹。

方法改造自 **GTA: Generative Trajectory Augmentation with Guidance for Offline Reinforcement Learning**（NeurIPS 2024），把它从"依赖 reward 的 offline RL 增广"改造成"不依赖 reward 的运动数据增广"。

---

## 1. 这个方案解决什么问题

人形机器人策略训练时，转腰类动作的数据分布很窄：实测 27 条真实采集的 `waist_yaw` 中位幅度只有 21.8°，最大 39.3°，而任务需要覆盖 15–70°、多种肩外展与屈膝姿态。

单纯用扩散模型生成会撞两个墙：

1. **生成的轨迹物理不可行**——模型学的是关节角序列的联合分布，不建模重心平衡，直接执行会摔倒。
2. **生成的不像人**——训练集里真实数据只占 1%，模型被程序化锚点主导。

本方案对这两个问题各有针对性处理，见下文"核心设计"。

---

## 2. 流程概览

```
真实采集 (27 条 36 列 CSV)
   │
   ├─ retarget_real_with_pelvis_cap.py   ← 骨盆受限重定向，让真实数据物理可行
   │
仿真锚点 (gen_waist_aug_anchors.py)      ← IK + MuJoCo 验收的 1000+ 条参数化轨迹
   │
   ▼
train_waist_aug.py                       ← 轨迹级条件扩散（真实数据 10× 权重）
   │
   ▼
sample_waist_aug.py                      ← 部分加噪 + CFG + 约束梯度 → IK 后处理 → 物理验证
   │
   ▼
export_waist_aug_windows.py              ← 导出 (N,10,59) 窗口，直接接入现有 prior 训练
```

---

## 3. 快速开始

### 3.1 环境

```bash
uv sync          # 或 pip install -e .
```

依赖见 `pyproject.toml`：`mjlab==1.6.0`、`mink==1.4.0`、`mujoco`、`torch`、`tyro`、`numpy`。

**在无显示器的服务器上**注意：`scripts/csv_to_npz.py` 需要 EGL 渲染。本方案用 `export_waist_aug_windows.py`（纯 `mj_forward`，不依赖渲染）替代，专门适配无 EGL 环境。

### 3.2 数据获取

三个数据包在仓库的 `v1.0-data` Release 里下载：

| 包 | 大小 | 内容 | 解压到 |
|---|---:|---|---|
| `waist_aug_real.tar.gz` | 8.3 MB | 27 条真实采集（原始 + 骨盆受限） | `.` |
| `waist_aug_anchors.tar.gz` | 158 MB | 1064 条 IK 验收锚点 | `logs/reproduction/waist_aug_v1/` |
| `waist_aug_data.tar.gz` | 36 MB | 95 条生成轨迹 + 7790 窗口 | `logs/reproduction/waist_aug_v1/` |

```bash
mkdir -p logs/reproduction/waist_aug_v1
tar xzf waist_aug_real.tar.gz -C .
tar xzf waist_aug_anchors.tar.gz -C logs/reproduction/waist_aug_v1/
tar xzf waist_aug_data.tar.gz -C logs/reproduction/waist_aug_v1/
```

解压后目录结构：

```
data/real/                  27 条原始真实采集（30 fps，36 列）
data/real_pelvis_capped/    27 条骨盆受限版本（50 fps）
logs/reproduction/waist_aug_v1/
  anchors/clips/accepted/   1064 条锚点
  humanlike_full/           95 条生成轨迹（最终交付）
  windows_humanlike/        7790 个 (10,59) 窗口
```

锚点也可用 `scripts/gen_waist_aug_anchors.py` 从零生成（16 核约 20 分钟），不必下载。**真实采集无法重新生成**——它们来自原始视频采集。

### 3.3 数据准备（自备数据时）

**真实采集**：36 列无表头 CSV（`root xyz(3) + quat xyzw(4) + 29 关节角`），放到 `data/real/`，文件名以 `P0` 开头。

**先做骨盆受限重定向**（关键，否则真实数据过不了物理验证）：

```bash
PYTHONPATH=src python scripts/retarget_real_with_pelvis_cap.py \
  --cfg.real-csv-dir data/real \
  --cfg.out-dir data/real_pelvis_capped \
  --cfg.pelvis-yaw-limit 0.0524     # 3 度
```

这一步会把真实数据的骨盆转动（典型 30°）压到 3° 以内，被截掉的部分**转移到腰部**——得到"真人风格但物理可行"的样本。

### 3.4 生成仿真锚点（可选，仓库不附带）

锚点是物理合法性训练的基底。若没有现成的，用：

```bash
PYTHONPATH=src python scripts/gen_waist_aug_anchors.py \
  --out-dir logs/reproduction/waist_aug_v1/anchors --workers 16
```

1200 个参数组合（站距 × 膝屈曲 × 腰幅度 × 起转速度 × 肩外展），MuJoCo 验收后约 85% 通过，16 核约 20 分钟。

### 3.5 训练

```bash
PYTHONPATH=src python scripts/train_waist_aug.py \
  --cfg.real-csv-dir data/real_pelvis_capped \
  --cfg.num-epochs 600 --cfg.batch-size 256 --cfg.num-noise-samples 4 \
  --cfg.real-weight 10 --cfg.device cuda
```

单张 A100 约 19 分钟，最终 val loss ≈ 0.125。

### 3.6 采样 + 验证

```bash
PYTHONPATH=src python scripts/sample_waist_aug.py \
  --cfg.ckpt <checkpoint.pt> \
  --cfg.amp-grid 15,25,35,45,55,70 \
  --cfg.speed-grid 9,13,17 \
  --cfg.abduction-grid 15,32,55 \
  --cfg.knee-grid 18,33 \
  --cfg.pin-frames 12 \
  --cfg.include-real-seeds --cfg.real-seed-prob 0.5 \
  --cfg.real-csv-dir data/real_pelvis_capped \
  --cfg.out-dir logs/reproduction/waist_aug_v1/humanlike_full
```

产物：`<out>/clips/{accepted,rejected}/train/<condition>/*.csv`，36 列、50 Hz，与仓库现有 G1 数据格式一致。

### 3.7 导出下游格式

```bash
PYTHONPATH=src python scripts/export_waist_aug_windows.py \
  --cfg.input-glob "<out>/*/clips/accepted/**/*.csv" \
  --cfg.out-dir logs/reproduction/waist_aug_v1/windows
```

输出 `(N,10,59)` NPZ，可直接被 `smp.pretrain.dataset.MotionWindowDataset` 读取。

---

## 4. 核心设计

### 4.1 表示

每帧 38 维：`root_pos(3, xy 相对段首、z 世界高度) + root_rot 6D + 29 关节角`。

用 6D 旋转而非四元数，避免流形约束；末端位置与速度由 FK 导出，不参与生成（它们与关节角冗余）。段长 H=160 @30 fps（5.33 s），stride 32。

### 4.2 条件：18 维属性向量（GTA 的回报 → 属性）

GTA 用折扣回报做条件，作为标量它单调可外推。关节属性不是，所以改为显式指定目标：

| 组 | 维度 |
|---|---|
| 腰 | 带符号幅度、平均角速度、方向、起转时长 |
| 下肢 | 膝屈曲、髋 yaw |
| 左/右臂 | 肩 roll/yaw/pitch、肘、腕 roll |
| 初态 | 初始腰 pitch / roll |

条件 dropout 10% 训练 CFG。

### 4.3 采样：部分加噪 + 约束引导（GTA Stage-B 改造）

```
种子 = 与目标条件最接近的窗口（关键维度加权 ×8，见 4.5）
x ← 种子加噪到 start_timestep = 15 / 100
每步：
  ε = ε_∅ + w·(ε_c − ε_∅)               # CFG，w 由 1 线性升到 2
  ε ← ε − λ·√(1−ᾱ)·∇R(x̂₀)              # 约束引导，λ=5
  前 12 帧固定为种子                     # 保证"同起态"
```

约束残差 `R` 三项，全部可在 38 维上直接算（无需 FK）：

| 项 | 阈值 | 目的 |
|---|---|---|
| root yaw 相对首帧 | ≤ 5° | 禁止靠骨盆转身代替转腰 |
| 双髋 yaw 段内变化 | ≤ 8° | 禁止脚转代偿 |
| 腰 pitch/roll 相对中位 | ≤ 3° | 禁止用前后仰/侧弯冒充转腰 |

### 4.4 IK 后处理：物理可行性的保证

**这是整个方案最关键的一环。** 扩散生成的轨迹运动学合理，但动力学上会摔倒——锚点的 CoM 平衡是 IK 的 `ComTask` 保证的，而扩散模型不建模它。

做法：把生成轨迹当**软参考**送进 `PersonalRetargeter`：

- 上身跟随生成的关节增量
- **腿部由 CoM + 足端约束重新求解**
- 生成器的臂/膝姿态通过 `reference(..., overrides)` 保留（否则会被重定向器的固定站姿表覆盖）

实测：纯生成 0/4 通过，加 IK 后处理 105/108 通过。

### 4.5 让生成"像人"：三个必须同时成立的改动

| 改动 | 效果 |
|---|---|
| 真实数据骨盆受限重定向 | root yaw 30.1°→1.6°，waist 21.9°→36.1°（转动转移到腰） |
| 训练时真实数据 10× 过采样 | val loss 0.206 → 0.125 |
| 采样时 50% 概率用真实种子 | 避免被 1000+ 锚点淹没 |

**注意**：只把真实数据放进种子池（靠距离匹配）是**无效的**——锚点参数覆盖更全，真实窗口抢不到选中机会。必须按概率强制。

---

## 5. 文件清单

### 新增

| 文件 | 作用 |
|---|---|
| `src/smp/pretrain/waist_aug_dataset.py` | 38 维表示、起转定位、切片、18 维条件 |
| `src/smp/pretrain/trajectory_sampling.py` | DDIM + CFG + 约束梯度引导 |
| `src/smp/data/waist_aug_specs.py` | 锚点参数网格 |
| `scripts/gen_waist_aug_anchors.py` | 批量锚点生成（IK + MuJoCo 验收） |
| `scripts/retarget_real_with_pelvis_cap.py` | 真实数据骨盆受限重定向 |
| `scripts/train_waist_aug.py` | 条件扩散训练 |
| `scripts/sample_waist_aug.py` | 采样 + IK 后处理 + 物理验证 |
| `scripts/export_waist_aug_windows.py` | 导出 (N,10,59) 窗口（无需渲染） |
| `scripts/report_waist_aug_coverage.py` | 期望表覆盖清单 |
| `scripts/report_aug_marginal_coverage.py` | 边际覆盖分析（相对锚点的新增格子） |
| `tests/test_waist_aug_dataset.py` | 数据集单元测试（15 项） |

### 修改（均向后兼容）

| 文件 | 改动 |
|---|---|
| `src/smp/pretrain/model.py` | `DiffusionDenoiser` 支持连续条件 `condition_dim` |
| `src/smp/data/shoulder_ik.py` | `ClipSpec` 与站姿参数化（肩外展/膝/腕/趾/髋） |
| `src/smp/data/personal_retarget.py` | `reference()` 姿态 override、`pelvis_yaw_limit`、`allow_onset_fallback` |

---

## 6. 实测结果

108 组目标（6 幅度 × 3 速度 × 3 外展 × 2 膝），每格 1 条：

| 指标 | 数值 |
|---|---|
| 物理通过率 | **105/108（97.2%）** |
| 生成 → 真人数据距离 | **0.0744 rad**（初版 0.178，降低 58%） |
| 生成 → 锚点距离 | 0.0411 rad |
| 腰幅度覆盖 | 15.6–64.7° |
| 肩外展三档 | 12–16 / 28–33 / 47–53° |
| 膝两档 | 21–37° |
| 约束实测 | root yaw 0.02°、髋 1.7°、腰俯仰 0.2°（阈值 5/8/3°） |

未通过的 3 条是 shoulder-endpoint 语义项（完成时间/端点位置），不是物理崩溃。

---

## 7. 已知限制

1. **起转速度中位 8.4°/s，略低于目标下限 9**。p95 已达标（12.5）。损失来自重定向的 Savitzky-Golay 平滑（11 帧）；收紧到 7 帧只能提升 5–6%，代价是轨迹更抖，不划算。

2. **真人相似度是相对这 27 条采集**（3 名受试者）。受试者覆盖面有限，生成结果继承该局限。

3. **骨盆受限改变了真实数据的原始形态**。这是为了让数据物理可行而做的转换，不是原始采集值；原始数据应保留在 `data/real`。

4. **未做下游策略验证**。验收标准是"数据本身合格"，已达成；但这批数据能否提升 RL 策略，需要另外的对照实验。

5. **`norm_stats.npz` 未重算**。用 LAFAN 统计归一化时约 3.2% 的值超出 ±1，接入训练前建议用包含转腰数据的统计重算。

---

## 8. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `Source lacks a neutral prefix...` | 源轨迹缺 1 s 中性前缀或末段保持 | 采样时 `allow_onset_fallback=True`（已在 `sample_waist_aug.py` 启用）；真实数据用中位数姿态 padding |
| 生成轨迹执行后摔倒 | 没走 IK 后处理 | 必须用 `PersonalRetargeter.reference`，不能让扩散输出直接执行 |
| 生成的臂/膝姿态和设定不符 | 重定向器用固定站姿表 | 传 `overrides`（`sample_waist_aug.py` 已实现） |
| 生成的全贴着锚点 | 种子池只有锚点 | 启用 `--cfg.include-real-seeds` 并设 `real_seed_prob`，**不能只靠距离匹配** |
| 速度统计偏小 | 50 fps 数据按 30 fps 换算 | 统计时显式传 `fps=50` |
| CFG 完全不生效 | 训练与采样的 null 分支不一致 | 训练时无条件样本必须走 `condition=None` |

---

## 9. 引用

- GTA: Lee et al., *Generative Trajectory Augmentation with Guidance for Offline Reinforcement Learning*, NeurIPS 2024.
- SMP: Mu et al., *Reusable Score-Matching Motion Priors for Physics-Based Character Control*, 2025.
- OmniControl: Xie et al., ICLR 2024（解析式引导的模板）。
- PhysDiff: Yuan et al., ICCV 2023（投影式物理引导）。