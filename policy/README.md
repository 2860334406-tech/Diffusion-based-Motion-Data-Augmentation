# 策略与配套文件交接

这里存放供合作者加载、评估或继续训练的配套文件。每个版本使用独立子目录，把策略、prior、配置、评估结果和说明放在一起。

## 目录结构

`_template/` 是版本模板。新增版本后，目录可以是：

```text
policy/
├── README.md
├── _template/
│   ├── README.md
│   └── manifest.json
└── chest_speed_gradient_v1/
    ├── README.md
    ├── manifest.json
    ├── policy.pt
    ├── prior.pt
    ├── calibration.pt       # 使用冻结 Score 校准的任务才需要
    ├── env.yaml
    ├── agent.yaml
    ├── evaluation.json
    └── SHA256SUMS
```

`_template/` 用于创建新版本。已归档的胸推版本还包含训练血缘、prior 元数据、原始源码差异和便携评估脚本。

## 已归档版本

| 目录 | 实验与策略 | 历史评估 |
| --- | --- | --- |
| [chest_speed_gradient_v1](chest_speed_gradient_v1/README.md) | EXP-022，`model_31995.pt`，匹配 static80 prior | 0.20 m/s 胸推，1024 回合，卸力成功率 98.83%，通过原 C 阶段门槛。 |

## 每个版本提供什么

| 文件 | 内容 |
| --- | --- |
| `policy.pt` | 原始 PPO checkpoint。保留 Actor 及其观测归一化状态；继续训练时还需要 Critic、optimizer 和迭代计数等完整恢复状态。 |
| `prior.pt` | 与该策略配套的 SMP prior，以及它使用的特征归一化统计。若统计不在 checkpoint 中，需要另外提供并记入清单。 |
| `calibration.pt` | 使用冻结 Score 校准时提供，必须与 prior、归一化及条件 schema 匹配；其他任务在清单中将此项设为 `null`。 |
| `env.yaml`、`agent.yaml` | 该次运行实际保存的环境与 PPO 配置。加载时按接收方路径调整文件位置，并显式使用对应任务参数。 |
| `evaluation.json` | 对这份 checkpoint 的评估结果，注明任务、推速、随机种子、回合数、通过情况与已知不足。 |
| `manifest.json` | 文件 SHA-256、代码版本、训练来源、归一化恢复方式和力观测接口。 |
| `README.md` | 本版本的用途、实际验证过的加载/评估/续训命令，以及文件路径如何对应。 |
| `SHA256SUMS` | 包内文件的校验清单，供合作者检查传输完整性。 |

已归档胸推版本使用 EXP-022 的 `model_31995.pt` 和 `pretrained_lafan_low_speed_static80.pt`。这套旧评分方案不使用独立的冻结 Score 校准文件。结果与局限见[正式评估报告](../docs/FORMAL_CHECKPOINT_EVALUATION_20261112.md)。

## 新增版本

维护者在 `master` 上发布文件。在仓库根目录首次创建一个新版本：

```bash
cp -R policy/_template policy/my_policy_v2
```

随后把配套文件放进新目录，填写该目录的 `README.md` 和 `manifest.json`。把清单的 `status` 从 `template` 改成 `candidate` 或 `evaluated`；是否通过验收由 `evaluation.gate_passed` 单独记录。

代码版本要区分：

- `runtime_commit`：接收方可以据此运行这套文件的、已核对兼容的代码提交。
- `training_commit`：实际训练所基于的提交。训练时如果还有未提交代码，记录 `training_code_dirty: true`，同时提供冻结源码或 diff，并填写其相对路径。
- 不清楚的字段保留 `null` 并在说明中解释，确认后再交接。完整记录可避免把后来的上传提交误当作训练版本。

在所有实际文件准备齐全后生成校验清单；如有 `calibration.pt` 或额外归一化文件，将它们也加入命令：

```bash
cd policy/my_policy_v2
sha256sum policy.pt prior.pt env.yaml agent.yaml evaluation.json > SHA256SUMS
sha256sum -c SHA256SUMS
```

把相应 SHA-256 填入 `manifest.json`，并核对清单中没有遗留的占位内容。回到仓库根目录后提交：

```bash
cd ../..
git add policy/my_policy_v2
git commit -m "Add policy handoff v2"
git push origin master
```

也可以通过 GitHub 网页上传相同目录结构。版本发布后，后续升级新建目录，保留已有版本。

## 合作者如何更新

合作者先提交自己的本地改动，再将文件清单所在的 `master` 合入联合分支：

```bash
git switch integration/force
git fetch origin
git merge origin/master
```

检查代码兼容性、文件 SHA-256 和该版本说明，按运行命令显式选择 checkpoint。先做短程加载与评估，再决定是否继续训练。每次联合实验记录选用的策略目录、SMP 代码提交以及 force estimator 的代码和权重版本。

## 文件跟踪范围

`policy/.gitignore` 只在本目录中允许提交 `.pt/.pth/.ckpt`；仓库其他位置的训练产物仍按原忽略规则管理。完整训练数据、运行日志、缓存和视频继续放在实验存储中。

GitHub 普通 Git 仓库阻止单文件超过 100 MiB。较大的权重可使用 Git LFS 或外部存储，并在版本清单中记录下载位置与 SHA-256。参见 [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
