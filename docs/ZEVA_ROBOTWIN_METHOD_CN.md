# ZeVA–RoboTwin：基于已训练 PI0.5 的因果记忆增强方法

执行更新（2026-09-15 01:01北京时间）：固定末步配对评测控制器已在aigc28启动并通过资源、checkpoint、seed/指令清单与端口检查；当前处于模型服务启动阶段，尚无新成功率。仅迁移运行节点，方法、权重、任务、H50/H15和评测协议均未改变。

**最新实测（2026-09-14）：固定teacher新轮的1000步训练与5874决策终检均已完成。** 同噪声H15误差：原Base004500=0.01018073、新Base001000=0.01003096、新ZeVA001000 residual-off=0.01014233、on=0.01014354。ZeVA虽优于训练起点，却比同预算Base差1.1224%，残差开启也略差于关闭；本轮没有证明ZTE增量收益，不能把固定teacher包装成已有效的方法。仍保留预定末步做配对闭环，不加训、不挑checkpoint/倍率；尚无新成功率。下列较早的执行进度和数值为历史记录，以本段及[最新诊断](ZTEV2_POST_EVAL_DIAGNOSIS_20260912.md)为准。

诊断口径补充：当前 step4096 的四项已测 Stage1 辅助指标均达到配置参考线；不能把固定的 `probe_gate_passed=false` 解释成实测不合格。全5874决策诊断及四组同机消融均已完成：H15双残差相对关闭残差仅改善0.0591%，context-only改善0.07443%，prior-only退化0.00500%，prior×50改善0.04983%、低于原始双残差。当前权重的离线收益主要来自context，简单增大prior门控未解决问题；差异很小，未证明统计显著或ZTE无效。这些不是成功率，新训练进展见下段。[完整数值、限制与下一步](ZTEV2_POST_EVAL_DIAGNOSIS_20260912.md)。

最新执行口径（2026-09-14 01:40核查，两支1000步均已完成）：普通Base和ZeVA均从训练好的Base004500出发，各训练1000新optimizer steps，固定比较末步；ZeVA新建零残差adapter，并以同一004500作immutable teacher。Base于01:24:59、ZeVA于01:30:46完成（北京时间），完整model/optimizer及ZeVA adapter均已保存。实际manifest确认两支global256、相同初始化；Base仅训练AE，ZeVA使用独立固定teacher，AE LR5e-6、新模块LR5e-5。仍冻结ZTE/bank/VLM，保留原双残差/Gaussian NLL和H50输出/H15执行。不是恢复旧ZeVA；不加载其005000 adapter。真实compiled teacher独立性、零残差等价和H15递归smoke均通过。现有preservation hinge只是对较差样本的专家标签损失重加权，不是动作蒸馏，不保证保能力。末步同样本/同噪声H15/H50诊断已启动，分别对照固定004500和本轮Base001000；尚无新的成功率结论。

最新完整结果（2026-09-12）：Base109/200=54.5%，ZeVA110/200=55.0%，Anchor101/200=50.5%。600个视频及配对seed/指令校验通过；ZeVA仅多成功1次，paired 95% CI为[-7.5,+8.5] pp，且Base未过57%参考线，故验收未通过。原轮训练与正式评测均已完成；以下早期运行状态均为历史快照。[完整十任务表与限制](ROBOTWIN_ZTEV2_PAIRED_RESULTS_20260912.md)。

> 文档状态：ZTE v2 实用接入主线与历史实验证据（2026-09-12，按用户最新要求修订）
> 目标：在已经完成 RoboTwin 训练的 PI0.5 上加入 ZeVA 的因果表征、记忆与检索能力，同时保持 PI0.5 的物理输入输出协议和基础能力。
> 主线定义：旧 v19 已停止。当前主线是三流 ZTE v2 经真实 H15 recurrent state 和 train-only bank 接入现有 action-expert 侧双残差/Gaussian prior，再做匹配的十任务 Base/ZeVA 训练和闭环比较。允许 ZeVA 方法变体；辅助表征指标用于诊断，不再要求全部通过后才接 PI。before-VLM prefix smoke 不是当前选定接入方式。下文历史实验及原研究门槛不覆盖本段最新执行口径。

当前 pipeline：Stage1 训练并选择 v2 checkpoint → 导出同源 bank/live queries、核对权重/协议/zero-init/H15 状态更新 → Stage2 冻结 ZTE/bank 和视觉语言 backbone，训练 action expert（`5e-6`）及 ZeVA 模块（`5e-5`），global batch 256 → 同 seed 配对闭环；只有发现需要针对性处理的缺陷才考虑 Stage3。Base 与 ZeVA 从指定 best-v1 出发，在相同十任务及匹配预算下训练；untouched best-v1 另作 Anchor。最终必须报告正常 Base 与 ZeVA 的真实成功率，不能通过弱化 Base 或挑测试 seed 制造优势。截至2026-09-12 13:30（北京时间），v2接入、修复版Stage1和全量产物已完成，Base完成5000步；ZeVA从SIGTERM前最后完整4500在空闲aigc24启动恢复（PID4082455），此刻仍在加载，尚未确认新step。恢复保留model/adapter/optimizer/scheduler、四卡及global256；数据副本全量哈希一致，路径迁移有显式证明，核心Torch/Transformers/Accelerate版本一致。Stage2未保存RNG且辅助库版本仍有差异，不能称为bit-exact续训；本轮训练及闭环交付尚未完成。

最新执行状态（2026-09-12 14:04，北京时间）：两支Stage2均已完成5000步，ZeVA在aigc24的恢复正常结束，完整模型/adapter/optimizer已保存。按预定held-out flow选择Base004500（0.01872689）与ZeVA005000（0.01828257），进入正式十任务×20同seed评测的preflight。以上覆盖前段13:30的加载状态；offline flow不能替代闭环成功率，本轮暂未产生正式成功率。

14:13更新：preflight通过，aigc24已启动Base→Anchor→ZeVA正式pipeline（PID4129168），模型server正在加载。每支10×20，从seed1000筛首20个expert-valid seeds，并共享精确seed/指令及H50/H15、Large_D435、seen randomized协议；结果目录为 `eval/formal-ztev2-selected-pair-20260912`。正常性须核对Base≥max(同seed Anchor,57%)，最终优势须实际ZeVA>Base；不声称此刻已有成功率。

## 1. 问题定义与设计原则

历史参考可比性补充（19:43）：历史Anchor114/200=57%与本轮协议语义一致，但不是同一seed/instruction manifest：200个task-seed重合181个，其中仅2个同时指令相同；model host也由aigc29变为aigc24。不能把历史57%与本轮Base54.5%的差值直接解释成训练退化。本轮Base、Anchor、ZeVA的冻结样本则严格一致，可作paired比较；保留既定57%参考门槛，不回头挑测试样本或权重。详细来源见实现验收记录。

19:02更新：同seed Anchor完整结果101/200=50.5%，视频与seed/指令校验通过；本轮Base109/200=54.5%，样本内高4个百分点。仍保留57%历史参考门槛并核查历史样本的可比性，不直接认定训练造成退化，也不宣称统计显著提升。ZeVA已开始评测，完整对比待完成。

截至2026-09-12 16:27，Base正式结果为109/200=54.5%，200个视频及精确seed/指令已核验；低于预设57%参考线，当前不能宣称正常性达标。Anchor正在评测，ZeVA随后运行，完整配对结果尚未产生。保持预定选步和种子不变，不用这一测试结果回头挑checkpoint。

当前这轮的选步口径：两支均完成5000步预算后，分别在完整保存的checkpoints中按 held-out validation **flow loss最低**选择，完全同分取较早step。不能用混有Gaussian NLL的ZeVA total loss代替共同指标，也不依据正式闭环成功率或ZeVA内部residual-off代理挑选最终pair。路径固定后再做同seed闭环比较；仅有offline flow改善不算交付成功。

RoboTwin 的基础策略已经是一个在相同 50-task 数据分布上训练完成的 PI0.5。我们的目标不是重新训练一个 VLA，而是在不破坏其已有能力的前提下，让策略利用执行动作之后真实发生的视觉变化，推断：

1. 当前任务是什么；
2. 当前处于任务的哪个因果阶段；
3. 已执行动作造成了什么效果；
4. 当前 episode/attempt 中哪些历史经验与下一步决策相关；
5. 如何把上述表征以可归因、zero-init 安全的方式接入 PI action expert。

因此，最终方案遵循四个原则：

- **PI0.5 是强基线**：untouched best-v1 是固定能力参照 Anchor。用户要求交付的 Base 与 ZeVA 均从该权重出发，在选定 10 个任务上做匹配训练；不得把较弱或不匹配的 checkpoint 选作 Base。增强在 zero-init 时必须数值等同其 PI 起点，联合微调后同时核对 Base 相对 Anchor 未异常退化及 ZeVA 相对 Base 的 paired 闭环收益。
- **因果状态必须来自真实 action-effect transition**：不使用帧编号、oracle progress 或测试成功标签构造部署时状态。
- **训练与部署使用相同递归过程**：从真实 `B0` 初始化，每执行 H15 后更新一次 Mamba、BIT 和 PIM。
- **用诊断定位问题、用闭环衡量收益**：correct/zero/shuffled 和分支消融帮助判断表征是否被利用，不强制完成全部研究实验才允许策略训练；数据泄漏、错误加载或递归协议错误仍必须先修复。

## 2. 与 BehaviorVLA 迁移方案的关系

本项目保留 BehaviorVLA“先学习独立表征，再接入基础策略，最后按需做策略适配”的分阶段思想，但将方法内容替换为 ZeVA 的因果学习口径：

| 项目 | BehaviorVLA 式思路 | 当前 ZeVA–PI0.5 方案 |
|---|---|---|
| 时序模块 | 原实现中的循环时序模块 | Mamba Causal Transition Encoder |
| 时序输入 | 历史视觉/行为上下文 | 严格的 `(s_t, a_{t:t+14}, s_{t+15})` action-effect transition |
| 初始状态 | 方法相关上下文初始化 | `B0 = F_init(g, s0)`，其中 `g` 是冻结 PI0.5 的任务语言表示 |
| 外部知识 | 方法相关 memory/retrieval | train95-only causal bank + task/phase retrieval |
| 在线记忆 | 历史上下文 | BIT（短期）+ PIM（持久） |
| 接入基础模型 | global prototype prefix + behavior prior | 当前复用 action-expert 侧 context/prior 双残差与 Gaussian prior；不是原文 prefix 的逐行复刻，不做最终动作 residual |
| Stage 2 | PI 与 PBD 联合训练 | 权重/zero-init/H15 smoke 后，以 action expert `5e-6`、新模块 `5e-5` 联合训练；视觉语言 backbone、ZTE/bank 冻结，global batch 256 |
| Stage 3 | 可选的针对性适配 | 仅在 Stage2 paired 闭环诊断明确指出具体缺陷后决定是否需要 |

这里最关键的改动是：保留 ZeVA 的 action-effect 表征和记忆增强目标，但不要求完全复刻论文接入方式，也不再启动 candidate medoid 搜索。训练和部署使用真实 H15 recurrent phase，禁止 `episode_index` oracle lookup；表征诊断不足时可针对性优化，但是否有价值最终由公平的 Base/ZeVA 闭环比较决定。

## 3. 冻结的 RoboTwin–PI0.5 契约

当前 handoff：

```text
/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
```

完整物理契约如下：

| 字段 | 固定设置 |
|---|---|
| 任务 | RoboTwin 50 tasks |
| 训练数据 | 每任务 50 Clean + 500 Randomized，共 27,500 episodes |
| 数据划分 | 固定 seed=1000 的 episode-level train95/validation5 |
| 状态 | absolute Joint14 |
| 相机 | head、left wrist、right wrist |
| 原始分辨率 | RGB 480×640，Large_D435 |
| PI 图像输入 | CHW float32 `[0,1]`，PI 内部再变换到 `[-1,1]` |
| 动作 | chunk-start-relative EEF16 |
| 动作排列 | left xyz+xyzw，right xyz+xyzw，left/right gripper |
| Gripper | slots 14/15，`1=open` |
| PI 预测长度 | H50 |
| 环境执行长度 | 每次只执行前 H15，然后重规划 |
| 归一化 | handoff 保存的 state/action mean-std；禁止重算 |
| PI 推理 | BF16，10 个 flow inference steps |

这里有两个不同的 PI 权重目录：

```text
checkpoint/pretrained_model
checkpoint/pretrained_model-best-v1
```

Stage1 的 B0 语言坐标、causal bank 与 retrieval 由 `pretrained_model` 对应的冻结语言初始化产生；当前 v14 的 Base 严格使用 `pretrained_model-best-v1`。两者模型 hash 不同，因此 v14 显式加载独立的 Stage1 language checkpoint 来保持既有语言坐标，不把 best-v1 的 embedding 偷换进已冻结的 Stage1。更换 PI-base 时的规则见第 16 节。

## 4. 方法总览（当前主线：ZTE v2 + 双通道接入）

v18 全任务 phase-gated consensus 已完成闭环开发验证：Base `43/80`、ZeVA `43/80`，其中 13 对只由 Base 成功、13 对只由 ZeVA 成功。这个结果否定了“离线 expert-action MSE 改善可以代理闭环收益”：即便 medoid 在 validation MSE 上改善 `13.76%` 且 10/10 任务为正，整体闭环收益仍为零。

v19 在完成新 split-k 前已停止，因为它仍在对一个未证明有效的旧 Stage1 表征做任务域门控，无法回答“ZTE 是否提供了更好的表征”。其 partial baseline 不进入任何成功率结论。以下旧 selector 结构保留为失败记录；当前完整规格以 `docs/ZTE_V2_SCIENTIFIC_DESIGN_CN.md` 为准。

```text
离线 Stage1
PI-base 冻结 task embedding + s0 ──► B0
(s_t, 已执行 EEF16 H15, s_{t+15}) ──► ZTE/Mamba 递归
                                      ├── phase / causal signal
                                      └── train95 bank / live queries / task retrieval
                                           在 Stage2 开始前全部冻结

当前 ZTE v2 与部署的因果条件链
(g, s0) ──► B0
(s_t, ordered H15, s_t+15) ──► 三流 causal Mamba
                                ├── z_proto / z_phase / e_causal
                                ├── M_t ──► zero-init causal-prompt prefix/KV
                                └── PBD(z_proto,z_phase) ──► Gaussian prior
三相机 + Joint14 + instruction + prompt ──► PI0.5 action expert
noisy action embedding + prior ──► flow denoising ──► EEF16 [50,16]
                                                        └── 执行 H15 后递归更新
```

2026-09-11 的当前监督路径对照：forward-effect 从执行前 `pre_context` 预测，严格隐藏当前 after-image；新增 `action_prediction_context=phase` 从实际导出的 normalized `z_phase(t)` 预测下一段 H15。后者在当前 H15 已执行完毕后才使用 `s_{t+15}`，不得读取下一动作或未来图像。旧 pilot 使用 `pre`，其 next-action loss 对 phase head 无梯度；新路径尚无能力验收结论。BehaviorVLA 官方代码的 action predictor 读取 shifted-action stream `h_a`，本变体不是其逐行复制。详细已执行证据见 `docs/ZTE_V2_IMPLEMENTATION_STATUS_CN.md`。

同日损失单位审计发现：BehaviorVLA 预测损失是坐标平方误差求和后取有效时间均值，而旧 v2 将所有坐标的 SmoothL1 取均值。新增 `vector_mse` 对照保留外部权重，只修正为沿效果特征/EEF16 坐标求和，沿有效 transition/H15 平均。256 维效果预测的小误差梯度因此恢复到原 reduction 对应的尺度；这不是闭环收益证明。后续以同初始化、同 4096-step 预算的旧 reduction / vector-MSE 学习曲线检验，而不是按测试成功率搜索损失权重。

**以下仅描述已停止的 v18 selector，不属于上图当前方法：** candidate-0 在每次 replan 先消耗正常 Base RNG；保存其后的 RNG 状态，再切到独立 proposal RNG 生成三个额外候选，最后恢复 Base RNG。故选择 candidate-0 时，本次动作与未来 Base 随机流都严格一致。medoid 是四个候选中到其余候选平均 H15 距离最小的实际样本，不做坐标平均。`0.85` 来自 train phase-bank 最大余弦相似度的第 1 百分位，在闭环结果出现前固定。离线指标只决定能否进入闭环，最终唯一判据是相同 seed/instruction/RNG 下的成功率。

旧表示审计已经完成：11,679 个决策样本来自 4,529 个 episode group，使用按 task/episode 分组且无 group overlap 的 5-fold。action-only AUC 为 `0.5563`，旧 Stage1+action 为 `0.5416`，PI-VLM+action 为 `0.5473`，联合表征为 `0.5375`；四种 selector 的 expert-MSE 都劣于 candidate-0。结论是 **discard old Stage1 for candidate ranking**，不是否定 ZTE。这个反例现在成为 ZTE v2 必须超过的 frozen baseline。下文 v11--v19 的训练公式、残差校准和历史目录仅保留作失败分析；凡与新规格冲突者均不是当前方法。

方法包含六个核心部件：

1. **ZTE / Causal Transition Encoder**：使用 Mamba 编码 action-effect transition。
2. **离线 causal bank**：只从 train95 专家轨迹构建的只读任务/阶段因果知识库。
3. **Task-language retrieval**：使用 PI-base 的冻结任务语言坐标选择 task prototype，不使用 `episode_index` 或 oracle task ID。
4. **BIT（Brief Interaction Trace）**：保存当前 attempt 的近期因果证据。
5. **PIM（Persistent Interaction Memory）**：合并同一 episode 内跨 attempt 的相似阶段证据。
6. **双通道接入**：global causal prompt 进入 action expert 可读的 KV 上下文，phase-conditioned Gaussian prior 进入 noisy-action embedding；禁止最终动作 residual。

Stage3 不是固定必训阶段。Stage2 训完后先使用同一批环境 seed、instruction 和固定的模型 RNG 起点做 paired PI/ZeVA 闭环评测；每个 condition 内的 flow RNG 按 handoff 口径连续消耗。只有 Stage2 不回退但收益仍不足时，才进入可选 Stage3。

## 5. B0 与任务语言

对于任务文本 `l`，使用冻结 PI0.5 tokenizer 和 language embedding table 计算任务语言表示：

```text
g = MeanPool(EmbeddingPI0.5(Tokenize(normalize(l) + "\n")))
```

在当前 RoboTwin 实现中，Stage1 的 canonical B0 使用规范化任务名，例如：

```text
beat_block_hammer -> "beat block hammer\n"
```

初始三相机图像 `s0` 经独立 resize、共享 ImageNet-pretrained ResNet-18 和 view fusion 得到视觉状态，随后：

```text
B0 = F_init(g, Vision(s0))
```

部署时自然语言 instruction 首先通过 task-language retrieval 映射到 50-task canonical table，再使用对应 canonical `g` 初始化 ZTE。这保证 Stage1 和部署的 ZTE 初始语义空间一致；原始自然语言本身仍进入 PI0.5 和 ZeVA task-schema projector。

## 6. Stage 1：学习 ZeVA Causal Transition Encoder

### 6.1 输入与递归边界

在每个完整专家 episode 中，从 frame 0 开始以 H15 连续递归。第 `k` 个 transition 为：

```text
T_k = (s_{15k}, a_{15k+1:15k+15}, s_{15k+15})
```

其中：

- `s` 是三相机 RGB 观察；
- `a` 是相对 transition 起点的 EEF16；
- 动作使用 handoff 的冻结 mean-std 归一化；
- 每个 transition 的输出描述动作完成后的状态；
- 不把动作先输入 Mamba、再让同一个状态预测该动作，避免 label leakage。

### 6.2 ZTE 输出

ZTE 对每个 transition 输出：

- `phase_token z_k`：局部任务阶段表示；
- `causal_signal c_k`：动作导致的因果变化；
- `phase_progress p_k`：单调任务进度；
- `global_task_embedding u`：episode 级任务坐标；
- `predicted_effect`：预测视觉特征变化；
- `predicted_action`：从 transition 前的因果状态重建 H15 动作。

视觉 effect target 使用同一冻结 EMA 视觉空间：

```text
delta_s = EMA_Vision(s_after) - EMA_Vision(s_before)
```

这样不会把 online/target encoder 或 BatchNorm 漂移误认为物理变化。

### 6.3 Stage1 损失

```text
L_stage1 =
    λ_effect      L_effect_prediction
  + λ_action      L_action_reconstruction
  + λ_task        L_task_contrastive
  + λ_task_proto  L_task_prototype
  + λ_phase       L_phase_progress
  + λ_phase_key   L_phase_key_alignment
  + λ_mono        L_phase_monotonic
```

当前主要权重：

| 项 | 权重 |
|---|---:|
| effect | 1.0 |
| action | 0.1 |
| task contrastive | 2.0 |
| task prototype | 0.1 |
| phase progress | 1.0 |
| phase key | 1.0 |
| monotonic | 0.2 |

任务对比学习使用跨 rank、同任务不同 episode 的正样本；task-paired distributed sampler 保证每个全局 batch 存在可用正样本。阶段标量通过正 hazard 累积，因此 `p0=0` 且天然单调；phase key 使用非周期 Gaussian RBF 坐标，避免任务头尾发生环绕混淆。

### 6.4 Stage1 训练设置

| 设置 | 当前值 |
|---|---:|
| steps | 30,000 |
| GPU | 8×H100 |
| 每 GPU batch | 1 个完整 episode sequence |
| global batch | 8 episodes |
| transition/effect/executed horizon | 15/15/15 |
| optimizer LR | 1e-4 |
| vision LR | 1e-5 |
| warmup | 1,000 steps |
| Mamba layers | 4 |
| model/phase/signal dim | 256/128/256 |
| vision init | ImageNet-1K ResNet-18 |

### 6.5 Stage1 通过标准与当前结果

Stage1 必须在完整 validation5 episode 上同时通过：

- global task macro R@1 ≥ 95%；
- phase-end MAE ≤ 0.10；
- phase-end Spearman ≥ 0.80；
- monotonic violation ≤ 5%；
- bank phase-end MAE ≤ 0.10；
- initial B0 bank MAE ≤ 0.05；
- action/effect 相对 zero baseline 均提升至少 20%；
- offline/stateful latent parity max error ≤ `5e-3`；
- auxiliary output parity max error ≤ `2e-2`。

当前 step 30000 已通过 gate：

| 指标 | 结果 |
|---|---:|
| task macro R@1 | 99.926% |
| phase-end MAE | 0.09898 |
| phase-end Spearman | 0.88186 |
| phase monotonic violation | 0% |
| bank phase-end MAE | 0.05303 |
| action zero-baseline improvement | 60.83% |
| effect zero-baseline improvement | 34.70% |

## 7. Stage 1.5：构建部署所需的冻结检索产物

Stage1 结束后冻结 ZTE，并生成三个部署前产物。

### 7.1 Train95 causal bank

只使用 train95 专家 episode，从真实 B0 连续回放整个 episode。对每个任务保存：

- episode-global task prototype；
- 32-bin phase key；
- 对应 causal value；
- 每个 episode 的显式 B0 phase entry；
- 每 episode 4 个分层采样 causal outputs。

稀疏保存只减少 bank 条目数量，不截断 Mamba 历史。该 bank 是冻结的先验知识，不是在线 PIM。

### 7.2 H15 live-query cache

再次从 B0 连续回放完整 train/validation episode，在每个真实 H15 decision boundary 保存：

- live phase query；
- preceding causal signals；
- decision frame index。

Stage2 使用这一 cache 重建与部署一致的 BIT/PIM 历史，禁止使用 frame progress 代替 live phase。

### 7.3 Task-language retrieval

训练一个小型 retrieval head，把冻结 PI0.5 的自然任务语言 embedding 映射到 train95 causal bank 的 50 个 task prototype：

```text
instruction -> PI0.5 frozen embedding -> retrieval head -> task prototype ID
```

真实 task ID 只用于计算准确率，不用于部署检索。当前 train/validation accuracy 分别为 99.763% 和 99.704%，超过 95% gate。

## 8. 在线记忆：BIT 与 PIM

### 8.1 BIT

BIT 保存当前 attempt 中最近的因果信号，容量为 8。它主要反映刚刚发生的动作效果和局部错误恢复信息。

### 8.2 PIM

PIM 容量为 256，用于保存同一 episode 中更持久的经验。新 causal signal 与已有条目根据 phase 相似度和 signal 相似度进行合并：

```text
score_merge = 0.6 * sim_phase + 0.4 * sim_signal
```

合并阈值为 0.85，检索 top-k 为 5。

### 8.3 Reset 语义

- `reset(scope="attempt")`：清空 Mamba recurrence 和 BIT，保留同 episode 的 PIM；
- `reset(scope="episode")`：清空 Mamba、BIT、PIM 和 task retrieval 状态；
- 模型参数在部署期间始终冻结；
- 当前正式评测每个环境 seed 都执行 episode reset。

## 9. Stage 2：从正常 Base 初始化，冻结完整 PI0.5 训练 ZeVA

### 9.1 冻结与可训练参数

冻结：

- PI0.5 PaliGemma vision tower；
- PI0.5 PaliGemma language backbone；
- PI0.5 Gemma action expert；
- PI0.5 action input/output projection 与 time MLP；
- Stage1 ZTE/Mamba；
- train95 causal bank；
- task-language retrieval head。

ZeVA adapter 训练：

- task token projector；
- memory context encoder；
- normalized EEF16 action prior；
- causal-context action projector；
- prior action projector；
- context/prior sigmoid gates。
- 由 task language schema 与真实 recurrent H15 phase 驱动的两路 gate router。

两个注入 projector 零初始化；两个 scalar gate 的初始概率为 0.25。router 输出乘数被限制到 `[0,2]` 且零参数初始化对应乘数 1，可对有害任务/阶段压到接近 0，又不会无界放大。关闭 ZeVA 或在初始状态下，wrapper 对相同随机种子必须与 Base step3000 bit-identical。

ZeVA fusion/action-prior/router 使用 `5e-5`。首个 backward 后断言所需 ZeVA 模块有梯度，而完整 PI0.5、ZTE 与 retrieval 均无梯度。梯度仍需穿过冻结 action expert 才能到达注入输入，因此真实 dropout/layer 保持 eval，同时只打开 released Gemma checkpoint scheduler 所依赖的两个 parent training flags；否则显存与速度会异常。

### 9.2 上下文与注入

在 decision boundary `k`：

1. 用自然语言预测 task prototype；
2. 用 live ZTE 得到当前 phase；
3. 从冻结 bank 取 task/phase-matched causal values；
4. 从 BIT/PIM 取当前 episode 的在线证据；
5. 将 task schema、phase、offline memory 和 online memory 编码为 `causal_context`；
6. 把 `causal_context` 投影到 action-expert width，并广播到 H50 noisy-action tokens；
7. 生成 normalized EEF16 H50 对角 Gaussian prior `(mu, log_sigma)`，用 NLL 监督；
8. 将每步 `mu` 投影为另一条残差；两条 residual 都只加到 PI0.5 每轮 flow 的 noisy-action embedding，绝不进入 PaliGemma；训练时以 0.4 概率丢弃整段 prior residual；
9. 未丢弃时，注入强度继续乘以 retrieval confidence、learned scalar gate，以及由 task-language schema 和当前 H15 phase 产生的 `[0,2]` 有界 router multiplier。

训练保留 handoff 原始的 PaliGemma/action-expert 联合 attention forward；完整 PI0.5 参数 `requires_grad=False`，autograd 只穿过 action expert 更新 ZeVA 注入支路。该设计不增加 PI token 数，不改变 attention mask、position ID、prefix length 或 action-expert tensor shape。

### 9.3 Stage2 损失

```text
L_stage2 =
    L_PI_flow
  + λ_prior    L_Gaussian_NLL
  + λ_preserve mean_i max(0, L_PI_flow_i - L_residual_off_i + 5e-5)
  + λ_gate     L_gate
```

`L_Gaussian_NLL` 对 EEF16 维度求和，再对 batch 和 H50 求平均；`log_sigma` 限制在 `[-5,2]`。`L_residual_off_i` 使用冻结的 Base step3000 action expert、关闭 ZeVA residual，并与 enhanced forward 对每个样本共享完全相同的 flow noise。保护 hinge 带正 margin `5e-5`，在逐样本层面计算，禁止某一任务的改善抵消另一任务的退化。每 2 个 optimizer step 计算一次 matched teacher，并把该步 preservation 项乘 2；validation 逐 batch计算 matched teacher，同时报告 paired win fraction 与平均正退化。训练仍使用 phase noise 0.02、whole-memory dropout 0.1 和独立 prior-residual dropout 0.4（keep=0.6）。

### 9.4 Stage2 训练设置

| 设置 | 当前值 |
|---|---:|
| steps | 2,000 |
| GPU | 8×H100 |
| 每 GPU micro-batch | 16 |
| gradient accumulation | 2 |
| effective global batch | 256 |
| video backend | TorchCodec，32-entry/worker decoder LRU |
| matched teacher | 每 2 step 一次，逐样本 hinge，采样步 ×2；validation 全量覆盖十任务 |
| foundation forward | `torch.compile(mode="default")`，hook 安装后编译 |
| PI0.5 action expert LR | 冻结，不进入 optimizer |
| ZeVA LR | 5e-5 |
| warmup | 250 |
| Gaussian NLL weight | 0.01 |
| preservation weight | 8.0 |
| positive paired margin | 5e-5 |
| gate regularization | 1e-5 |
| phase noise | 0.02 |
| memory dropout | 0.1 |
| prior residual dropout | 0.4（keep=0.6） |

batch 16、accumulation 2、8 张 H100 保持 global batch 256。每个 checkpoint 保存完整 `model.safetensors`、`zeva_adapter.pth` 和 optimizer/scheduler state。审计要求完整 PI0.5 与 Base step3000 位级一致，并确认八个 ZeVA 模块进入 optimizer；action path 不允许更新。

### 9.5 Stage2 通过标准

Stage2 的离线 gate 应同时检查：

- task retrieval accuracy ≥ 95%；
- enhanced validation mean flow 优于 matched frozen baseline；
- validation 逐样本 paired win fraction ≥ 50%，并优先最小化平均正退化；
- ZeVA-off 路径严格复现 baseline action；
- 所有 ZeVA fusion/action-prior/projector/gate/router 模块均有梯度；
- 完整 PI0.5、ZTE/Mamba 与 task retrieval 无梯度；
- train/eval 图像进入 PI 前均为 CHW float32 `[0,1]`；
- 最终必须通过同环境 seed、同 instruction、同模型 RNG 起点和 continuous 口径的 paired closed-loop rollout。

最后两项是当前需要补强的关键 gate；仅凭 offline loss 不能选择最终 checkpoint。

## 10. Stage 3：可选的 selective action adaptation

Stage3 不是必经阶段。只有在 Stage2 已完成 paired baseline/ZeVA 闭环评测后才决定是否进入。

建议冻结：

- PI0.5 VLM/vision-language prefix；
- ZTE；
- causal bank；
- retrieval head；
- 已学到的 ZeVA memory representation。

只解冻：

- PI0.5 action expert 的最后若干 block，或最终 action projection；
- 必要时连同 Stage2 residual 做极小学习率联合调整。

Stage3 学习率应比 Stage2 小 10–20 倍，继续保留对 Stage2 anchor 的 preservation loss，并根据 paired rollout early stop。

当前仓库中的旧 `train_robotwin_stage3.py` 是历史 VLM-retrieval 方案；task retrieval 已移动到 Stage1.5，因此它不属于本文定义的最终 Stage3。

## 11. 部署状态机

一次 episode 的严格流程为：

```text
1. reset_episode()
2. 获取 instruction 和初始三相机观察 s0
3. task retrieval；B0 = F_init(g_task, s0)
4. 检索 offline task/phase memory，融合空 BIT/PIM
5. PI0.5 + ZeVA 预测 chunk A0，shape=[50,16]
6. 将 chunk-start-relative camera-frame EEF16 转成 RoboTwin world absolute target
7. 执行 A0[0:15]
8. 获取 s15
9. ZTE.step(s0, executed_A0[0:15], s15)
10. 更新 BIT/PIM
11. 重新检索并预测下一段 H50
12. 重复 6–11，直到成功或达到 step limit
```

只提交环境实际执行完成的完整 H15 transition。成功中途停止、或 episode 已结束时，不再生成虚假的 effect transition。

## 12. 训练—部署一致性与防泄漏规则

必须保持：

- 相同的 camera key、顺序、分辨率、RGB 通道和像素范围；
- 相同的 Joint14 / EEF16 slot 语义；
- 相同的 mean-std artifact；
- 相同的 H15 transition 和 H50 policy chunk；
- 相同的 canonical B0 task table；
- 相同的 stateful Mamba recurrence；
- 相同的 bank/BIT/PIM 检索与合并逻辑；
- 相同的 task-language retrieval 与置信度门控。

禁止：

- 把 validation/test episode 写入 train causal bank；
- 使用测试 success label、oracle task ID 或 oracle progress；
- 从随机独立 frame 拼接伪造 Mamba 历史；
- 在 episode 之间保留 PIM；
- 在评测期间更新模型参数；
- 在未重建下游产物时更换 PI checkpoint、tokenizer 或 normalization statistics。

## 13. 正式 RoboTwin 评测协议

当前目标评测协议：

| 字段 | 设置 |
|---|---|
| Tasks | 当前专项模型为固定 10 任务（完整 benchmark 另为 50 任务） |
| Scene | Hard / Randomized |
| Camera | Large_D435，640×480，三视角 |
| Instruction | seen |
| Episode | 每任务 20，当前专项评测共 200 |
| Seeds | 从 1000 起筛选每任务前 20 个 expert-valid seeds |
| Model output | EEF16 `[50,16]` |
| Execution | 前 15 步，H15 重规划 |
| RNG | `model_rng_seed=20260907` 固定每个 condition 的进程初始 RNG；随后 continuous，episode 不用环境 seed 重置 flow RNG |
| Reset | 每 episode 清空全部在线 causal state |
| Validation | 每任务必须有 20 个视频，视频标签与 progress/summary 一致 |

用于模型对比时，动态筛 seed 只能执行一次。随后必须保存：

```text
task -> [(seed, instruction), ...]
```

所有实际比较条件必须重放完全相同的环境 seed 和 instruction，并从相同的模型 RNG 起点启动。当时的 anchored-v9 最终实验包含三种语义条件：untouched best-v1 `Anchor`、固定的 action-expert Base step3000 `Base`、以及从该 Base 初始化的双残差 `ZeVA`。由于不同策略可能在每个 episode 使用不同数量的 H15 replans，flow RNG 在轨迹分叉后自然连续消耗而不再逐采样强行对齐；禁止用环境 seed 每 episode 重置来改变 frozen handoff 的策略分布。20 episodes/task 可作为本轮正式表格，但小于 1–2 个百分点的差异不应在没有逐 episode paired 结果时解释为真实提升或回退。

RoboTwin 的 GPU physics 不是位级确定的，因此在 Base 中已经通过 expert filtering 的 seed，换到 Anchor/ZeVA 进程初始化同一场景时仍可能偶发 `UnStableError`。fixed-manifest 条件下必须在任何 policy inference 之前原样重试同一个 `(seed, instruction)`，最多 20 次；严禁递增 seed 或另选替代 seed。此类初始化重试发生在模型调用前，不消耗 continuous diffusion RNG。若客户端进程因此退出，恢复时保留同一模型服务和端口，从 progress 的同一 frozen seed 继续；不得重启模型服务后跳过已完成 episode。

### 十任务专项模型的本轮验收

`configs/robotwin_zeva_advantage10.json` 定义的十任务模型不使用 50-task 均值冒充完整 benchmark。v7 step1750 仅由 train95/validation5 选出，随后在 seed5000 与 seed6000 两个互斥的闭环 split 上分别得到 `Base 47/80 vs ZeVA 44/80` 和 `Base 44/80 vs ZeVA 41/80`，合计 `-6/160`；因此 v7 已拒绝，未启动 seed10000 正式测试。

v8 的两支 5,000-step 训练已经结束，但没有 checkpoint 同时满足 paired improvement、胜率和十任务逐项不退化，因此 `eligible_count=0`。low-LR joint v9 固定 v8 Base step3000 并训练 action expert + ZeVA，两个新闭环 split 合计 Base `84/160`、ZeVA `89/160`，只提升 `+5`，低于预注册 `+6`，已拒绝且没有触碰 final seed10000。

当前 frozen-PI adapter 仍从 Base step3000 精确初始化，但冻结完整 PI0.5，只训练 ZeVA。它完成 2,000 global steps、每 250 steps 保存，八个 checkpoint 均包含完整模型、adapter 与 optimizer state。validation5-only 选择 step1250：5 个 checkpoint 合格，选中点 aggregate paired improvement `+3.386e-5`、win fraction `58.83%`、retrieval `99.85%`、最差任务 improvement `+2.199e-5`，十任务均为正。新的闭环验证使用与 5000/6000/7000/8000/10000 均不重叠的 seed9000 和 seed12000；每 split 为 10×8，要求各自 delta≥0 且合计至少 `+6/160`。协议仍为 Large_D435 640×480、Joint14、EEF16 H50 输出/H15 执行、seen instruction 和逐 episode 视频。扩散 RNG 设为 `20260907` 后在每个 condition 内连续消耗；episode reset 只清 recurrent/causal state。历史 untouched best-v1 的 `114/200=57.0%` 仍是硬下限，最终门槛固定为 `Base>=max(同 seed Anchor,57%)` 且 `ZeVA>Base`。parent foundation SHA256 必须是 `7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe`。

该预注册闭环门槛已经通过：seed9000 的 split-e 为 `Base 43/80、ZeVA 43/80`（差值 `0`），seed12000 的 split-f 为 `Base 39/80、ZeVA 46/80`（差值 `+7`）；两组 seed 逐任务互斥，合计为 `Base 82/160、ZeVA 89/160`，即 `+7/160`。因此 step1250 获准进入保留的 seed10000 正式 10×20 测试。正式测试只作最终报告，不参与 checkpoint 选择；在三组 condition 和视频审计全部完成前，不报告运行中的成功率为最终结果。

当前训练命令与输出：

```bash
bash scripts/train_robotwin_advantage10_anchored_v9.sh adapter
# candidate: .../advantage10-anchored-v9/adapter
# fixed Base: .../advantage10-action-expert-v8/baseline/003000
python3 scripts/select_robotwin_anchored_v9.py \
  /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-anchored-v9/adapter \
  /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-action-expert-v8/baseline/003000
bash scripts/robotwin_eval/launch_anchored_v9_validation.sh
bash scripts/robotwin_eval/launch_anchored_v9_fresh_final.sh
# 持久协调器：训练完成后依次执行离线选步、双 split 验证和满足门槛后的 final
bash scripts/robotwin_eval/wait_and_run_anchored_v9.sh
```

选择器固定 Base step3000，只在 adapter checkpoints 中选择；要求十任务 validation 覆盖完整、retrieval≥95%、ZeVA 相对 exact residual-off Base 的 paired improvement>0、win fraction≥50%，并且十任务各自 mean improvement 都不为负。合格候选先最大化最差任务 improvement，再比较 aggregate improvement、win fraction 与 ZeVA flow；闭环结果不参与选 step。最终选择已冻结为 step1250，后续 seed9000/12000 闭环只决定该 checkpoint 能否进入 final，不能反向挑选其他 step。

三任务、每任务 5 次的 checkpoint gate 只用于发现加载错误和明显闭环退化，不作为统计验收：在固定模型初始 RNG 后，同一 Anchor 因 RoboTwin GPU physics 非位级确定而在重复 gate 中出现 `10/15` 到 `13/15` 的波动。第一轮闭环筛选保留 action-expert baseline step `1000` 和 ZeVA step `250`；诊断目录为：

```text
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5/formal-b1000-z250-seeded-v1
```

该目录的 Base 得到 `99/200 = 49.5%`，而同协议 untouched PI anchor 得到 `114/200 = 57.0%`。因此 Base-1000 已被正式判定为异常退化，不满足 `Base >= Anchor`；即使该目录中的 ZeVA 后续完成也只能用于诊断，不能作为最终对比交付。中途成功率和 15-episode gate 均不得写成最终结果。

替代方案不把 untouched anchor 改名为 Base，而是重新做等预算、低扰动的正常微调。`scripts/train_robotwin_advantage10_conservative_variant.sh` 为 Base 与 ZeVA 都采用 frozen PaliGemma、action-expert LR `1e-7`、global batch 256（每卡 16、累积 2、8 卡）、250 optimizer steps、250-step warmup，并在 125/250 保存完整 checkpoint。ZeVA 分支额外以 LR `5e-5` 训练双残差与 Gaussian-NLL action prior，prior residual dropout 为 0.4；Stage1 ZTE/Mamba、causal bank、H15 live queries、任务语言表及 retrieval 全部冻结。离线验证选择 Base step 125（held-out flow 最优）和 ZeVA step 250（prior 已成熟且注入分支 flow 优于相同权重的 injection-off 分支）。

第一组保守候选的正式目录为：

```text
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5/formal-conservative-b125-z250-seeded-v1
```

它原样复用第一轮冻结的 10×20 `(seed, instruction)` manifest，并使用完全相同的相机、动作、H50/H15、continuous model RNG 与视频协议。该 Base 在完成 194 条时为 107 次成功；即使余下 6 条全部成功，理论上限也只有 `113/200`，低于现有同协议 untouched PI anchor 的 `114/200`。因此该候选已在数学上确定无法满足门槛后停止，状态标记为 `rejected_early`，194 条结果只保留作诊断，没有浪费计算继续跑其 Anchor/ZeVA。

下一组匹配候选只把两支 action-expert LR 继续降为 `5e-8`，其他设置完全不变。离线验证选择 Base step 125（flow `0.020620564`）和 ZeVA step 250（注入 flow `0.021710902`，相同权重关闭注入为 `0.021731101`，retrieval `99.7803%`）。完整 checkpoint 缓存于 aigc32 的 `/data1/dingxin/zeva-checkpoint-cache-ae5e-8-v1`，正式目录为：

```text
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5/formal-conservative-ae5e8-b125-z250-seeded-v1
```

该目录仍须三路各 200 个 episode、600 个视频、return code、seed/instruction 和报告全部通过独立审计，且 `Base >= Anchor`、`ZeVA > Base` 后才可交付。

该 `5e-8` Base 也已被严格淘汰：运行至 `187/200` 时只有 `99` 次成功，即使余下 13 条全部成功也只能达到 `112/200`，低于已经建立的正常 PI 结果 `114/200`。因此没有继续浪费计算运行它的 Anchor 和 ZeVA，目录保留为 `rejected_early` 诊断证据。这说明仅继续减小 action-expert LR，仍不足以可靠保持已经在 RoboTwin 上训练好的 PI 闭环能力。

下一候选对两个匹配分支采用相同的 anchor-preserving post-training merge。闭环评测开始前固定 `alpha=0.25`：每个 action-path 权重为 `0.75 × 原始 PI + 0.25 × 已微调权重`；ZeVA adapter 保持完整，不参与插值。`scripts/interpolate_robotwin_action_expert.py` 只允许 209 个 action-path 张量插值，并逐一验证 604 个冻结 PaliGemma 张量位级完全不变。这个方法没有把 untouched Anchor 改名为 Base：Base 和 ZeVA 都保留同尺度的训练位移，区别仍只是 ZeVA 的双残差、Gaussian prior 与 causal memory。

新正式目录为：

```text
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5/formal-wise025-b125-z250-seeded-v1
```

该轮除重新评测 Anchor 外，还把既有正常 PI 下限 `0.57` 直接写入 manifest、acceptance 和独立 audit。最终门槛因此是 `Base >= max(本轮 Anchor, 0.57)` 且 `ZeVA > Base`，不能因为 GPU physics 导致本轮 Anchor 偶然偏低而放宽用户要求。

该插值候选也被闭环严格淘汰：Base 运行至 `199/200` 时为 `101` 次成功，最终理论上限只有 `102/200`。这说明早期 250--1000 step 的短预算 action-expert 候选不能代表正常微调，并促成了后续 frozen-PI v7；但 v7 在两个新 split 上同样得到 `-6/160`。随后 v8 从同一 best-v1 独立分叉并完成 5,000-step 等预算训练，但 ZeVA 仍没有 checkpoint 通过独立配对门槛。当时的 anchored-v9 因此固定 v8 Base step3000，以它同时作为 ZeVA 精确初始化和独立 teacher，并继续用 untouched Anchor 双重约束正常能力。

第一轮完整冻结 PI 的 scalar-gate adapter 已按预注册协议完成：`Base=114/200=57.0%`，`ZeVA=110/200=55.0%`，paired delta 为 -2 个百分点（ZeVA-only 31，Base-only 35，exact McNemar p=0.712）。400 个条件视频和结构审计均通过，因此是模型失败而非协议失败，只能作为诊断保留，不能交付。诊断显示原
preservation hinge 先对 batch 求均值，会允许跨任务正负抵消；同时所有任务
共享同一对 scalar gates。当前修正版 Stage2 使用：

```bash
bash scripts/train_robotwin_advantage10_safe_router.sh
```

训练 1500 optimizer steps，global batch 256（每卡 16、累积 2、8 卡），warmup 250，ZeVA LR `5e-5`，Gaussian NLL 权重 `0.01`，action-prior residual dropout `0.4`。每 2 步计算一次逐样本 matched frozen-PI hinge，权重为 4；gate regularization 为 `1e-2`。保留 TorchCodec、compiled H50/H15 和真实 recurrent phase。冻结项包括完整 PI0.5、ZTE/Mamba、causal bank、task retrieval；optimizer 只包含 task projector、memory encoder、Gaussian prior、两路 action residual projector、scalar gates 与有界 task/phase router。只允许根据 validation5 的 250-step checkpoints 选择，禁止根据闭环测试挑 step。validation 设置 64 个 global batches，覆盖完整 5874 个十任务 validation decisions；入选先要求十任务覆盖完整、retrieval≥95%、十个任务各自的 mean paired improvement 均不为负、aggregate improvement>0、逐样本 win fraction≥50%；再依次最大化 win fraction、最大化最差任务 improvement、最小化平均正退化、最大化 aggregate improvement、最小化 prior NLL、选择较早 step。

等待训练并自动启动两路 paired 评测（Base 为训练前已登记的 untouched PI 证据）：

```bash
bash scripts/robotwin_eval/launch_frozen_adapter_formal_eval.sh
```

safe-router v3 最终完整结果为 `Base=114/200=57.0%`、
`ZeVA=112/200=56.0%`，ZeVA-only/Base-only 为 `33/35`，paired bootstrap
95% CI 为 `[-0.09, 0.07]`，exact McNemar `p=0.904`。独立结构审计通过：
两边各 200 个 episode、每任务 20 个视频，seed、instruction、相机、动作和
H50/H15 协议全部一致。因此这是模型选择代理失效，不是评测错位：逐样本
offline flow non-regression 仍不能保证 closed-loop success。

修正版 v4 在 Stage2 后增加 **独立闭环 residual trust calibration**。它不改变
PI0.5、ZTE/Mamba、causal bank、retrieval、两路已学习 residual 或 H15 recurrent
state，只让 task-language 检索出的任务为 context/prior 两路 residual 共同选择
一个保守乘数。候选预先固定为 `0.25/0.5/1.0`，在从 seed 2000 开始、与正式
seed 完全不重叠的每任务 8 个 expert-valid episodes 上比较；只有某任务的最佳
候选比 paired Base 至少多 2 次成功时才启用，候选并列时选择更小 scale，否则
scale=0，严格回退 untouched PI。校准器拒绝读取正式测试结果，并在 adapter 中
记录 validation/formal seed manifest 哈希及 `test_metrics_used=false`。执行入口：

```bash
bash scripts/robotwin_eval/launch_closed_loop_residual_calibration_v4.sh
```

该流程只计算一次 validation Base，三个 scale 严格复用相同 seed 和 instruction；
校准表落盘后才会启动固定 10×20 正式评测。

v4 的实际正式结果未通过：单 split 校准选择了
`put_bottles_dustbin=0.5`、`stack_bowls_three=1.0`，其余任务为 0；固定正式
manifest 上 `Base=114/200=57.0%`，`ZeVA=100/200=50.0%`。两边各 200 个
视频、seed/instruction 精确配对及物理协议审计都通过，因此不能把失败归因于
评测错位。另对保存的 foundation 与 untouched PI0.5 做了 813 个 tensor 的逐一
`torch.equal` 审计，差异数为 0；文件 hash 不同只来自 safetensors metadata。
结论是单个 8-shot 闭环验证集方差过大，不能可靠决定任务级 residual trust。

当前 v5 改为 **双独立闭环验证集复现门槛**：split A 从 seed 2000 开始，split B
从 seed 3000 开始，每任务各筛 8 个 expert-valid episode；两个 split 与正式
seed-1000 manifest 逐任务两两无交集。每个 split 的 Base 只评一次，候选
`0.25/0.5/1.0` 严格复用各自 Base 的 seed 和 instruction。某任务某 scale 只有
在 A、B 两个 split 的 paired gain 都不小于 0、且合计至少 `+3/16` 时才有资格；
从合格候选中最大化总 gain，平局选更小 scale，否则回退 scale 0。选择器保存
全部 manifest hash、`seed_sets_pairwise_disjoint=true` 和
`test_metrics_used=false`。v5 启动器在生成 adapter 后停在
`calibration_complete_pending_audit`，不会自动消费正式测试。

```bash
bash scripts/robotwin_eval/launch_closed_loop_residual_calibration_v5.sh
```

v5 双 split 校准已于 2026-09-08 完成。split B 的固定 manifest SHA256 为
`8a088aa85bee0e757a0fd74b5808a9c67c3e0a20f75e4c614307be68bcf6cff2`；
paired Base 为 `43/80`，scale `0.25/0.5/1.0` 的 ZeVA 分别为
`47/80、47/80、45/80`。每个条件均有 80 个视频、10 个 worker `rc=0`，且
seed/instruction 逐条相同。只有 `put_bottles_dustbin` 的 scale `0.5` 在两个
split 上复现：split A `+3/8`、split B `+2/8`，合计 `+5/16`。最终校准表仅将
该任务设为 `0.5`，其余九任务及 default 均为 `0`；adapter SHA256 为
`8a12b0dbed2da7741aeff407e62d6eeaa676e958dc1eef1f772aafd8f95810cc`，并记录
`seed_sets_pairwise_disjoint=true`、`test_metrics_used=false`。人工审计后已在
`formal-calibrated-v5` 完成了固定 seed-1000 的 10×20 正式评测。最终
`Base=114/200=57.0%`，`ZeVA=109/200=54.5%`，下降 2.5 个百分点；
Base-only/ZeVA-only 为 `35/30`，exact McNemar `p=0.620`。独立审计确认
200 个 ZeVA 视频、10 个 worker `rc=0`、seed/instruction 逐条配对以及
H50/H15、相机和动作协议全部正确。因此 v5 是模型/选模失败，不是评测错位，
只能作为负结果保留。

同时新增不加载 PI0.5 的残差分支审计。它对十任务的平均 train-language embedding
与 causal bank 全部 phase bins 计算门值和注入后 RMS，只用于定位结构问题，不作为
成功率或选模指标。v3 adapter 的可复现实测为：context gate 均值 `0.02047` 且跨
任务几乎不变，prior gate 均值 `0.00803`；context residual RMS `0.004881`，prior
residual RMS `0.0000826`，前者是后者的 `59.1×`。因此若 v5 仍失败，下一步应
隔离 context/prior 分支并修复 prior 强度或训练，而不是继续盲目调整共同 scale。

v6 按上述预注册诊断拆分分支：context projector 精确清零，Gaussian action
prior 改为绝对 gate `0.5`，与 BehaviorVLA 推理 guidance 的量级一致；保留训练好
的 task/phase router、task-language 检索和真实 H15 recurrent phase，PI0.5、
ZTE/Mamba、causal bank 与 retrieval 仍全部冻结。两个互斥闭环验证集结果为：
split A `Base=50/80、candidate=50/80`，split B
`Base=43/80、candidate=45/80`。沿用“每个 split 非负且合计至少 +3”任务门槛，
仅启用 `beat_block_hammer`（`+1,+3`）和 `blocks_ranking_rgb`（`0,+3`）；
其余 8 个任务及 default 精确回退 PI0.5。校准 adapter SHA256 为
`28d99c1fd9092310acb6fd7a9040df619de45b773e6caa5feaa1d1caafc84709`，
metadata 记录 `seed_sets_pairwise_disjoint=true`、`test_metrics_used=false`。冻结
adapter 的第三个互斥 holdout split 得到 `Base=41/80、ZeVA=46/80`，但真正启用
residual 的两个任务只贡献 `+1`；其余 `+4` 来自 residual 关闭任务，只能解释为
GPU physics 波动，不能算作模型收益。固定 10×20 正式评测最终得到
`Base=114/200=57.0%`、`ZeVA=102/200=51.0%`，下降 6 个百分点；Base-only/
ZeVA-only 为 `44/32`，exact McNemar `p=0.207`，paired bootstrap 95% CI 为
`[-14.5,+2.5]` 个百分点。独立审计确认两侧各 200 episodes/视频、每任务 20 条、
seed/instruction 精确配对且无协议错误。因此 v6 已正式拒绝，是模型失败而不是
评测错位。

```bash
bash scripts/robotwin_eval/launch_prior_guidance_validation_v6.sh
bash scripts/robotwin_eval/launch_prior_guidance_formal_v6.sh
```

下一训练候选 v7 不再把训练时约 1% 的 prior gate 在训练后硬改为 50%。新增
`prior_adapter` 训练模式，从第一步起固定 prior gate=`0.5`，同时保持
prior projector 零初始化，因此初始化仍与 untouched PI0.5 完全一致。完整 PI0.5
（含 action expert）、ZTE/Mamba、causal bank、task retrieval、两个 scalar gate
以及 context 分支全部冻结，context projector 精确为零；只训练 task projector、
memory encoder、Gaussian prior、prior action projector 和有界 task/phase router。
训练继续使用 prior residual dropout `0.4`、逐样本 matched-PI 保护、真实 H15
递归相位、H50 输出以及 global batch 256。checkpoint 只根据 train95/validation5
选择，正式测试指标不进入优化或选步。

```bash
bash scripts/train_robotwin_advantage10_prior_only_v7.sh
```

第一轮 v7 在 `step 250` 审计时发现 Accelerate scheduler 漂移：保存状态中的
`last_epoch=2000`，说明原定 250 个 global optimizer steps 的 warmup 每步被 8 个
process 重复推进，实际约 step 32 已结束。因此该轮虽有表面上正的离线 paired
指标，也已停止并禁止用于选模。修正后显式设置
`step_scheduler_with_optimizer=False`，manifest 记录 scheduler 契约，并在每个
optimizer update 后强制断言 `scheduler.last_epoch == completed global steps`。
干净 v7 从 PI0.5 的 step zero 重新训练至 2,000 steps；v6 的 seed-1000 正式结果
不会进入优化或 checkpoint 选择。

训练完成后，先只用 validation5 的 paired 指标执行不可变 checkpoint 选择，再跑两组
新的闭环验证 seed；只有两个 split 均不退化且合计至少 `+6/160`，才允许启动完全未见
的 seed-10000 最终测试：

```bash
python3 scripts/select_robotwin_prior_adapter_checkpoint.py \
  /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-prior-only-v7-corrected/zeva
bash scripts/robotwin_eval/launch_prior_adapter_validation_v7.sh
bash scripts/robotwin_eval/launch_prior_adapter_fresh_final_v7.sh
```

最终测试会同期重新运行 untouched PI0.5 和 ZeVA，使用相同的十任务、每任务 20 个
expert-valid `(seed,instruction)`、continuous model RNG 与 200 条视频/condition；门槛
为 `Base>=57%` 且 `ZeVA>Base`。最终 manifest 必须与两组闭环验证 seed 逐任务互斥，
最终结果不能反向修改 checkpoint、gate 或任何参数。

```bash
PYTHONPATH=src python scripts/audit_robotwin_residual_branches.py \
  --adapter /path/to/zeva_adapter.pth \
  --goal-embeddings /path/to/pi05_task_embeddings.pt \
  --causal-bank /path/to/train_causal_bank.pt \
  --task-manifest configs/robotwin_zeva_advantage10.json \
  --output /path/to/residual_branch_audit.json
```

## 14. 当前产物与实验状态

当前主线产物：

```text
PI foundation:
/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model

B0 language embeddings:
/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt

Stage1 ZTE:
/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth

Train95 causal bank:
/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt

H15 live queries:
/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/live_queries_h15.pt

Task retrieval:
/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1.5-task-retrieval/task_retrieval.pth

十任务 Stage2 训练产物（完整冻结 PI，逐样本保护 + task/phase router；v3 正式闭环 112/200，已拒绝直接交付）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-safe-router-v3/zeva

十任务 v4 独立闭环 residual trust 校准与后续正式评测：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v4-closed-loop

十任务 v5 双独立 split residual trust 校准（正式 109/200，已拒绝）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v5-multisplit

v5 正式评测：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v5-multisplit/formal-calibrated-v5

十任务 v6 prior-only 双 split 校准与正式评测（正式 102/200，已拒绝）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-prior-guidance-v6

十任务 v7 fixed-gate prior-only Stage2（修正 scheduler 后从零重训）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-prior-only-v7-corrected/zeva

十任务 v7 首轮（scheduler 每 global step 错误推进 8 次，已停止且禁止选模）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-prior-only-v7/zeva

十任务 safe-router v2（35/1500 预热时发现 validation 尚未逐任务落盘，主动停止，无 checkpoint）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-safe-router-v2/zeva

十任务第一轮 frozen-PI scalar-gate adapter（正式评测未超过 Base，已拒绝）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-frozen-pi-adapter-v1/zeva

十任务 alpha=0.25 paired eval（理论上限 102/200，已拒绝，仅诊断）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5/formal-wise025-b125-z250-seeded-v1

十任务 action-expert LR 5e-8 原始候选（理论上限 112/200，已拒绝，仅诊断）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5/formal-conservative-ae5e8-b125-z250-seeded-v1

十任务 action-expert LR 1e-7 候选（理论上限低于 anchor，已拒绝，仅诊断）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5/formal-conservative-b125-z250-seeded-v1

十任务第一轮 Base-1000 / ZeVA-250（Base 低于 anchor，已拒绝，仅诊断）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5/formal-b1000-z250-seeded-v1

十任务旧 v1（Transformers 4.53 compatibility runtime，闭环已证明失真，仅审计）：
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-best-v1-pair-v1

历史 Stage2 action-expert-v7（FFmpeg/eager，每步 teacher，已停止）:
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage2-action-expert-v7-dual-residual

历史 Stage2 action-expert-v6（prefix/action 双位置注入，已停止，不可续训）:
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage2-action-expert-v6-gaussian-nll

历史 Stage2 full-v5 Gaussian NLL（已停止，不可续训）:
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage2-full-v5-gaussian-nll

历史 Stage2 full-v4 deterministic-MSE（已停止，不可续训）:
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage2-full-v4-duallr

历史 Stage2A adapter-only（图像域错误，已废弃）:
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage2a-adapter/005000

当前诊断性正式评测:
/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-zeva-only-stage2a-h15-seen-v1
```

当前状态判断：

| 模块 | 状态 | 结论 |
|---|---|---|
| Stage1 ZTE | gate passed | 可保留 |
| Stage1 causal bank/live queries | hash 与 H15 递归一致 | 当前 foundation 下可保留；PI-base 替换见第 16 节 |
| Stage1.5 retrieval | validation 99.704% | 可保留；正式评测仍有 4/1000 episode 误检索 |
| Stage2 v14 H15 direct output residual | 当前正式候选 | 完整 best-v1 与 Stage1 冻结；直接学习 `expert - Base`；step1500 离线 10/10 任务改善，正在同 seed/RNG 闭环验证 |
| Stage2 v13 Base/prior convex interpolation | 已拒绝 | gate 到 step1000 坍缩为 `8.5e-8`，最终等价于不用 ZeVA |
| Stage2 v11 H15 prior/action-expert | 已拒绝 | split-j Base 43/80、ZeVA 41/80，token 注入闭环回退 |
| Stage2 frozen-PI prior adapter v7 | 已拒绝 | 两个互斥 split 均为 -3/80，合计 -6/160；未进入 final |
| Stage2 anchored-v9 joint | 已拒绝 | Base 84/160、ZeVA 89/160，仅 +5，低于预注册 +6；未进入 final |
| Stage2 anchored-v9 adapter | 历史已拒绝 | 旧 frozen-PI 双 residual 方案；闭环证据仅保留作失败分析，不再是当前候选 |
| Stage2 action-expert-v8 | 已拒绝 | 两支均完成 5000 steps，但自身 residual-off teacher 无法约束跨分支漂移，`eligible_count=0`，不得进入闭环 |
| Stage2 action-expert-v7 | 历史 eager/FFmpeg 主线 | **已停止；保留审计，不作为 v8 续训点** |
| Stage2 action-expert-v6 | 历史 prefix/action 双位置注入 | **已停止；结构与 v8 不兼容，不得续训** |
| Stage2 full-v5-gaussian-nll | 历史 full-PI0.5 | **已停止；不得作为 v8 续训点** |
| Stage2 full-v4-duallr | 历史 deterministic-MSE | **已停止；与 v5 结构不兼容，不续训** |
| Stage2A-005000 | 历史 adapter-only | **已废弃：训练 PI 图像数值域错误** |
| 68.5% formal eval | 685/1000 | 只能作为诊断，不作为最终论文结果 |
| 与旧 18-task Ours 对比 | 237/360 vs 239/360 | 未保存相同 seed/instruction，不能视为严格回退 |

## 15. 已修复的 Stage2 图像域问题

自定义 FFmpeg loader 产生 CHW `uint8 [0,255]`。ZTE 会检测最大值并正确除以 255，所以 Stage1 正常；旧 Stage2 把同一图像直接送入 PI0.5 的 `VISUAL=IDENTITY` processor。PI0.5 内部只执行：

```text
x_pi = 2 * x - 1
```

不会替 uint8 输入除以 255。抽样图像的真实结果是：

```text
FFmpeg decoded:       [0, 245]
Stage2 PI input:      [-1, 489]       # 错误
Formal-eval PI input: [-1, 0.9215686] # 正确
```

所以 Stage2 的 baseline/enhanced validation 都是在错误图像域中计算，`0.390532 < 0.399140` 不能证明正确部署域中的 non-regression。

代码现已完成以下修正；旧 `stage2a-adapter/005000` 仍然无效：

1. 共享 `prepare_robotwin_pi_image()` 在进入 PI preprocessor 前显式转换为 contiguous CHW float32 `[0,1]`；
2. trainer 和 evaluator 共用 dtype、shape、finite、range assertion；
3. `verify_robotwin_stage2_image_contract.py` 用真实视频帧验证训练 CHW 与评测 HWC 路径逐像素一致；
4. 当前 v11 manifest 和 training-state 会记录 foundation、untouched best-v1 immutable teacher、冻结集合与数据协议，代码拒绝从错误图像契约或身份不匹配的 checkpoint resume；
5. frozen-PI safe-router/v7、v6、full-v5、action-expert-v8 与 anchored-v9 均已停止并仅作审计保留；当前候选位于 `advantage10-prior-action-expert-v11/zeva/000500`，只允许按预注册四 cell 做 paired Base/ZeVA rollout。
6. Stage2 启动时锁定并记录 handoff-native Transformers 5.5.4、tokenizers distribution metadata 0.22.2 和 host-compatible tokenizers module 0.21.4；Transformers 4.53 的 cache、attention mask、image-return 和 tokenizer shims 在 native runtime 下全部关闭。原始 best-v1 在 native runtime 的闭环 anchor 为 4/5，而 4.53 compatibility runtime 为 0/5，故旧 4.53 训练/评测产物不得进入最终结果。
7. v11 使用 baseline 同款 TorchCodec，并以 32-entry/worker LRU 复用 decoder；实帧三相机与 FFmpeg 逐像素一致。
8. v11 保持 global batch 256，每卡 16、累积 2；H15 prior hook 安装后再编译原 joint forward，prior projector 在初始化时保持精确零值。
9. residual-off exact Base teacher 每 2 个 optimizer step 运行一次；preservation weight 为 8，hinge 目标包含 `5e-5` 正 margin，validation 仍逐 batch paired。

## 16. 当前最终训练流水线与 PI base 替换规则

如果继续使用当前 `pretrained_model`：

```text
保留 Stage1 + bank + live queries + retrieval
        -> 修正 Stage2 图像 loader/断言
        -> 先固定已训练且 held-out flow 最优的 Base action expert
        -> 从该 Base 初始化 ZeVA，冻结完整 PI0.5并用residual-off路径做paired preservation
        -> 冻结同一批 seed+instruction
        -> paired untouched Anchor vs trained Base vs ZeVA
        -> 若显著 non-regression 且有收益，Stage2 即最终模型
        -> 若 Stage2 稳定但提升不足，再决定 Stage3
```

如果只替换 PI base，需要分两种情况：

1. **可兼容的 policy-only 替换**：架构、tokenizer/vocabulary、三相机、Joint14/EEF16、H50/H15 和 normalization 完全相同，并且显式继续使用 Stage1 导出的冻结 task-embedding table。此时 **Stage1 ZTE/bank/retrieval 可复用，从新 PI base 重训 Stage2 即可**。
2. **语义坐标或物理契约变更**：使用新 PI base 自己的 language embedding，或 tokenizer、normalization、camera/state/action contract 任一变化。此时旧 B0 和 bank lineage 不再对齐，必须重新导出 B0 embedding，并重训 Stage1/重建 bank 与 retrieval。

当前代码支持把实际 PI foundation 与 ZeVA 的冻结辅助语言坐标显式分开：

- `--foundation-checkpoint` 决定真正参与视觉—语言前向的 PI0.5；是否优化 action expert 由 Stage2 variant 决定，active adapter 不优化；
- `--goal-embedding-checkpoint` 只为 ZeVA 的 task schema、task retrieval 和 Stage1 `B0` 提供冻结 token embedding table，并在 tokenizer 不完全一致时拒绝启动。

因此，当新 PI base 的物理协议和 tokenizer 完全兼容时，可以让 foundation 使用 `pretrained_model-best-v1`，同时将 `goal_embedding_checkpoint` 固定为旧 Stage1 的 `pretrained_model`。这属于上面的“policy-only 替换”：旧 ZTE/bank/live queries/retrieval 可以复用，只重训 Stage2，但 manifest 必须同时记录两条 lineage。普通 PI baseline 不读取这一辅助语言表。

如果要求 ZeVA 也改用 `pretrained_model-best-v1` 的新语言坐标，而不是上述显式解耦：

```text
重新导出 B0 language embeddings
        -> 重训 Stage1
        -> 重建 causal bank/live queries
        -> 重训 task retrieval
        -> 从同一新 PI base 重训 matched action-expert Base 与 ZeVA Stage2
        -> paired closed-loop evaluation
        -> 可选 selective Stage3
```

原因是 best-v1 的 PI language embedding table、VLM feature、action expert 和 flow behavior 均可能不同，旧 ZeVA 产物不能被视为与新 foundation 兼容。

2026-09-09 的 v7 十任务实验是已拒绝的显式解耦、frozen-PI 方案。逐 tensor 检查确认 best-v1 与旧 foundation 的若干语言/视觉/action 权重不同，但 tokenizer、预处理、normalization 与物理协议兼容；因此它证明 Stage1 辅助语言坐标可以显式固定，却没有证明当时的 prior-only 注入有效。v8/v9 继续复用这条 Stage1 lineage，但真实 policy foundation 始终是 best-v1；当时的 anchored-v9 adapter 冻结完整 PI0.5，只开放 ZeVA fusion/prior/projectors/gates/router，并由 residual-off exact Base path 监督。

## 17. 最终模型的通过定义

一个 ZeVA checkpoint 只有同时满足以下条件，才能进入论文主结果：

1. Stage1 representation/capability gate 全部通过；
2. artifact hash 和 foundation lineage 完整；
3. Stage2 train/eval preprocessing parity test 通过；
4. ZeVA residual-on 必须在相同当前 action-expert 权重、相同 flow noise 下与 residual-off teacher 做逐样本配对；
5. Stage2 offline matched-flow non-regression；
6. task retrieval 达到阈值，并报告真实 rollout 误检索率；
7. untouched Anchor、训练 Base 与 ZeVA 使用完全相同的 episode manifest、condition 初始 RNG seed 与 continuous 口径；
8. 训练 Base 不低于同轮 Anchor 和历史 57% 下限，且 ZeVA paired closed-loop success 必须严格高于训练 Base；
9. 所有测试 episode 均有视频、progress JSON 和 summary 三方一致性校验；
10. 测试数据、失败轨迹和成功标签没有进入任何训练 bank 或参数更新。

## 18. 代码对应关系

```text
src/openpi/zeva/robotwin_contract.py       PI0.5 handoff 与物理契约
src/openpi/zeva/transition_encoder.py      Mamba ZTE
src/openpi/zeva/causal_bank.py             train95 causal bank
src/openpi/zeva/memory.py                  BIT/PIM
src/openpi/zeva/context.py                 memory context fusion
src/openpi/zeva/robotwin_policy.py         PI0.5 wrapper、检索与残差注入

scripts/export_robotwin_goal_embeddings.py B0 task-language embedding
scripts/train_robotwin_zte.py               Stage1
scripts/eval_robotwin_stage1.py              Stage1 gate
scripts/export_robotwin_causal_bank.py       causal bank export
scripts/export_robotwin_live_queries.py      H15 live-query export
scripts/train_robotwin_task_retrieval.py     Stage1.5 retrieval
scripts/train_robotwin_stage2.py             多 variant Stage2 trainer；active 为 output_residual
scripts/cache_robotwin_base_actions_v13_8gpu.sh immutable Base action cache
scripts/train_robotwin_advantage10_output_residual_v14.sh active v14 训练入口
scripts/audit_robotwin_v14_checkpoint.py     frozen PI 与 H15 direct residual 契约审计
scripts/calibrate_robotwin_v14_task_scales.py development-only 二值安全路由
scripts/train_robotwin_advantage10_prior_action_expert_v11.sh 历史失败 v11 入口
scripts/train_robotwin_advantage10_anchored_v9.sh anchored-v9 训练入口
scripts/select_robotwin_anchored_v9.py       独立 Base teacher 的 validation5-only 选点
scripts/audit_robotwin_residual_branches.py  不加载 PI0.5 的双残差强度审计
scripts/calibrate_robotwin_residual_trust_multisplit.py 双验证集 non-regression selector
scripts/robotwin_eval/zeva_policy.py         RoboTwin 部署适配器
scripts/robotwin_eval/launch_formal_eval.sh  正式 ZeVA-only 评测
scripts/robotwin_eval/launch_paired_formal_eval.sh paired baseline/ZeVA 评测
scripts/robotwin_eval/launch_closed_loop_residual_calibration_v5.sh 双 split 校准入口
scripts/robotwin_eval/launch_anchored_v9_validation.sh 参数化的 v9 双 split 验证；active adapter 用 seed9000/12000
scripts/robotwin_eval/launch_anchored_v9_fresh_final.sh v9 seed10000 三条件 final
scripts/robotwin_eval/wait_and_run_anchored_v9.sh v9 持久门控协调器
```

## 19. v11：基于掉点机制修正后的正式 Stage2

此前掉点不是简单的学习率或训练步数问题：

1. v8/v9 先训练 action-expert Base，正式闭环时 Base 本身已低于 untouched PI 的 `114/200=57%`；
2. v10 的 direct-context residual RMS 随训练增长到 prior residual 的约 `12.6` 倍，所谓 prior 增强实际由另一条高维分支主导；
3. v7 的 prior-only H50 方案在 stack/ranking 任务上把轨迹拉长约 146--238 simulator steps并更频繁撞到 step limit，说明 H50 残差与 H15 执行/重规划周期存在结构错配；
4. v6/v7 曾在 offline paired flow 上为正，却在闭环下降，所以离线 loss 不是最终 checkpoint 判据。

因此 v11 不是继续调 gate 碰运气，而是删除这几个混杂因素：

- 初始化模型与 immutable teacher 都是 untouched best-v1；
- 冻结 PaliGemma/VLM、Stage1 ZTE/Mamba、causal bank、retrieval；
- action expert 使用 `5e-7`，task/context-to-prior、Gaussian prior 与 prior projector 使用 `5e-5`；
- direct-context projector 硬清零并冻结，运行时另有 hard-disable 标志；
- gate router 清零并冻结，`2*sigmoid(0)=1`，因此 prior guidance 真正固定为 `0.5`；
- task language、BIT/PIM context 与真实 recurrent H15 phase 只条件化 Gaussian prior；
- prior 只注入 action tokens `0..14`，Gaussian NLL 也只监督同一 H15；
- prior residual dropout 为 `0.4`，global batch 为 `16×2×8=256`，TorchCodec 与 `torch.compile` 保留；
- 每两步抽样一次 immutable same-noise teacher，逐样本 preservation hinge 权重为 8、margin 为 `5e-5`。

训练共 2,000 optimizer steps、每 250 步保存。250/500 步只产生候选，不因 offline flow 变好直接进入 final。当前预注册候选固定为 step500：它必须先通过 direct-context-off、fixed-router、H15 injection/NLL、foundation lineage 和分支强度审计，再完成四个不能反向挑 checkpoint 的同-seed paired closed-loop cell：`seed15000×RNG20260907`、`seed16000×RNG20260908` 及交换 RNG 后的两组 cross cell。每组都是 10 tasks×8 episodes；要求四组各自 ZeVA≥Base、每个任务汇总不退化，并且 320 个 paired episodes 总增益至少 `+12`。最终才使用用户指定的 seed1000 起始 expert-valid 10×20 manifest；硬门槛仍是 Base≥57% 且 ZeVA>Base。

正式入口：

```bash
bash scripts/train_robotwin_advantage10_prior_action_expert_v11.sh
bash scripts/robotwin_eval/launch_prior_action_expert_v11_validation.sh
bash scripts/robotwin_eval/launch_prior_action_expert_v11_cross_validation.sh
# 只有 four_cell_validation_summary.json 通过时才会启动
bash scripts/robotwin_eval/launch_prior_action_expert_v11_final.sh
```

若四 cell 门槛失败，不允许继续调 gate、dropout 或 guidance 碰运气。唯一下一组归因实验是从 untouched best-v1 训练 `action-expert-only`：保持 action-expert LR、步数、paired teacher 和四 cell 协议完全相同，同时关闭 prior/context 与 Gaussian NLL，以区分 action expert 漂移和 prior 注入两种失败源。

## 20. 一句话总结

本方法先用 Mamba 从真实 H15 action-effect transition 学习任务阶段与因果变化，再通过 train95 causal bank、BIT/PIM 和 task-language retrieval 得到部署同口径的在线因果上下文；v11 让该上下文只生成 H15 Gaussian action prior，以固定 0.5 guidance 注入 action expert，并在 untouched best-v1 teacher 保护下以极小学习率适配 action path。是否进入可选 Stage3，只由严格的多 split paired 闭环 non-regression 决定，而不是由离线 loss 决定。
