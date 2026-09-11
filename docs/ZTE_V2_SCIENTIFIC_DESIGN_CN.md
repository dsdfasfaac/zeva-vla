# ZTE v2 与 PI0.5 因果提示接入：可证伪设计

> 状态：2026-09-11 冻结的重构规格。任何训练参数调整不得绕过本文的表示与闭环准入门槛。

## 1. 已证伪的假设

当前 Stage1 在 4,529 个 episode group、11,679 个决策样本的无泄漏五折审计中，没有为 PI 候选排序提供增量信息：action-only AUC 为 0.5563，Stage1+action 为 0.5416，PI-VLM+action 为 0.5473，联合表征为 0.5375。这个结果只否定当前实现，不否定 ZTE 的研究目标。

代码审计显示当前实现存在四个结构问题：

1. H15 动作经过均值池化，弱化了 chunk 内的接近、抓取、闭合和撤回顺序。
2. effect decoder 的输入已经包含真实 after-effect，容易退化为重建而不是前向动力学预测。
3. task language 被加入 `B0`，同时又直接进入 task head，task loss 可走语言捷径。
4. 历史 Stage2 把 context/prior 加到 action token 或最终动作；它既不等价于 BehaviorVLA 的双通道接入，也不等价于 ZeVA 的 causal prompt。

## 2. 表征目标

ZTE v2 将行为表示显式分解为：

```text
z_proto：整条轨迹近似不变的任务拓扑
z_phase(t)：随执行演化的局部阶段
e_causal(t)：已执行 H15 导致的真实状态变化
```

每个重规划边界的输入严格为：

```text
(三相机 s_t, ordered EEF16 a_{t:t+14}, 三相机 s_{t+15})
```

`B0 = F_init(g, s0)` 保留任务语言 `g`，但所有宣称“来自 sensorimotor trajectory”的 probe 都必须在 language-masked 分支上成立，防止任务文本代替因果学习。

## 3. ZTE v2 结构

### 3.1 三流因果 Mamba

参考 BehaviorVLA VBE，为 visual、action、effect 三个流分别设置 causal Mamba：

```text
h_v = Mamba_v(V(s_t))
h_a = Mamba_a(OrderedChunkEncoder(a_{t:t+14}))
h_e = Mamba_e(E(s_t, s_{t+15}))
```

H15 action encoder 使用带位置编码的 causal 1D encoder/Mamba，输出保留 ordered step tokens；禁止直接 mean pool 原动作。逐 transition 先进行 vision-action mutual attention，再让 behavior/effect stream 查询二者：

```text
h_v <- h_v + Attn(h_v, h_a, h_a)
h_a <- h_a + Attn(h_a, h_v, h_v)
h_z <- h_z + Attn(h_z, [h_v; h_a; h_e], [h_v; h_a; h_e])
```

训练使用完整 episode 的 H15 序列；部署使用同一个 Mamba incremental cache。batch forward 与逐步 forward 必须数值等价。

### 3.2 非泄漏的前向预测

定义 pre-effect state `q_t`，它只能读取 `s_t`、历史和当前 ordered action，不能读取 `s_{t+15}`。由它预测：

```text
hat_delta_t = F_dyn(q_t, a_t)
hat_a_{t+1} = F_act(z_phase(t))  # 新增 opt-in 对照，见下文时间边界
```

目标视觉特征来自 stop-gradient EMA encoder：

```text
delta_t = SG(V_ema(s_{t+15}) - V_ema(s_t))
```

真实 effect 仅在生成 post-transition `e_causal(t)` 时输入，不允许进入 `hat_delta_t` 的预测路径。

2026-09-11 梯度审计修订：历史 pilot 的 `action_prediction_context=pre` 使用 `F_act(q_t)`，next-action loss 对导出的 `phase_head` 和 `post_fusion` 均无梯度。新增 `phase` 对照直接从 normalized `z_phase(t)` 预测下一 H15，不拼接可绕过 phase 的 hidden state。它在执行完当前 H15、已观察到 `s_{t+15}` 后预测下一段动作，故可读取当前 after-image，但不可读取下一段动作或未来图像；forward-effect 仍严格不能读取当前 after-image。旧 checkpoint 缺省保持 `pre`，不能隐式改变已训练语义。

这不是 BehaviorVLA 原代码的逐行复制：所核查官方实现是 `action_predictor(h_a)`，其 action stream 输入经过时间右移；不是 `action_predictor(h_b)`。本对照检验的是“让实际导出的 phase 直接受到未来动作监督”能否带来增量价值，必须通过同预算验证，不能预先声称更优。

## 4. Stage1 损失

```text
L_stage1 =
    0.2 L_next_action
  + 0.2 L_forward_effect
  + 2.0 L_global_supcon
  + 1.0 L_local_temporal
  + 1.0 L_causal_effect
  + 0.1 L_variance_covariance
```

- `L_next_action`：`phase` 对照从 post-transition phase 预测下一 H15 chunk；历史 `pre` 对照仍从 pre-effect state 预测，二者分别记录配置。
- `L_forward_effect`：JEPA-style forward dynamics；预测端看不到 after image。
- `L_global_supcon`：完整轨迹 `z_proto` 的同任务跨 episode 对比学习；主损失在 language-masked view 上计算。
- `L_local_temporal`：同一 transition 的两种图像增强为正样本，其他阶段为负样本，并加 pairwise order ranking；不使用“向量与自己对比”的退化正样本。
- `L_causal_effect`：同一 action-effect 在不同相机增强下为正；打乱 action 或 after-effect 为 hard negative。
- `L_variance_covariance`：防止 phase/causal token collapse。

训练时随机执行 language masking 和 task-language permutation consistency：`e_causal` 对语言置换应近似不变，`z_phase` 对外观增强应稳定，`z_proto` 必须仍能由 sensorimotor 轨迹识别任务。

## 5. Stage1 准入门槛

Stage1 不再用总 loss 或固定 epoch 直接判定。必须同时满足：

1. held-out episode 的 next-action MSE 优于 per-task mean 与旧 ZTE。
2. held-out forward-effect MSE 优于 action-only predictor 与旧 ZTE。
3. language-masked task retrieval Recall@1/Recall@5 优于旧 ZTE，且随机语言置换下降不超过 2 个百分点。
4. phase probe 的 Spearman 相关和 pairwise ordering accuracy 优于旧 ZTE；该 progress 只用于离线 probe，不输入部署模型。
5. cross-task effect retrieval 优于 task-only、w/o-effect 和 shuffled-effect 三个对照。
6. correct-effect、zero-effect、shuffled-effect 的 intervention 必须产生方向一致且显著的 causal-token/预测差异。
7. batch forward 与 incremental H15 recurrence 在 dropout-off 时最大绝对误差小于 `1e-4`。
8. 在 PI rollout state 上，检索准确率和特征范数不能显著低于 expert validation；否则先做 scheduled-rollout/domain adaptation，禁止进入 Stage2。

论文和公开代码的训练长度并不完全一致：BehaviorVLA 论文附录给出 batch 16、40 epochs，公开 README/训练脚本示例为 batch 8、80 epochs。ZTE v2 以验证 gate 选 checkpoint，首轮评估 40 epochs；只有指标仍稳定改善才延长到 80，不把 epoch 数当成正确性的证据。

## 6. 科学的 PI0.5 接入位置

接入采用两个互补通道：

### 6.1 Causal prompt / global context

```text
M_t = F_mem([g, z_phase(t), P_brief(H_brief), P_retrieved(R_t)])
```

`M_t` 通过 zero-init projector 变成 prefix causal-prompt token。它随 vision/language prefix 一起形成 KV cache，使 diffusion/action expert 的 full-attention 能读取它。这对应 BehaviorVLA 的 global prototype prefix，也对应 ZeVA 在 full-attention 中注入 causal prompt。它不作为最终动作残差。

### 6.2 Phase-conditioned Gaussian prior

PBD 使用 `z_proto` 展开有序 waypoint manifold，再以 `z_phase(t)` 做 Progress-Attention，输出 H50 Gaussian：

```text
p(a | z_proto, z_phase) = Normal(mu_prior, diag(sigma_prior))
```

训练以 Gaussian NLL 监督；`mu_prior` 经 zero-init projector 后加到 noisy-action embedding：

```text
e_tilde(a_sigma) = e(a_sigma) + m * Proj(mu_prior)
m ~ Bernoulli(0.6) during training
guidance = 0.5 during inference
```

禁止把 `mu_prior` 直接与最终 EEF16 输出插值，禁止 post-diffusion residual。

## 7. Stage2 训练与防泄漏

- ZTE v2、memory bank 和 retrieval key/value 全冻结。
- 训练检索使用 task-language + initial observation，且从候选 bank 排除当前 episode；禁止 `episode_index` oracle lookup。
- 首先冻结 PI，仅训练 zero-init prompt/PBD 并做 attribution smoke；确认 correct prompt 优于 shuffled prompt 后，才进入 joint tuning。
- joint tuning 使用 PI LR `5e-6`、新模块 LR `5e-5`、global batch 256。先比较 5k/10k checkpoint；只有闭环 paired validation 与 loss 均继续改善才延长。
- 保存完整 PI `model.safetensors`、ZTE/PBD/prompt adapter、optimizer 和 scheduler state。
- 最终交付的普通 Base 与 ZeVA 都从指定 best-v1 出发，在相同选定 10-task 数据、训练预算和 PI 优化设置下训练。untouched best-v1 单独作为 Anchor 核查基础能力，不替代用户要求的训练后 Base；不得选择异常退化的 Base 制造优势。

Stage2 准入闭环前必须通过四个固定消融：Base、global-only、prior-only、both。记录每支 prompt/prior RMS、gate、gradient norm、flow delta、prior NLL/std，并做 correct/zero/shuffled retrieval 因果置换。

## 8. 评测协议

保留标准 RoboTwin 10x20 single-attempt paired 评测：Hard/seen、Large_D435、Joint14、relative EEF16、H50 预测/H15 执行、相同 seed/instruction/model RNG。

另外增加与 ZeVA 论文一致的 fixed-episode repeated-attempt 评测：同一 scene/object/robot initialization 最多四次 attempt；BIT 每次 attempt 清空，PIM 在四次之间保留，换 episode 才清空。Base 接受相同 attempt 数和随机预算。必须同时报告 attempt-1 SR、各 attempt SR 和 cumulative success；不得用 cumulative 指标替代标准 single-attempt 结果。

## 9. 决策规则

- 任一 Stage1 gate 未通过：只修 ZTE，不启动 Stage2。
- correct prompt 与 shuffled prompt 无可归因差异：修注入/训练，不做闭环大评测。
- single-attempt ZeVA 低于 Base：不得进入 final，即使 repeated-attempt cumulative 更高。
- 只有两个独立 paired split 均为正，才运行最终 10x20 并交付成功率。
