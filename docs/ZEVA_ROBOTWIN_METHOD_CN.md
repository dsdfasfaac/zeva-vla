# ZeVA：CTE、BIT、PIM 与 EAP

更新：2026-09-19。本页定义当前 ZeVA 方法及训练口径；旧路线仅保留在历史文档中。

## 核心表示

- **CTE（Causal Transition Encoder）**：三相机视觉、上一段真实执行的 H15 EEF16 动作和因果状态三路建模，输出当前执行阶段 `z_t`。
- **BIT（Boundary Interaction Token）**：CTE 在每个 H15 边界预测的短期 effect token，描述当前 attempt 下一边界的视觉变化。BIT 随当前 attempt 实时更新。
- **PIM（Persistent Interaction Memory）**：同一 episode 内已经结束的 attempt 的 BIT 序列。attempt reset 时提交，episode reset 时清空。
- **EAP（Effect Action Prior）**：以任务全局先验、当前 phase 和 PIM context 预测 H50×16 Gaussian action prior，并在动作专家的 noisy-action embedding 中注入 mean residual。

当前已验证的 CTE+BIT+EAP 路径使用两个 prefix：任务全局 token 与 BIT。PIM 扩展增加第三个 prefix，并将检索到的跨 attempt context 同时送入 EAP。第一 attempt 的 PIM 为空，其前向必须与已验证模型逐位一致。

## Stage1：训练 CTE 与 BIT

Stage1 使用 action reconstruction、next-vision prediction、global task contrastive、local temporal distinctiveness 和 effect prediction：

`effect_target_t = stopgrad(EMA_visual(o_{t+1}) − EMA_visual(o_t))`

`L = 0.1 L_action + 0.2 L_vision + 2 L_global + L_phase + 0.2 L_effect`

三路 640×480 图像分别缩放至 224×224；动作按 RoboTwin baseline mean/std 归一化。模型为 ImageNet 初始化 ResNet18、4 层三流 Mamba、dim256。固定训练 80 epochs，batch8，vision LR `1e-5`，其余 LR `1e-4`。Stage1 不因增加 PIM 而重训。

## Stage2：CTE+BIT+EAP

冻结 CTE、train-only task memory 和语言检索；完整训练 PI0.5 与 EAP。任务语言先预测任务，再在该任务的 CTE memory 中 top-5 cosine/softmax 聚合任务全局 token。BIT 作为实时短期 prefix；EAP residual 只进入动作专家 embedding，不直接改最终 EEF 动作。保持 H50 输出、H15 执行和真实 recurrent 更新。

已验证 checkpoint 使用 global batch256、5000 steps、PI LR `5e-6`、EAP LR `5e-5`。冻结 10×20 配对结果为 Base `111/200=55.5%`、ZeVA `122/200=61.0%`，提升 `+11/200=+5.5pp`。这通过工程 gate，但 McNemar `p=0.2543`，不宣称统计显著。

## Stage2-PIM：跨 attempt BIT

PIM 是 Stage2-only 扩展，从上面的已验证 checkpoint 初始化：

1. 一个 attempt 中，每个 H15 边界记录 `(phase, BIT)`；不会立刻写入 PIM。
2. `reset(scope="attempt")` 将完整 BIT trace 均匀压缩后提交至 PIM，随后清空 CTE recurrence 和当前 BIT。
3. 下一 attempt 用当前 phase 对历史 `(phase, BIT)` 做 attention，形成 PIM context。
4. PIM context 作为第三个 prefix，并条件化 EAP；`reset(scope="episode")` 清空全部 PIM。

离线训练不做全局随机配对。对每条当前轨迹，只从 train95 选择同任务且不同 episode 的历史：50% 同条件初始 CTE phase 最近邻、20% 同条件远邻、10% 跨条件最近邻、20% PIM-off。所有匹配均不使用成功率标签；validation 和 formal 轨迹绝不进入 PIM 训练源。

PIM Stage2 固定首轮 2000 steps、global batch256。前 500 steps 仅训练 PIM 模块（LR `5e-5`）；之后 PI 使用 `1e-6`、已有 EAP 使用 `1e-5`、PIM 保持 `5e-5`。CTE 始终冻结。PIM-off batch 用于维持第一 attempt 的原模型能力。

## Reset 与评测协议

```python
# 新 episode：清空 CTE、BIT 和 PIM
policy.reset(scope="episode")

# 同一 episode 内一次失败后重试：提交 BIT 到 PIM，只清短期状态
policy.reset(scope="attempt")
```

评测保持 Large_D435 640×480 三相机、Joint14→relative EEF16、H50 输出/H15 执行、seen instruction 和冻结 seed manifest。PIM 版先做 validation5 的 PIM-aligned/PIM-shuffled/PIM-off 检验，再用与正式测试不相交的多-attempt 开发 seeds 做配对闭环；不能用正式 success labels 选择 checkpoint、seed 或 gate。

## 来源与许可

三流时序编码和动作先验实现参考了 Apache-2.0 项目 `iLearn-Lab/ICML26-BehaviorVLA` 的 commit `0dbabc7e79791a325c4e76acde0ddfd7a18e8326`。来源信息保留在第三方声明中；ZeVA 对外术语统一为 CTE、BIT、PIM 和 EAP。RoboTwin 三相机/H50-H15、task-language 检索、BIT effect 目标和跨-attempt PIM 均是当前 ZeVA 的独立训练与部署契约。

历史正式失败 Base `111/200`、旧 ZeVA `106/200` 仍完整保留；不得删除失败记录或将 PIM 离线指标表述为闭环成功率。
