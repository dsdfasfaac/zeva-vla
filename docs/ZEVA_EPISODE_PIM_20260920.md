# ZeVA Episode-PIM：单个 episode 内的长短期记忆

## 固定定义

- **BIT** 是当前 episode 内、当前 H15 边界的短期交互状态。
- **PIM** 是同一个 episode 中更早边界 BIT 的因果长期记忆。
- 当前边界先读取 `PIM[0:t)` 并预测动作，之后才把当前 BIT 写入 PIM；因此不存在当前目标泄漏。
- PIM 只在 episode 结束时清空，不读取其他 episode，也不再定义 attempt 提交或 attempt reset。

这条新分支与旧的跨-attempt PIM checkpoint 不兼容。旧实现和失败结果继续保留用于复现，但不再代表当前方法。

## 训练

从已验证的 CTE+BIT+EAP Parent（epoch40 Stage2 step5000）初始化。CTE、BIT、task memory、语言检索、PI 和 EAP 全部冻结，只训练 PIM 的 phase/key、BIT/value、query、prefix projector 和 EAP context projector。训练数据仅来自 train95；时间 `t` 的样本只配同一 episode 的 `[0,t)` 历史，不使用成功标签、validation 或正式评测数据。

固定设置为 8×H100、每卡 batch32、global batch256、2000 optimizer steps、PIM LR `5e-5`、最多保留64条历史 BIT。每500步保存完整模型、adapter、optimizer、训练状态和8-rank RNG，但只允许固定step2000进入验证。

## 验证边界

先验证 episode 起点 PIM 为空时与 Parent 精确一致，再在冻结 validation5 比较 aligned causal PIM、同 episode 错位 PIM 和 PIM-off。通过后，只在新的 disjoint 10任务×8、单 episode 单次执行开发集上配对比较固定 Base、固定 Parent 和 Episode-PIM；开发 seed 从3000000开始预声明选取。

该候选是在原正式集合已经评测后定义，必须标注为 post-formal candidate；不得利用已见正式标签挑 checkpoint、任务、seed 或门槛，也不能用新结果改写历史正式结论。

## 启动前验证

7项因果/路由单元测试通过。真实Parent与真实train95数据的单卡global8两步反传通过；8×H100、batch32/卡、global256的两步容量测试也通过，PIM梯度finite且非零。容量测试checkpoint含8份rank RNG，冻结Foundation模型SHA与Parent完全一致，Parent EAP的23个tensor逐位一致。证据见 `docs/results/robotwin-episode-pim-20260920/runtime-smoke.json`。

第一次单卡smoke在更新前被数据guard拒绝：原始adapter包含50任务，而冻结CTE artifact有本轮10任务。失败目录完整保留；guard已改为跳过无关40任务，同时反向验证artifact内5230条train episode全部存在。该修复没有改变模型、训练数据或gate。
