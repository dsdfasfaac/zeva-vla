# H15 闭环失败后的机制诊断（2026-09-17）

正式配对评测的结果是 Base 111/200、ZeVA 106/200，参见[完整结果](ROBOTWIN_H15_ROUTE_PAIRED_RESULTS_20260917.md)。以下诊断只使用 train95/validation5，不读取正式测试成功标签选模型或参数。

## 假说 1：Gaussian NLL 抢占 ZeVA 梯度裁剪预算

对已评测的 ZeVA 第500步，用与 H15-route 训练相同的连接 H15 flow、`0.01×Gaussian NLL`、gate regularizer 和 NLL-to-context stop-gradient，在 4 个固定 train95 microbatches、共 32 条决策上分别计算 ZeVA 组的原始梯度；无 optimizer、无参数更新。[完整只读报告](results/robotwin-h15-route-20260916/clip_audit_000500.json)记录固定样本索引 SHA256 `e319e328e801e59933db662ed8ef422a6c7d27eee3035750905d6f25e262dd98`、模型/adapter SHA 与逐模块范数。

| batch | H15 flow 梯度范数 | 加权 NLL 梯度范数 | 合并范数 | max-norm=1 裁剪系数 |
|---:|---:|---:|---:|---:|
| 0 | 0.000585 | 0.262855 | 0.262856 | 1.0 |
| 1 | 0.000376 | 0.376440 | 0.376440 | 1.0 |
| 2 | 0.000957 | 0.217878 | 0.217880 | 1.0 |
| 3 | 0.000914 | 0.336152 | 0.336153 | 1.0 |

这四批 NLL 梯度主要集中在 action-prior head，且**均未触发** ZeVA 组的范数 1 裁剪；若仅从这四批推断「NLL 通过裁剪压小了 H15 flow 梯度」是错误的。此前 32 条数据上的原始 NLL/flow 范数比很大，并不能推出裁剪真的发生。这个检查未测完整 global256 梯度累积、采样的 Base-preserve hinge 或 Adam 更新，故只是把该具体假说的优先级降低，并非全程排除。

更直接的问题是实际受 flow 监督的 action-prior head 梯度仅约 `1e-6`，而 NLL 对它是 `0.22–0.38`；prior 注入相对 noisy-action embedding 的范数也只有约 `1.65e-5`。context projector 的 H15 flow 梯度约 `0.0004–0.0010`，所以当前路径的主要有效更新在 context 分支，不能因为 prior head NLL 收敛就认为 ZeVA 已利用它改善 PI 动作。

## 假说 2：ZTE 对 Base 的改错信息没有被当前注入利用

旧 Stage1 probe 在 validation5 显示 phase/causal 组合能比 task-only 更好预测**专家动作**，但那不是 Base 的动作残差，也没有闭环鲁棒性结论。现针对当前已评测 checkpoint 增加一个只读、全 5874 条 validation5 的对照：保持 checkpoint、语言检索、gate、Base、样本顺序与同噪声不变，仅把每个 batch 内**由任务语言推断出的同一任务**的 phase/causal/mask 整组循环错位；不使用 oracle task ID、episode index 或测试集。若原始对齐状态确有针对当前观测的增量作用，原始 H15 flow 应优于错位条件；否则须重新设计表征目标／注入，而不是继续盲目增大 gate 或加训练步数。

两批 32 条决策的只读 smoke 已通过，32/32 条确实被错位；正式全量只读诊断在 aigc29 GPU4 运行，PID `2690047`，输出 `.../h15_route_full/validation_diagnostics_000500_zte_within_task_roll.json`。此测试是机制诊断，不是闭环成功率，也不会写 checkpoint。原始对齐的全量报告已在[上一轮验证记录](results/robotwin-h15-route-20260916/validation_diagnostics_000500.json)固定。

## 决策边界

不因这次正式测试各任务的涨跌选择任务或 seed。全量错位对照完成前，不启动新的训练。完成后先判断 ZTE 状态对 H15 改错是否有实质而非微弱的增量；若没有，应改 Stage1/Stage2 目标或条件接入；若有，再分析为何专家轨迹离线优势无法转化为 on-policy 闭环优势，优先用独立开发 split 的闭环证据检验，而不是继续把正式测试集当调参集。
