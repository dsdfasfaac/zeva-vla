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

两批 32 条决策的只读 smoke 已通过，32/32 条确实被错位。全量 5874 条验证已完成：[一步错位报告](results/robotwin-h15-route-20260916/validation_diagnostics_000500_zte_within_task_roll.json)与[半批错位报告](results/robotwin-h15-route-20260916/validation_diagnostics_000500_zte_within_language_task_half_roll.json)各有 5871/5874 条实际重排，样本顺序 SHA 与原始报告完全相同。原始对齐 H15 flow 为 `0.01006796956`；一步错位为 `0.01006785687`，半批错位为 **`0.01006470714`**，两者均更低。半批错位使已训练 ZeVA 的 H15 误差相对原始对齐再低约 `3.26e-6`（约 0.0324%）；与自身 residual-off 的 win fraction 从 `53.37%` 升至 `53.90%`。这个结果**没有证据表明当前注入利用了精确的逐状态 ZTE 对齐**，甚至提示现有对齐可能带来噪声；但同任务批内错位仍可能保留共享任务/阶段分布，不能据此断言 ZTE 表征本身无用。原始对齐的全量报告已在[上一轮验证记录](results/robotwin-h15-route-20260916/validation_diagnostics_000500.json)固定。两种错位都是验证诊断，不是闭环成功率，也不会写 checkpoint。

## 决策边界

不因这次正式测试各任务的涨跌选择任务或 seed。当前两种错位都未降低验证性能，因此不再延长同一个 frozen-ZTE H15 双残差候选或简单扫 gate。下一道诊断直接固定 Base1000 的 H15 采样动作、专家动作与同一 Stage1 v2 live queries，用同容量探针比较“Base 动作+任务”与“Base 动作+任务+ZTE”对**Base 动作残差**的 held-out 预测力：若没有增量，优先改 Stage1 表征目标；若有增量，优先改 Stage2 条件接入或引入独立开发 split 的 on-policy 监督。探针仍只看专家状态，不能把离线 MSE 作为闭环收益。不得继续把正式测试集当调参集。

[Base1000 残差探针预先声明的协议](../configs/robotwin_base1000_zte_residual_probe_20260917.json)固定 Base1000 模型 SHA `bcf1d4f5…`、与正式 Base 相同的 best-v1 task-language 坐标、Stage1 v2 H15 live queries、8192/4096 的均衡样本上限、同容量 MLP 的两组输入和 1000 步固定训练预算。8 卡缓存已完成，[缓存元数据](results/robotwin-base1000-zte-residual-probe-20260917/base_action_cache.json)确认真正获得 train `8190`、validation `3489` 决策（各任务可用样本数使实际数低于上限）；没有用任何正式测试标签。只保存 Base 采样动作和样本索引的约 18 MB 缓存在远端，不加入 git。首两次预检发现默认数据根不存在、task-language 初始化不同于正式 Base，均在产出缓存前停止并保留日志；第三次使用训练数据根和正式 Base 的 best-v1 坐标成功。

[预注册两臂探针报告](results/robotwin-base1000-zte-residual-probe-20260917/base1000_zte_residual_probe.json)：原始 Base H15 normalized EEF16 MSE `0.01134641`；“Base动作+任务”校正探针为 `0.01174905`（反而退化）；加入同一观测对齐的 ZTE phase/causal 后为 **`0.01048606`**，相对“Base动作+任务”低 `10.75%`，且 9/10 任务改善，超过预声明的表示信息门槛。相对未校正 Base 低约 `7.58%`。两臂都使用相同的诊断性 oracle task one-hot 来隔离 ZTE 的增量作用；它**不是部署策略**，部署仍必须从任务语言检索。这仍是专家状态的离线结果，不是闭环成功率。

为排除“只是增加了输入维度或任务先验”而非状态对齐的解释，在第一份探针报告之后追加了[同任务 ZTE 乱序负对照](results/robotwin-base1000-zte-residual-probe-20260917/base1000_zte_residual_probe_shuffled_control.json)：训练和验证分别独立、确定性地在同任务内打乱 ZTE 特征，Base 动作和任务坐标保持原样；train `8182/8190`、validation `3485/3489` 条实际错位。同容量、同初始化和同采样序列下，乱序 ZTE 的 MSE 为 `0.01225184`，**真实对齐 ZTE 低 14.41%，10/10 任务均优于乱序**。该对照是在初次结果后设计的机制检验，不冒充预注册选模条件，也没有使用正式测试标签。这说明 ZTE v2 **确实含有可用于纠正 Base 动作的状态信息**；当前 H15 token 双残差不能利用它，不能据此说 ZTE 表征本身不 work。

## 下一候选与当前阻塞

按[训练前固定的配置](../configs/robotwin_base1000_ztev2_output_residual_20260917.json)，下一候选冻结已训练的 Base1000、Stage1 v2、causal bank 和语言检索，把经身份核验的 Base H50 动作作为输入，只对实际执行的 H15 预测有界输出残差；不用先前占主要梯度的 Gaussian NLL 辅助项。训练 500 optimizer steps、global batch256，在 250/500 保存。只允许使用 train95/validation5；[选择器](../scripts/select_base1000_ztev2_residual_checkpoint.py)按全量 validation5 的样本加权 H15 MSE 至少降低 3%、至少 8/10 任务不退化且残差/门控有限值，选第一个达标点。专家状态离线门槛只是进入新独立开发 split 的必要条件，不是成功率声明。

截至本次更新，训练代码和[单卡缓存训练入口](../scripts/train_robotwin_base1000_ztev2_output_residual.sh)已就位；本地语法与配置检查通过，但当前执行环境拒绝到 aigc29 的 SSH 连接（`Operation not permitted`），未能做远端 smoke 或启动 500 步训练。没有新 checkpoint、开发集结果或 +4pp 闭环结果。旧正式 200 episode 已在之前实验中被观察过；新候选不得用其任何成功标签、逐任务涨跌或 seed 结果选点。即使将来重复该固定评测，也必须透明披露这种历史暴露，不能将其称为全新盲测。
