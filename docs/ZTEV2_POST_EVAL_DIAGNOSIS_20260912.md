# ZTE v2：配对评测后的诊断边界与下一步

更新：2026-09-12。本文区分已测事实、未测假设和计划，不改变已完成实验。

## 已测事实

- 固定的正式配对结果：Base 109/200，ZeVA 110/200，Anchor 101/200。ZeVA 多成功一次，不构成稳定优势证据；完整结果见[评测记录](ROBOTWIN_ZTEV2_PAIRED_RESULTS_20260912.md)。不据此回头选择 checkpoint、任务、seed 或指令。
- 当前 Stage1 为 step4096。1350 条验证轨迹上，task=96.8889%、order=88.6432%、effect cosine=0.375731、language consistency=0.851242，均超过配置参考线 50%、80%、0.05、0.80。这些是辅助诊断，不是 BehaviorVLA 官方标准。
- `train_robotwin_zte_v2.py` 将 `probe_gate_status` 写为 `diagnostic_only`，并固定写入 `probe_gate_passed=false`；不能把该布尔值解释为某个测量不及格。order 来自独立 progress head，不能代替实际导出 phase token 的质量测量。
- Stage2 没有指定固定 anchor 权重时，`_matched_baseline_flow` 调用当前 student 的 foundation，关闭 ZeVA 注入作为对照。action expert 同时更新，因此这个对照不是独立训练 Base，也不是固定能力锚点。
- 当前正式选步使用 held-out H50 flow。部署实际执行 H15；二者契约没有冲突，但既有汇总没有单独解释前 H15 的预测收益。

## 尚不能下的结论

- 约 1% 的 sigmoid gate 不足以证明注入太弱：还需测量投影后、乘 confidence/gate 后的残差相对 action embedding 的幅度。
- ZTE 辅助分数达到参考线，不足以证明它提供了 PI 尚未掌握的动作信息；同样，闭环增益不足也不能单独证明编码器无效。
- 某个最后训练 batch 的 NLL 与全验证集 NLL 不能直接当成泛化差距；需要同口径、同样本统计。
- H50 输出/H15 执行是用户指定协议，不把它作为待修复错误，也不改为 H10 或 H15 输出。

## 当前限定工作：先增加验证观测

Luna worker 正在修改本地 Stage2 trainer 和专门测试，增加默认关闭的 validation diagnostics；尚未将改动部署到远程，也没有新训练。要求：

1. 不改变历史 loss、模型权重、门控、优化器、训练步数或默认验证输出。
2. 分别记录 H50 与前 H15 的 flow error，明确 action padding 与 EEF16 维度处理；不从已归约的标量推测 H15。
3. 当前 student residual-off 与固定 teacher 单独命名；没有加载并核验固定 teacher 时明确 unavailable，不能冒充已测 Base 对照。
4. 测量或明确标注重建的 context/prior 残差和相对幅度，不用 gate 数值代替。额外 forward 必须恢复 RNG 和临时注入状态。
5. 单测完成后，先对既定 ZeVA005000 在独立验证样本上做只读重放；保留 checkpoint SHA、样本、噪声与运行时来源。正式 rollout 成功标签不进入这个测量。

这些是当前实施要求，不代表已经获得新的数值。下一轮训练须根据实际观测确定单一可解释改动，继续冻结 ZTE/bank/VLM，保留 AE LR5e-6、新模块 LR5e-5、global256 和现有双残差/Gaussian prior 路径。

若采用额外 action-expert continuation，必须给普通 Base 匹配的额外训练预算与数据；只给 ZeVA 增加 AE 更新不能称为表征增益的干净隔离实验。未启动新的 Stage1、Stage2 或 Stage3。
