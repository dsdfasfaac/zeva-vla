# ZTE v2：配对评测后的诊断边界与下一步

更新：2026-09-12。本文区分已测事实、未测假设和计划，不改变已完成实验。

## 2026-09-13 验证进度

01:32 更新：真实 ZeVA005000 的单批 smoke 已完成，模型/adapter/ZTE/bank/retrieval SHA 与正式选定产物完全匹配。两条真实验证决策上，原 H50 flow 与 raw replay 均为 `0.0061195502`，差值为零；H15 residual-on=`0.0051624347`，current residual-off=`0.0053198677`。同批 H50 略差而 H15 略好，仅说明两种观测不能互相替代，**不能从两条样本外推总体增益**。H15 context/prior 相对范数的重建均值分别约 `0.002976` / `0.00001348`，不是直接捕获的 BF16 累加后差值。完整 smoke JSON 已[归档](results/robotwin-ztev2-20260912/step5000-diagnostics-smoke-b2x1.json)。

独立只读 CLI 与加载器已完成，相关远程测试累计 **16 passed in 15.34s**。首次固定 Base 对照加载被原 `load_foundation_anchor` 的起点一致性保护拒绝，未进行验证：该方法要求先在 teacher 权重上建立 anchor。修复为先逐张量确认两模型的冻结路径相同，再加载 Base004500、建立 anchor、恢复 ZeVA005000；未修改 policy 或训练权重。失败日志保留在原隔离目录，不伪装成成功。

完整 **5874 个 validation5 决策**的只读测量已在 aigc24 GPU6 启动，batch8、单进程、不 compile、无 optimizer；同时分别比较 current residual-off 与固定 Base004500。新隔离目录为 `/mnt/100T/users/dingxin/VLA/diagnostics-ztev2-fixedteacher-20260913-a5evvD`，日志 `step5000-full-base4500-b8.log`，完成后独占写入同名前缀 JSON。此时仍在加载，尚未声称完整测量已完成。正式测试成功标签不输入此工作流，未启动新训练。

- 默认关闭的诊断实现已完成初版；远程 aigc24 隔离目录 `diagnostics-ztev2-20260913-lubDbq` 中，5 项新诊断测试与 5 项既有 Stage2 v2 测试实际通过（13.13 秒），不是本地缺 PyTorch 导致的 skip。生产源码与已有 checkpoint 未覆盖。
- 代码审查发现可选 raw-forward 的失败可能因 rank 而异，而其 gather 是条件调用；已限定诊断为单进程，避免多卡条件 collective 挂起。普通训练默认路径不受影响。
- 加入单进程保护回归测试后，远程复测为 **11 passed in 9.30s**。
- 核对实际 handoff `PI05Policy.forward`：历史 flow 截取 EEF16 后对 H50/动作维取均值，并不使用 `action_is_pad`。新增 valid-only H15/H50 是另行标注的诊断统计，不能在有 padding 时冒充历史指标的精确重放；这项发现本身还不证明闭环掉点由 padding 引起。
- 只读 checkpoint 验证入口正在实现，尚未得到真实权重上的 H15 或注入幅度数值。context/prior 分量根据实际投影、gate、confidence 和 embedding dtype 重建；不声称已经直接捕获了 BF16 累加后的有效差值。需要先核对重放结果，再据结果决定训练策略。

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
