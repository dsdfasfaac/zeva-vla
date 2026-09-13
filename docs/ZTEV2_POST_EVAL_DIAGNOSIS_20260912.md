# ZTE v2：配对评测后的诊断边界与下一步

更新：2026-09-13。本文区分已测事实、未测假设和计划，不改变已完成实验。

## 当前结论：全量验证已完成

**最新执行状态（2026-09-13 23:56 核查）：四组完整消融均已结束。** aigc29 GPU0/1/2/3上的context-only（PID2655485）、prior-only（2655489）、prior×50（2655493）和同机原始1/1对照（2655497）均完成735 batches，结果文件时间约23:35，进程已退出。每组5874决策、batch8、seed1000，OMP/MKL各8线程，checkpoint及冻结Base不变；同机对照用于控制从aigc24迁移带来的环境差异。没有启动新训练或新的闭环评测。

### 同机四路消融结果

| 配置（context倍率 / prior倍率） | 样本平均 H15 flow，越低越好 | 相对关闭双残差改善 | 样本平均 H50 flow |
|---|---:|---:|---:|
| 原始对照 1 / 1 | 0.010179412551 | +0.05914% | 0.018888637424 |
| context-only 1 / 0 | 0.010177855380 | +0.07443% | 0.018888257444 |
| prior-only 0 / 1 | 0.010185945779 | −0.00500% | 0.018895527348 |
| prior 放大 1 / 50 | 0.010180360638 | +0.04983% | 0.018888372928 |

四组的checkpoint、adapter、fixed teacher、全部lineage、诊断源码哈希及除干预倍率外的protocol逐项相同；均`complete=true`、5874决策。样本顺序SHA一致，关闭双残差的H15均值均为`0.010185436345636845`，固定Base的H15均值均为`0.010180731303989887`。新同机原始对照的H15/H50均值还精确复现了先前aigc24报告。这里的均值一致支持对照可比性，不冒充逐样本误差的bit-exact核验。

**可支持的结论：当前权重下，弱小的离线收益主要来自context分支；prior单独没有降低平均H15误差，加入context后反而略抵消其收益。简单把prior放大50倍没有改善H15，不能将“gate太小”当成充分解释。** H50放大后略好、H15略差，进一步说明部署前15步必须单列观察。差值很小，未计算episode级置信区间，不能声称统计显著，也不能推出ZTE信息本身无效。未经该倍率训练的压力测试不等价于重新训练后的结果。

原始报告：[原始对照](results/robotwin-ztev2-20260912/same_host_control.json)、[context-only](results/robotwin-ztev2-20260912/context_only.json)、[prior-only](results/robotwin-ztev2-20260912/prior_only.json)、[prior×50](results/robotwin-ztev2-20260912/prior_strength_test.json)、[运行计划](results/robotwin-ztev2-20260912/plan.json)。这些结果不能用作正式测试选checkpoint或直接决定部署倍率；不据此删除用户指定的prior路线。下一步收敛到prior监督/梯度路径的最小训练改动审查，而不是继续扫gate。

输出目录：`/mnt/100T/users/dingxin/VLA/diagnostics-ztev2-ablation-20260913-fGari2/full-fourway`，包含四组各自的`.pid`、`.log`，完成后写同名`.json`；`plan.json`固定实验设置。单次启动脚本有空闲GPU检查、输出目录拒绝覆盖与独立日志，不是另建监视任务。

Luna已确认已有a29/a24数据全量identity reports的四组件指纹完全相同，报告自身SHA与恢复provenance一致；当前a29 adapter SHA=`8ac54abcec7704b0111b7c28be3fb3a18e27e0ebe8e3dcf8ddff948b36100f8f`匹配原证明。此次只复核报告和adapter，未重读80GB原始数据。显式使用a29原路径`/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data`，不修改历史manifest里的a24路径。

a29未安装pytest（测试命令未执行，不计通过）；8项隔离CPU Tensor/控制流检查在Torch2.7.1+cu126上通过。真实prior×50预检也通过：模型/adapter/fixedBase SHA一致、冻结路径逐张量一致、样本顺序SHA=`a9c6a8e30aa3f7ecbfcb9ce81d6a141ef563300d41e339a22d06fb75a0418ac1`；H50 replay差值0。[两条预检记录](results/robotwin-ztev2-20260912/ablation-preflight-aigc29-prior50-b2x1.json)仅用于实现核验，不能外推总体收益。

固定 ZeVA005000 / Base004500 的 **5874 个验证决策、735 batches** 已全部完成（结果文件时间 09-13 01:52；验证循环17分48秒），两模型冻结路径逐张量一致。模型、adapter、ZTE、bank、retrieval 和 normalization SHA 与既定产物匹配。[完整原始报告](results/robotwin-ztev2-20260912/step5000-diagnostics-full-base4500-b8.json)。

| 相同样本、相同 flow noise 的 H15 指标 | 平均误差 |
|---|---:|
| ZeVA 当前 action expert，关闭双残差 | 0.0101854363 |
| ZeVA 当前 action expert，开启双残差 | 0.0101794126 |
| 独立训练的固定 Base004500 | 0.0101807313 |

双残差带来的平均相对改善仅 **0.0591%**，对固定 Base 的平均相对改善仅 **0.0130%**。这不是成功率，也没有据此声称统计显著。残差开启相对关闭的样本胜率为50.85%，相对固定Base为46.75%；同一episode内的决策有关联，不能当成5874次独立闭环试验。

H15 context/prior 残差相对 noisy-action embedding 的**重建范数比**分别为 `0.0034690`（约0.347%）和 `0.0000149977`（约0.00150%），prior 约为context的1/231。retrieval accuracy=99.8469%，prior NLL=6.76919。这说明目前接入带来的动作预测增量非常小，但不独自证明ZTE没有信息，也不证明增大gate一定有效。

原验证 `flow` 对batch均值再平均，末批只有2条；新诊断按样本平均，因此本次原指标与新H50聚合不作精确等价声明。单批无末批加权差异的真实smoke已验证差值0。当前数据未提供显式action padding mask，诊断使用`implicit_all_action_steps_valid`；不能据此断言轨迹末尾没有重复填充。

上述预先指定的验证集分支消融已完成，结果见顶部；不是正式测试选参，也未变更部署权重。保持同5874决策、batch8、seed1000、固定Base及所有物理协议。[预先固定的消融设置](../configs/robotwin_ztev2_validation_ablation_20260913.json)。放大后变差也不能单独证明ZTE无信息，因为当前权重并非在该幅度训练。

历史资源与实现状态（09-13下午，已被顶部启动记录覆盖）：Luna连续遭遇transport错误，root接手完成只读门控开关，默认1/1不替换方法；异常退出也恢复原方法。8个本地隔离控制流用例通过，当时未作完整tensor/runtime测试。8台授权H100当时全部高负载，未抢占他人作业；随后释放资源、完成真实预检并启动消融。没有新训练。

## 历史过程：2026-09-13 验证进度

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

## 历史实施要求：先增加验证观测（已完成）

该阶段由Luna修改本地Stage2 trainer和专门测试，随后在隔离远程副本完成验证；未覆盖生产源码、未启动新训练。原实施要求：

1. 不改变历史 loss、模型权重、门控、优化器、训练步数或默认验证输出。
2. 分别记录 H50 与前 H15 的 flow error，明确 action padding 与 EEF16 维度处理；不从已归约的标量推测 H15。
3. 当前 student residual-off 与固定 teacher 单独命名；没有加载并核验固定 teacher 时明确 unavailable，不能冒充已测 Base 对照。
4. 测量或明确标注重建的 context/prior 残差和相对幅度，不用 gate 数值代替。额外 forward 必须恢复 RNG 和临时注入状态。
5. 单测完成后，先对既定 ZeVA005000 在独立验证样本上做只读重放；保留 checkpoint SHA、样本、噪声与运行时来源。正式 rollout 成功标签不进入这个测量。

上述验证现已完成，数值见本文顶部。下一轮训练须根据实际观测确定单一可解释改动，继续冻结 ZTE/bank/VLM，保留 AE LR5e-6、新模块 LR5e-5、global256 和现有双残差/Gaussian prior 路径。

若采用额外 action-expert continuation，必须给普通 Base 匹配的额外训练预算与数据；只给 ZeVA 增加 AE 更新不能称为表征增益的干净隔离实验。未启动新的 Stage1、Stage2 或 Stage3。
