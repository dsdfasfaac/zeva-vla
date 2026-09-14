# ZTE v2：配对评测后的诊断边界与下一步

更新：2026-09-14。本文区分已测事实、未测假设和计划，不改变已完成实验。

最新运行状态（09-14 13:31 heartbeat 后核查）：显式 SAPIEN 选卡已通过真实 client hook smoke。`CUDA_VISIBLE_DEVICES=6` + `ZEVA_SAPIEN_RENDER_DEVICE=cuda:0`，通过 `sapien.core.SapienRenderer()` 创建的 PID3447001 经 nvidia-smi 确认在物理 GPU6 / PCI BA:00.0，输出480×640×4帧。证据转录见[选卡验证](results/robotwin-fixed-anchor-20260914/renderer-device-smoke.json)；原独立日志未保存，不能冒充原始日志。4项client测试、4项映射测试和3项只读runtime测试本地通过，修复已部署隔离release，未改公共RoboTwin。

本轮启动前发现新的资源健康阻塞：aigc24 SSH/hostname正常，但全局 nvidia-smi 超过2分钟不返回，单独GPU2/6查询也在8秒超时后强制结束（rc137）；进程出现D态，其他任务的查询同时挂起。尚不能确定驱动或硬件根因。未重置GPU、未停止他人任务，正式输出目录仍不存在。专用入口增加GPU查询超时并失败即停止；正在只读检查其他授权节点，不能声称正式闭环已经开始。训练与checkpoint不变，无新成功率。

14:07 heartbeat复核：aigc24的限时GPU查询仍以rc137结束，正式目录仍不存在。Luna上一轮完成其余7台授权节点的完整进程检查，未发现至少2张无占用的GPU：31每卡约3.8GiB且同一计算进程占据8卡；32/29/15/28约75–80GiB，01/14约51GiB。不能把31利用率0%视为无人占用。当前保持等待安全资源，不重置驱动、不抢占他人进程；这不是新的模型或评测结果。

14:08追加快照：aigc31每卡仍约3809MiB/0%，但默认nvidia-smi进程表变为`No running processes found`，与上一轮PIDS查询结果不一致。已交Luna核实PIDS/compute-apps及设备使用者，未据此宣布卡空闲或启动新进程。

## 最新：固定teacher方案未证明增量收益，准备固定末步闭环

两路终检于09-14 02:03均完成735 batches/5874决策。checkpoint、adapter、Stage1/bank/live/retrieval、样本顺序SHA、seed1000、batch8、源码及预处理协议核对一致，两路student residual-on/off统计完全相同；两种teacher的冻结权重均逐张量核验相同。传回本地曾遇SSH/SFTP挂起，终止本次传输进程并通过带超时的rsync恢复，未影响已完成的训练/验证。

| 固定同样本、同噪声的 H15 路径 | 样本平均 flow error |
|---|---:|
| 训练起点 Base004500 | 0.010180731304 |
| 同预算新 Base001000 | 0.010030957870 |
| 新 ZeVA001000，关闭残差 | 0.010142331943 |
| 新 ZeVA001000，开启残差 | 0.010143543594 |

ZeVA相对训练起点改善0.3653%，但新Base改善1.4711%，ZeVA比新Base差1.1224%；开启残差比同权重关闭残差略差0.01195%，逐样本胜率48.47%。H50 residual-on=0.018861809745、off=0.018864864483，H50微小正收益不能替代实际执行H15上的负点估计。prior NLL=11.60375。未计算episode级置信区间，不能宣称统计显著；这些不是成功率。

**预设动作收益检查：相对固定起点不回退满足；普通Base相对起点不回退满足；residual-on应优于off不满足，且ZeVA不及同预算Base。** 因而不支持“仅换固定teacher足以提供ZTE增益”，不继续该方案的盲目加步数/倍率搜索。它仍未证明Stage1编码器无效。保留预定step1000做正常配对闭环，不能把负的离线点估计直接解释成成功率下降，也不根据正式标签改选checkpoint。

固定产物：ZeVA model SHA=`f5e4812a01e01da9936ee23a8f2c9e7d5e70037112f821e759a329d37bc4d11a`，adapter=`cbf0100994d43d7faa2aba23d4f38ae665331f565df53636d6435654c25a1c57`，新Base model=`bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17`。[对固定起点原始报告](results/robotwin-fixed-anchor-20260914/vs-fixed-anchor.json)、[对同预算Base原始报告](results/robotwin-fixed-anchor-20260914/vs-matched-base.json)。

闭环将复用原`eval/formal-ztev2-selected-pair-20260912/seed_manifest.json`（SHA=`1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b`）中的200组seed/实际seen指令，不重筛；本地已有原manifest，哈希再次一致。维持10tasks、Large_D435640×480、demo_randomized语义的`zeva_randomized`配置、H50/H15、视频审计。拟同时重跑untouched Anchor核验正常性，Base门槛仍为max(同条件Anchor,57%)；不放松。当前Luna正在检查空闲renderer并做隔离初始化smoke；正式闭环尚未启动。

配置预检首次SSH调用超时，但后续经aigc24共享路径确认远端已成功生成3个配置及完整manifest（mtime09-14 08:10），没有重跑或覆盖目录。新Base/ZeVA的model SHA与完整诊断报告吻合，train-only bank/语言及物理协议通过检查。[配置预检原始manifest](results/robotwin-fixed-anchor-20260914/eval-staging/robotwin_eval_ztev2_staging_manifest.json)，远端目录为`RUNROOT/eval/formal-fixed-anchor-pair-20260914-staging`。正式launcher增加可选`READ_ONLY_RUNTIME=true`和`NATIVE_TRANSFORMERS_RUNTIME`，复用已验证共享overlay，不改公共runtime symlink；3项隔离shell分支/语法测试通过。默认历史分支保持不变，新开关不改推理数学或模型权重。

09-14中午资源复核：不能把`--query-compute-apps`为空当作GPU空闲。aigc24多张卡上有他人的RLBench **G类图形进程**，显存仅约141MiB但利用率100%；未停止或占用这些进程。确认可用的是物理GPU2/6；aigc29和其余授权H100均有训练占用。Luna已完成按slot显式`MODEL_GPU_IDS`/`RENDER_GPU_IDS`映射，4项映射测试及3项只读runtime测试通过。计划在aigc24同机使用映射`2,6,2,6,2,6,2,6`，保留8个独立model RNG流及原任务分配，不把逻辑slots改为2；每张卡需承载4个模型/renderer，启动后必须监测实际显存，不能预先保证性能。

专用入口为`scripts/robotwin_eval/launch_fixed_anchor_pair_20260914.sh`，已部署于同一隔离release；它要求新输出目录、再次确认GPU2/6空闲、核对seed/task/model哈希和19300–19307端口。当前仍在完成SAPIEN实际物理选卡核验，**正式入口尚未执行**；不能凭CUDA环境变量设置就声称不会落到其他卡。前次普通相机smoke验证了640×480渲染，但不替代PID→物理GPU的核验。此前中断保留了映射文件，恢复后未重复启动评测。

随后物理选卡smoke发现真实问题：`CUDA_VISIBLE_DEVICES=6`下PID3399952的SAPIEN **G进程实际位于GPU0/PCI18:00.0**，而GPU6/BA:00.0只有Xorg；smoke自然退出。这证明仅传CUDA映射不足以约束Vulkan renderer。正式入口继续暂停；Luna正在验证显式SapienRenderer设备参数/最小process-local钩子，不能改公共RoboTwin runtime。专用入口将要求保存的`renderer-device-smoke.json`证明与显式设备选择一致，否则拒绝启动。不要运行旧的未验证CUDA-only映射。

历史记录（2026-09-14 01:40核查）：两支均完成1000步。Base于01:24:59、ZeVA于01:30:46完成，两个launcher进程退出，`COMPLETE`及`latest.json step=1000`均确认；每支model.safetensors为9354050752字节，均有optimizer/scheduler training_state，ZeVA另有adapter。训练循环含验证/保存耗时分别45分07秒、50分42秒，另有启动开销。[Base完整manifest与末步记录](results/robotwin-fixed-anchor-20260914/baseline/manifest.json)、[ZeVA完整manifest与末步记录](results/robotwin-fixed-anchor-20260914/zeva/manifest.json)已归档。末步旧口径H50 validation：Base自身flow=0.02059016；ZeVA flow=0.02085952、同次固定teacher=0.02076325（该次ZeVA略差），NLL=11.61310。两支validation RNG状态不同，不能将两个flow直接当同噪声paired比较，也不能据此回头挑其他checkpoint。

预设末步的完整只读终检已启动：隔离目录`/mnt/100T/users/dingxin/VLA/fixed-anchor-pair-20260914-P8hRUO`下，aigc29 GPU0/PID2767372输出`vs-fixed-anchor.json`（teacher=原Base004500），GPU1/PID2767373输出`vs-matched-base.json`（teacher=本轮Base001000）；同名`.log`记录过程。两路均使用ZeVA001000、batch8、seed1000、eval_batches0覆盖完整5874决策，分别给出H15/H50、current residual-off和固定teacher结果。它们不创建optimizer或修改checkpoint。当前仍在加载，没有终检结论或新闭环成功率。Luna同时只读核对后续正式评测的固定seed/实际instruction复用与启动协议，不重筛测试样本。

历史进度（2026-09-14 01:09核查）：固定teacher匹配训练正常运行。aigc29 GPU0–3为ZeVA（launcher PID2701483），GPU4–7为普通Base（PID2701482），各4卡、global256、1000新optimizer steps。Base已超过576步（000500完整checkpoint可用），ZeVA正在500步验证/保存（上次确认完整为000250，不能把当时空的000500目录当成完整产物）。稳态训练约Base1.9秒/步、ZeVA2.1–2.3秒/步，不包含阶段性验证/保存。active flow/NLL有限；Base的prior/retrieval NaN是禁用项，不是训练发散。实际manifest确认两支初始化相同、AE430098464参数/LR5e-6；ZeVA另有2910532参数/LR5e-5，teacher=`independent_frozen_base_action_path`、每4步抽样并同噪声重放，scheduler每optimizer step推进一次。真实compiled anchor/zero-init/H15预检及6项launcher契约测试均通过。训练输出根为`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914`，各分支`train.log`、`manifest.json`和checkpoint写在其`baseline/`、`zeva/`下。隔离代码与外层启动日志位于`/mnt/100T/users/dingxin/VLA/fixed-anchor-pair-20260914-P8hRUO`，分别`baseline-launch.log`、`zeva-launch.log`。不要启动第二份，也不要将此新实验伪装成旧ZeVA resume。

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

### 下一轮：固定 Base teacher 的匹配训练（2026-09-14，已提交启动）

Luna的监督/梯度审查没有发现NLL遗漏反传：NLL监督prior及上游context表示，flow监督action expert和两路注入投影；prior dropout只作用于注入而不关闭NLL。没有据此修改NLL、倍率或Stage1。已确认的可改机制是preservation当前默认使用随student更新的residual-off对照；它不是独立Base能力锚点。

计划让普通Base和ZeVA都从既定`baseline/004500`出发，各再训练1000 optimizer steps、warmup100、每250步存档，固定step1000为比较候选。ZeVA使用同一004500作为immutable preservation teacher并初始化新的零残差adapter；不加载旧ZeVA005000 adapter。两支均保留指定best-v1起源、共享历史4500步和同等后续AE预算，optimizer重新建立，明确是**新实验，不是旧ZeVA续训**。冻结ZTE/bank/VLM、AE LR5e-6、新模块LR5e-5、global256、双残差/Gaussian NLL、dropout0.4、H50输出/H15执行及门控初始化均保持。共有初始化和预算变化意味着不能把它与旧实验的差异解释为纯单因素teacher因果效应。

**边界：现有hinge是`relu(student_flow - teacher_flow.detach())`，只对较差样本增加专家标签的flow梯度权重，不是teacher动作蒸馏，更不保证策略能力被保留。** 固定teacher是否有益必须实际测量。训练后分别报告fixed anchor、current residual-off及同预算新Base的样本平均H15/H50；不拿H50替代H15，不以正式成功标签选步或倍率。若没有residual正增益或出现能力退化，记录不支持此假设，不把该轮包装成成功。1000步是有界验证预算，不代表已承诺训练充分或可交付。

真实模型预检位于`/mnt/100T/users/dingxin/VLA/fixed-anchor-pair-20260914-P8hRUO`，smoke PID2691401已完成，报告`passed=true`：共同Base初始化的零残差on/off均bit-exact、H15 transition=1；`torch.compile`下扰动student后teacher不变，调用后student权重正确恢复。模型SHA为既定`2f106633…`，Stage1/完整bank/live来源匹配。[原始smoke报告](results/robotwin-fixed-anchor-20260914/compiled-anchor-smoke.json)。预检使用合成图像/动作，只验证前向实现，不证明真实样本训练反传或任务表现。aigc29 `/data1`仅约2.2GB空余，新增checkpoint和编译缓存全部置于`/mnt/100T`；不清理或覆盖他人文件。随后提交两支训练，状态见顶部。[预注册配置](../configs/robotwin_ztev2_fixed_anchor_pair_20260914.json)固定预算、lineage和末步候选；6项标准库launcher契约测试在本地及aigc29均通过，不将它们冒充真实训练反传测试。

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
