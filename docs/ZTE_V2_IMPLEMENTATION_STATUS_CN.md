# ZTE v2 实现验收记录

日期：2026-09-11。本文记录已执行的验证；设计文档中的目标不代表已经实现或通过。

执行口径更新：辅助表征指标是诊断，不再是全项通过后才能接 PI 的门槛。当前优先完成 v2 → action-expert 双残差/Gaussian prior 接入和公平的十任务闭环比较。计划中的 legacy-Huber 4096-step 对照与扩展旧/新 frozen-phase probe 已取消，代码保留但不声称运行。

## 当前运行

2026-09-12 14:13（北京时间）正式pipeline已启动，模型server仍在加载：aigc24 launcher4129168（外层shell4129167），输出 `eval/formal-ztev2-selected-pair-20260912`。使用独立 `eval-release-ztev2-20260912/scripts/robotwin_eval` 快照，保留远端原评测脚本不覆盖，src/configs指向已核验源码。model/render均aigc24，显式MODEL_IP=172.16.80.158，8 slots；按Base→Anchor→ZeVA运行。各200episodes，Base先从1000筛首20 expert-valid seeds并保存实际指令，后两支精确重放，视频启用。Large_D435640×480、seen、demo_randomized相同随机化（配置名zeva_randomized仅改相机）、H50/H15等协议固定。正常性报告门槛保留Base≥max(同seed Anchor,57%)且ZeVA>Base，不按结果回头换checkpoint。此时无已完成episode或成功率。

2026-09-12 14:04（北京时间）核验：ZeVA正常完成5000步，PID4082455已退出，完整 `005000/model.safetensors`、`zeva_adapter.pth`、`training_state.pt` 及latest.json已保存；aigc24八卡释放。Base与ZeVA两支5000预算均完成。按预定独立held-out flow选步，Base004500=0.01872689090669155，ZeVA005000=0.01828256994485855；选择记录 `advantage10-ztev2-schedulerfix-pair-20260911/checkpoint_selection_final_20260912.json`。Base模型SHA=`2f106633403e5f2146bf7cd4f56858cbfdb856e9b2966d1c724a79ac4948c84f`，ZeVA模型SHA=`4ea8f6ca9a806b297e4881197041b222dedf29e25680aea44d2d5c465162611b`，adapter SHA=`967b78f509e4ba8e74f40967496f740c81149526ff2b46c913bcc81552f51e02`。进入正式10×20同seed评测preflight；offline flow不等于成功率提升。

恢复启动后核验：四个rank均输出 `All keys loaded successfully`，训练循环显示4500/5000，当前开始首步编译。实际新manifest与原备份的完整递归diff严格只有 `dataset_adapter`、`train_args.dataset_root`、`train_args.resume_checkpoint`；runtime版本、world4、global256、LR/损失/冻结、task samples及全部模型来源字段一致。尚未把起始计数4500当作已产生新optimizer step。

2026-09-12 13:30（北京时间）恢复已启动，仍在加载：aigc24 PID4082455，GPU0/1/3/4，从完整4500补齐5000预算，保留四卡×16×acc4=global256、LR5e-6/5e-5、所有冻结和损失设置。入口 `scripts/resume_robotwin_stage2_verified.py` 核对原trainer/policy SHA、源state/manifest、scheduler.last_epoch=4500及optimizer LR；`resume_provenance.json` 保存源model/adapter/training_state和manifest SHA，旧checkpoint不改写。未声称已经推进新step或开展正式评测。

跨主机数据证明：`dataset-identity-aigc{29,24}-stage2-resume-20260912.json` 各完整读取source110702、eef606、joint503及stats1个文件；四组件content SHA、文件数/字节数、去绝对路径后的adapter语义完全一致。训练入口由 `/data1/huangbingjia/.../data` 显式迁移到aigc24的 `/data1/dingxin/.../data`。选择器只在两份hash报告核验通过后允许dataset_root/dataset_adapter路径变化，仍拒绝其它科学配置改变；旧≤4500与新5000 checkpoint分别严格匹配各自manifest。selector16项、迁移helper2项单测通过。

环境：aigc24原Accelerate1.11与原训练1.13不一致，已将aigc29的1.13包复制到隔离 `/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912`，未修改公共库。实际Torch2.7.1+cu126、TF5.5.4、Accelerate1.13、tokenizers dist0.22.2/module0.21.4、TorchCodec0.5+cu128对齐。辅助库仍有差异：aigc29/aigc24为NumPy2.2.6/1.26.4、Pillow12.0/11.1、tyro1.0.13/0.9.35、psutil5.9.6/7.0、PyYAML5.4.1/6.0.2；不声称完整环境字节一致，加上RNG未保存，明确为non-bit-exact continuation。

2026-09-12 00:18（北京时间）复核：aigc29、31、32、24、15、01、14、28的全部64张GPU均约72–78GiB占用、利用率100%；不抢占其他作业，恢复尚未启动。Base已完成，ZeVA仍以完整4500为恢复源；正在离线准备严格校验旧manifest与恢复记录的选步兼容逻辑，不能绕过科学配置和来源校验。README和方法纵览已同步中断状态，本轮无正式成功率。

23:30 更新：ZeVA未正常完成5000步。原launcher751929已退出，runtimefix2日志显示23:07:47 rank2收到SIGTERM（exit -15），最后完整checkpoint为4500。信号来源尚未查明，不将其误报为OOM或模型发散。正在检查资源与恢复条件，计划从完整4500保留model/adapter/optimizer/scheduler补齐剩余预算；当前Stage2 checkpoint未保存完整RNG，若恢复须明确为非bit-exact continuation。

自动waiter另有自匹配bug：`ps|awk`把含训练脚本字符串的awk自身当成训练worker，因而无法退出等待。修复9fd3c72限定Python进程并通过回归测试；已仅停止旧waiter823924/823926，原eval目录仅log/state，无选步或rollout结果，完整保留。恢复真实训练进程后需新建waiter，不能把旧等待状态当作评测已运行。aigc29当时显存高占用但compute进程列表为空，未重置GPU或停止其他作业。

21:37 更新：普通Base完成5000steps，`baseline/005000` 与 `latest.json` 已保存，launcher710436不再存活；ZeVA仍在运行（约2829/5000）。双方完成后将使用 `scripts/robotwin_eval/select_robotwin_ztev2_pair.py` 分别按held-out flow选择；该selector的7项单测及真实mmap checkpoint元数据读取已通过。评测preflight使用真实Stage2 adapter/bank/retrieval核对且三份配置均通过真实server parser，尚未开展正式rollouts。当前正在接通完成后自动选步→preflight→Base/ZeVA/Anchor同seed评测的持久流程。

20:52 只读runtime审计完成：Base和ZeVA的foundation config/model/tokenizer SHA、trainer/policy SHA、runtime版本和实际导入PI0/PaliGemma/Gemma源码SHA均一致。两个native overlay只有同样5个symlink，均指向同一个handoff源；ZeVA隔离路径不引入不同数学实现。实际使用Transformers5.5.4，不是后置依赖目录中的4.53；tokenizers module0.21.4与distribution0.22.2的差异两支相同且已记录。当前Base已保存3500、ZeVA已保存1500并继续训练，无新的闭环结果。

20:06 更新：Base与ZeVA两个匹配Stage2进程均实际运行。Base PID710436，已保存step2500；ZeVA PID751929，GPU0/1/3/4、port29572，日志到step805，已保存step500。两者均5000steps、global256、AE LR5e-6；ZeVA新增模块LR5e-5、Gaussian NLL与prior residual dropout0.4，视觉语言backbone/ZTE/bank冻结。ZeVA最终产物为 `stage1-zte-v2-artifacts-scheduler-repaired-20260911`，完整26150/1350条、50tasks，retrieval train=98.4206%、validation=98.2963%，SHA连接核验完成。

ZeVA step500 validation flow=0.0192441、retrieval=99.8471%，同一ZeVA模型内部residual-off对照的平均flow改善约2.61e-6；这不是独立训练Base与ZeVA的闭环比较，不能声称成功率提升。Base step2500 validation flow=0.0191362。零注入真实best-v1+选定step4096联合smoke通过，report=`advantage10-zeva-foundation-smoke-step4096-20260911.json`。ZeVA使用独立runtime目录 `/mnt/100T/users/dingxin/VLA/runtime/transformers5-runtime-zeva-20260911`，启动log=`advantage10-ztev2-zeva-launch-20260911-runtimefix2.log`；正在补充它与Base原生runtime的源码一致性核对，不能仅凭版本字符串断言数学实现一致。

19:00 更新：修复后的Stage1已完成step4096，按完整validation loss选择不可变 `zte_v2_step_004096.pth`（SHA=`ce9402981b8e2f1ce597e2cb150b795ae246f84fb81b3bb72fae023a3decf899`）。最终1350条验证：loss=6.190151、task=96.8889%、order=88.6432%、effect cosine=0.375731、language consistency=0.851242。选择记录在修复run的 `checkpoint_selection.json`；不使用闭环结果选这个Stage1 checkpoint。

普通Base已实际开训：`advantage10-ztev2-schedulerfix-pair-20260911/baseline`，launcher710436，GPU2/5/6/7、port29571，global256（16×acc4×4卡），AE LR5e-6，compile/TorchCodec启用。step500已保存8.8GB `model.safetensors` 和2.1GB `training_state.pt`，validation flow=0.0190017；未启用prior/retrieval对应NaN字段表示不适用，不是flow发散。Base manifest明确ZTE仅作历史接口/同源检查、不用于预测或loss，BIT/PIM关闭，仅action expert及其投影可训练。

最终新bank/live/retrieval工作流已在GPU0/1/3/4实际启动，输出 `stage1-zte-v2-artifacts-scheduler-repaired-20260911`。ZeVA匹配训练安排在这套产物完整验证、检索训练和同checkpoint联合smoke后接续启动；截至本条核验尚未声称ZeVA进入optimizer steps。旧step2048产物不用于这次ZeVA策略训练。

18:12 更新：调度修复 `b4fbb0e` 已通过31项测试并实际补训。新目录 `stage1-zte-v2-phase-vector-mse-4096-20260911h-scheduler-repair-20260911i`，launcher `672753`，从旧step1024保留权重/optimizer/RNG，但显式重设 scheduler 到 global step1024；manifest 标记 `non_exact_scheduler_repaired_continuation`，不是 exact resume。step1536 的 scheduler.last_epoch=1536、LR=`[6.913417e-6,6.913417e-5]`，确认恢复了有效学习。最新step2560完整1350条验证：task=**96.1481%**、order=**87.3814%**、effect cosine=**0.350909**、language consistency=**0.848188**，四项均达到配置中的诊断参考；不把这个结果等同闭环成功率。

旧临时候选step2048的全量导出已完成：`stage1-zte-v2-artifacts-step2048-20260911`，26150 train +1350 validation、50tasks、`usable_for_training=true`，checkpoint SHA=`d3d21c0acac9855a0a06311472d9ad773d914b57df311cf2648bafa9bc14bc5d`，task-language retrieval train=97.8891%、validation=97.7037%。为避免用已明显落后的表征启动ZeVA长训，**ZeVA正式训练改用补训完成后选择的不可变checkpoint及重新导出的同源bank/live/retrieval**；旧产物保留作工程验证与Base的无记忆数据接口依赖，不静默替换任何权重。

已安排先在空闲4卡启动普通Base，另4卡完成Stage1补训后用于最终导出；Base与ZeVA保持同best-v1、十任务、5000steps、global256和相同物理协议。Base关闭全部ZeVA注入，其无记忆训练不得依赖所挂载ZTE/bank的内容；该性质需在启动审计确认。此处是启动安排，不代表Base已推进optimizer steps。

17:09 更新：恢复进程已结束并保存 step4096，但 checkpoint 审计发现 **学习率调度计数错误**，不能把这轮称为正常完成5epochs。保存的 `(global step, scheduler.last_epoch, LR)` 为 `(512,2048,[5e-6,5e-5])`、`(1024,4096,[0,0])`、`(2048,8192,[0,0])`、`(4096,16384,[0,0])`。Accelerate 包装导致每个 global step 推进4次，step1024 后无有效参数学习；后续 EMA 收敛使指标几乎不变。正在修复为每个 global step 只推进一次，并从 step1024 做显式 scheduler-repair continuation；这种改变学习率状态的补训不能称为 exact resume。

现有 checkpoint 中 step2048 的完整 validation loss 最低（6.957496039，和末步6.957496073几乎相同），仅选为**已冻结的临时 PI 接入候选**，不是证明它最能提高闭环成功率。其 task probe=90.2222%、order=76.9403%、effect cosine=0.183846。全量 bank/live 导出与同源 task-language retrieval 将使用这个不可变 step 文件；后续修复产生的新 ZTE 不能静默替换对应 bank 或 policy 的权重。

真实 best-v1 PI0.5 + h512 ZTE 联合 smoke 已通过（`zte-v2-bestv1-real-pi-h512-smoke-20260911.json`）：Base/residual-off/on 最大差0，实际 `[1,50,16]` 输出、pending `[1,15,16]`、三类 Mamba cache 和有限动作。输入为 synthetic zero images，不是闭环成功率。26项集成单测通过。完整导出器现支持不同长度 batch、固定大小 CPU bank 统计、分片覆盖与同源校验，8项单测和两分片 merge smoke 通过；这些 bounded smoke 明确标为 incomplete，不可用于正式训练。

当前长预算实验：`stage1-zte-v2-phase-vector-mse-4096-20260911h`，aigc29 原 launcher `513210`，GPU0/1/3/4，DDP port29541，每卡batch8、global32、4096 steps（约5.01epochs）、warmup256、每512步保存并完整验证。step512 已保存，完整验证1350条。原进程在 13:26 因 DDP reduction/未参与梯度参数报错退出，不能称为正常训完。

15:16 恢复核验：Luna worker 的修复 `bc5dbc2` 为 all-padding batch 的输出增加零值 autograd 依赖，避免 epoch 尾部被 mask 掉的 action head/task prototype 分支不触发 DDP reduction。新目录 `stage1-zte-v2-phase-vector-mse-4096-20260911h-ddp-recovery` 保留原 step512 权重、optimizer/scheduler 和逐 rank RNG，world4 与4096步计划不变，原目录不覆盖。launcher `574892` 存活，日志实际推进到 **step999**，已越过原报错位置；这证明恢复在推进，不表示4096步已完成。恢复入口为 `scripts/recover_robotwin_zte_v2_ddp.sh`，已有输出目录会拒绝覆盖。

最新 step512：task probe **78.7407%**，phase order **72.1441%**，effect cosine **0.156680**，language consistency **0.980452**。按配置中的诊断参考（50%、80%、0.05），只有 phase order 尚未达到。该 order 来自独立 progress head，不是 exported-phase probe；本 checkpoint 的 next-H15/task-mean 与 effect/zero-effect 独立比较尚未测。`probe_gate_passed=false` 是代码固定写入的 diagnostic-only 标记，不代表所有指标失败；这些参考阈值也不是 BehaviorVLA 论文规定的通过标准。

`stage1-zte-v2-pilot-20260911d` 与同预算 `stage1-zte-v2-pilot-taskpaired-20260911e` 均已完成 256 steps，分别约 9 分 57 秒和 9 分 4 秒（包含两次完整 validation5）。均使用 aigc29 四张 H100（0/1/3/4），每卡 batch 8，global batch 32，warmup 32；在 step128/256 保存并跑完整 validation5。唯一训练方法变化是同任务跨 episode 配对采样。最新一轮 launcher PID `477001` 已退出，不能把仍存在的 pid 文件当成运行状态。两轮均未通过 Stage1。

该预算是学习曲线与完整验证的 pilot，最多访问 8192 个 episode 样本，远未等同于 40/80 epochs。其职责是证明训练目标可学、观察 held-out 泛化与塌缩情况；不得因此自动放行 Stage2。Stage1 ZTE 使用原有 50-task 表征数据，后续 Base/ZeVA 的训练和正式比较仍在用户选定的 10 个任务上。

## 已验证的信息路径

在 aigc29 H100、PyTorch 2.7.1+cu126、真实 Mamba CUDA 路径执行：

```bash
PYTHONPATH=/data1/dingxin/zeva-runtime-deps:src CUDA_VISIBLE_DEVICES=0 \
ZEVA_TEST_CUDA=1 python3 -m unittest openpi.zeva.transition_encoder_v2_test -v
```

结果：8 tests，全部通过（1.859 秒）。七项使用小型视觉 fixture 隔离时序机制；一项使用真实 ResNet-18 验证训练模式下 BatchNorm 与梯度，不声称测试了预训练权重的任务效果。另有 3 项 representation objectives 测试通过，验证匹配/置换 effect、增强正样本与防塌缩损失的方向。

1. 置换 after-image 不改变 forward-effect/next-action 预测，改变 post-transition causal signal。
2. 修改未来的 before/after/action 不改变过去的 phase、causal signal 或预测。
3. 颠倒 H15 动作顺序改变 pre-context。
4. 右侧 padding 不改变有效 phase 或 masked global pool。
5. 逐步回放与整段编码一致。
6. effect 预测对动作有梯度，对 after-image 没有梯度。
7. 真实视觉网络训练时固定 BatchNorm 统计，未来图像不污染当前预测，视觉参数仍有梯度。
8. 多个 episode 放在同一 batch 时，动作递归不会跨越 batch 维度；修改第二个 episode 不改变第一个。

缓存版本另外验证了 phase、causal、forward-effect、next-action、progress 和 global token 与整段编码一致（容差 `1e-5`），且三个 Mamba cache 的 tensor 数量不随 transition 数增长。8 项测试全部通过；训练用的整段 forward 数学定义保持不变。

## 首轮完整验证与采样归因

| checkpoint | 有效 validation episodes | task probe | phase order | effect cosine |
|---|---:|---:|---:|---:|
| 首轮 step128 | 1350 | 7.037% | 53.026% | 0.00877 |
| 首轮 step256 | 1350 | 7.481% | 53.099% | 0.00838 |
| task-paired step256 | 1350 | 4.148% | 52.635% | 0.00850 |
| phase-action f step256 | 1350 | 4.074% | 52.647% | 0.00849 |

两次均未通过 Stage1。首轮仅约 0.31 epoch，不能据此否定表示结构；同时，完整采样审计定位到一个与目标函数直接冲突的问题：

| 完整 train95 sampler（26150 episodes） | 同卡内跨 episode 同任务正样本覆盖 | 不同任务负样本覆盖 | 每条 episode 恰好一次 |
|---|---:|---:|---|
| interleave + rank stride | 3.4646% | 99.9924% | 是 |
| same-task pairs | 99.8088% | 100% | 是 |

剩余约 0.19% 为各任务奇数条数的尾项，保留完整覆盖，不重复采样。复现审计：`bash scripts/run_robotwin_zte_v2.sh --run scripts/audit_robotwin_zte_sampler.py`。修正针对的是 global SupCon 缺少跨轨迹正样本的机制问题，不以闭环成功率搜索采样规则。

独立诊断使用相同 seed1000、每任务 2 条 train 原型轨迹及 2 条不重叠 validation 轨迹（共 100+100 条），没有用 validation 构建 bank：

| step256 对照 | masked-language 原型检索 | next-H15 MSE 相对任务均值改善 | effect MSE 相对零效果改善 |
|---|---:|---:|---:|
| 原采样 d | 30/100 | +2.21% | -4.59% |
| task-paired e | 39/100 | +1.76% | -6.49% |
| phase-action f | 39/100 | -3.72% | -6.49% |

原型检索与上表的分类头 task probe 是不同指标。检索点估计改善尚无 episode-group 显著性证据；action/effect 没有同步改善。各模型 EMA target 不同，不能直接用两者 effect 绝对 MSE 比较表征质量。这些 `representation_diagnostic_step256.json` 均为 diagnostic-only，不能放行 Stage2。

f 也已完成（约 9 分 16 秒），其独立诊断仍为 diagnostic-only。导出 phase 的动作路径接通并未在 0.31 epoch 短预算内改善动作预测；不据此声称新路径更优。以上 order 指标来自独立 `progress_head(post_context)`，不是对导出 phase token 的 frozen linear probe，后者正在单独设计。

### Next-action 到 phase 的梯度审计

Luna worker 在真实 H100、step128 checkpoint 上仅对 next-action loss 反传：`phase_head.grad=None`、`post_fusion.grad=None`，`pre_fusion` 梯度范数约 0.08877。原实现下一动作只从 pre-context 预测，因此该损失没有直接训练实际导出的 phase。这是结构证据，不是已证明的成功率掉点唯一原因。

新增 `action_prediction_context=phase` 对照让下一段动作直接从导出的 normalized phase 预测；当前 after-state 在 H15 执行完毕后已可见，允许用于下一动作，但 forward-effect 仍必须隐藏 after-state。旧 checkpoint 缺省 `pre`；此变化不得隐式套用旧权重。BehaviorVLA 官方实现实际是 `action_predictor(h_a)` 加 shifted actions，不是从 `h_b` 直接预测；本修正是需要独立验证的 ZeVA 设计。

该对照预先固定 paired sampler、seed1000、256 steps、原 LR/权重和全部验证 episode。新增 head 的维度不能扰动其他公共参数的初始化随机序列；需逐张量测试。验收先验证 action loss 到 phase 的梯度、当前/未来信息边界和 Mamba cache 等价，再比较动作增量与其他表示指标，不能因代码测试通过自动宣称能力提升。

实现 commit `a10b6c9`：真实 H100/Mamba 路径 13/13 tests 通过（3.837 秒），包括同 seed 公共权重逐张量一致、新路径 phase/post 梯度、当前 after 可见与未来不可见、cache 等价。旧 task-paired step256 的 463 个 state keys 以 `strict=True` 加载通过，旧 config 缺省为 `pre`。新实验必须从初始化训练，不能把 pre checkpoint 伪装成 phase resume。

`stage1-zte-v2-pilot-phaseaction-20260911f` 已在 aigc29 完成，launcher `495287` 已退出，step256 checkpoint 与独立诊断均存在。配置为上述唯一监督路径变化，日志/checkpoint/独立诊断与旧 e 对照分目录保存。

f 的原始 manifest 存在文字元数据勘误：旧 `information_flow.jepa_and_action_prediction` 未区分两条 head；其 `zte_config.action_prediction_context=phase` 与实际代码正确。保留原始 manifest 与源文件 hash，不事后改写实验档案。后续 trainer 已改为分别记录 forward-effect 与 next-action 信息边界；这是描述修正，不改变 f 的计算或权重。

该元数据修正与原有 sampler/mask/next-H15 tests 在隔离副本中 6/6 通过（不改写运行中 f 的 source）。GitHub `publish` 可读，但本次 HTTPS push 被 403 拒绝，SSH 也无可用 publickey；本地提交不等于已上传，需恢复仓库写权限后再推送。

恢复兼容补充：trainer 仅为历史 checkpoint 缺少的 `action_prediction_context` 补上已知的 `pre` 默认值，其他缺失设置不继承当前默认值。这样保留旧 pre 配置的语义兼容检查，同时仍拒绝 pre→phase、改变步数/损失/数据顺序的所谓 exact-resume。相关 sampler/targets/manifest/migration 共 7/7 unit tests 在隔离副本通过；这不等于已经执行一次旧长跑 checkpoint 的完整恢复训练。

### 损失梯度与 reduction 的归因

`zte-v2-objective-gradient-audit-pilot-e-step256-gpu2-20260911.json` 对固定 paired sampler 的 4 条真实 train95 episode 做无更新的 eval-mode 梯度审计。pre-fusion 上 action/effect/global/local/causal 的已加权梯度范数均值分别为 0.01334 / 0.002842 / 2.343 / 2.132 / 2.529；不是只比较 loss 数值。该 batch2 没有正常 batch8 的不同任务负样本构成，不能直接泛化，正在扩展正常 batch。

对应的代码级差异已定位：[官方 BehaviorVLA](https://github.com/iLearn-Lab/ICML26-BehaviorVLA/blob/main/src/openpi/BehaviorEncoder/train.py) 对预测坐标先 sum 再平均有效时间；旧 v2 为 coordinate-mean SmoothL1。新 `prediction_loss_reduction=vector_mse` 保留外部权重，MSE sum 最后一维、平均有效 transition 与 H15。单测验证 256 维小误差梯度相对旧值为 512 倍、EEF16 为 32 倍，复制 H15 为 H30 不额外改变 loss，padding 不产生梯度。8/8 trainer tests 已通过。以上倍数是确定性小误差测试结论；真实数据存在 Huber 线性区间，不能机械声称所有实际梯度都精确放大这些倍数。

旧 checkpoint 的 args 缺省补 `mean_coordinate_huber`，明确拒绝将 reduction 改变伪装成 exact-resume。新增控制训练 launcher 预注册 4096 steps（约 5.01 epochs），两支 phase 模型除 prediction reduction 外一致；启动状态以实时进程和独立目录为准，launcher 存在不等于已经开训。

正常 batch8×8 的同样本审计现已完成（两个 JSON 的 `sample_ids_by_batch` 完全相同）：

| pre-fusion 已加权梯度范数均值 | legacy Huber | vector MSE |
|---|---:|---:|
| action | 0.007152 | 0.283823 |
| effect | 0.001461 | 0.754703 |
| global | 6.800477 | 6.800477 |
| local | 0.577194 | 0.577194 |
| causal | 1.455897 | 1.455897 |

报告为 `zte-v2-objective-gradient-audit-pilot-e-step256-gpu2-actckpt-b8x8-20260911.json` 与 `zte-v2-objective-gradient-audit-pilot-e-step256-gpu2-vector-mse-b8x8-20260911.json`。使用 deterministic activation checkpoint 保持正常batch8内的正负样本构成，首个真实batch与eval forward最大差0，Dropout/MHA/BN冻结，EMA digest未变，无optimizer更新。原报告 limitation 文案误写 Two batches，实际 `data.batches=8` 与8组IDs为准；代码已修正动态计数，旧报告不事后改写。

`stage1-zte-v2-vector-mse-smoke-20260911g` 也已完成真实batch8、2steps的前向/反传/验证/保存（约30秒），没有OOM或非有限loss。该验证说明实现可训练，不证明表示已改善。

## 已完成的真实数据 smoke

统一结果根目录：`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/`。

| 实验目录 | 配置 | 已验证结果 |
|---|---|---|
| `stage1-zte-v2-smoke-b2-20260911a` | GPU0，batch 2，2 steps | 真实 TorchCodec 加载、预训练 ResNet、backward、验证、完整 checkpoint 保存完成 |
| `stage1-zte-v2-smoke-b8-20260911b` | GPU0，batch 8，2 steps；新增对比和防塌缩目标 | 完成，无 OOM；验证统计 8 个 episode，诊断 loss 12.7925 |
| `stage1-zte-v2-smoke-ddp2-20260911b` | GPU1/3，每卡 batch 2，2 steps | DDP 完成，验证跨卡汇总 4 个 episode，诊断 loss 6.0612 |

三个实验均保存 model/optimizer/scheduler，均输出 `probe_gate_passed=false`、`probe_gate_status=diagnostic_only`、`validation_complete=false`。这些 smoke 的目标和 batch 数不同，不能比较其总 loss 来判断表征优劣。单卡 batch 8 的两个步骤与一次小验证共 16 秒，仅是短 smoke 测量，不是正式训练耗时预测。

## 验收中修正的问题

- 单 visual-key attention 会让所有 action query 得到相同输出；补回 ordered action-token residual。
- action Mamba 改为按完整 episode 的 H15 子步展开，保留跨 chunk 历史。
- 取消强制单调的 progress 公式，避免顺序指标天然满分。
- next-action auxiliary head 仅预测下一 H15，训练使用时间移位标签；PI 的 H50 契约由独立策略保持。
- 逐步状态保存 raw image，避免再进入 forward 时二次标准化。
- 视觉计算采用 microbatch 与 activation checkpoint，限制全 episode 训练的显存。
- 缺少 Mamba 时 production 配置直接报错；测试 fallback 需明确关闭 use_mamba。

## 尚未通过的验收

- 真实 Mamba cache 已通过一致性和固定内存测试；尚未接入正式 RoboTwin serving。
- 多 episode 批处理和跨卡训练已通过 smoke；两轮 pilot 均已核实完整 validation5 覆盖 1350 条 episode。
- 预训练视觉、TorchCodec 与真实 backward/保存已通过；从 step2 恢复到 step4，与连续训练的 463 个模型张量最大差异为 `1.4901161193847656e-08`，达到数值一致但不是 bitwise 相同。对应目录为 `stage1-zte-v2-resume-control-20260911c` 和 `stage1-zte-v2-resumed-20260911c`。
- global SupCon、effect-shuffle negatives 与 variance/covariance 已接入；action intervention 和完整 Stage1 对照 probe 尚未完成。effect-shuffle negatives 不能独自证明因果识别。
- 工程候选注入在指定 `pretrained_model-best-v1` 加原生 Transformers5 上已复核：零初始化 Base 差异为 0，首步 projector weight 梯度范数 95.13，冻结参数无梯度，单步 diffusion 推理有限。证据为 `zte-v2-injection-bestv1-native-smoke-20260911.json`。输入为 synthetic，内部 padded 输出 `[1,50,32]`，不等于已经验证 RoboTwin EEF16 serving。该实现将残差加到所有已有 prefix embedding，并未增加新 prompt token，因此不能声称与 BehaviorVLA 的 global-token prepend 等价。它目前只是注入工程对照，不代表已选定最终双通道接入；真实表征 correct/shuffled 的归因尚待验证。
- 没有新的正式训练完成结果或闭环成功率；v18 的 Base 43/80、ZeVA 43/80 仍是最近已完成的对应实验。

仅通过结构测试不能声称表征或闭环已优于旧 ZTE；但按用户最新要求，不再因辅助指标未全部通过而阻止 Stage2。真实权重加载、bank 同源、无泄漏及 H15 状态一致性仍需先检查。
