# BehaviorVLA-aligned ZeVA + effect：实现与预声明

用户2026-09-18指定改法；不再继续旧输出残差候选。方法见[纵览](ZEVA_ROBOTWIN_METHOD_CN.md)。

## 入口

- `src/openpi/zeva/behavior_effect.py`：CTE、五损失、APN、global/effect prefix和prior suffix。
- `src/openpi/zeva/behavior_effect_policy.py`：新schema/SHA、语言检索、新memory、训练和在线接口。
- `scripts/train_robotwin_behavior_effect_cte.py`：train95完整序列、80epochs/batch8、验证和optimizer保存。
- `scripts/export_robotwin_behavior_effect.py`：重建train-only memory及真实H15递归cache。
- `scripts/train_robotwin_behavior_effect_policy.py`：Stage2 full PI＋PBD、global256/5000steps、完整model/adapter/optimizer/RNG；尚未启动，须等固定epoch80 gate及完整产物检查通过。
- `scripts/test_robotwin_behavior_effect.py`：真实CUDA/Mamba测试。
- `scripts/smoke_robotwin_behavior_effect_pi.py`：真实PI/真实数据单卡推理反传及双卡AdamW/梯度累积测试；不保存可提升的训练权重。
- `scripts/smoke_robotwin_behavior_effect_pipeline.py`：早期CTE、十任务train-only微型fixture memory、真实PI的两次H15边界端到端检查；不会保存或提升正式产物。
- `scripts/select_robotwin_behavior_effect.py`：固定step5000的validation5-only gate，拒绝覆盖缺失、重复sample、非同任务置换、正式标签、不同Base权重或checkpoint哈希不符。
- `scripts/enumerate_robotwin_behavior_effect_validation.py`：在Stage2前直接从原始adapter metadata及已冻结Stage1 split另行枚举validation5全部决策ID，无CTE/cache/PI输入；按SHA冻结清单。
- `scripts/report_robotwin_behavior_effect.py`：完整validation5采样动作误差生成器；需显式`--expected-decisions`读取上述已冻结清单、再独立枚举并验证相同，然后核验cache完整性；显式复用同一H50×32噪声、10步去噪，计算Base/aligned/同任务完整错位置换/effect-off四条件H15误差。
- `scripts/run_robotwin_behavior_effect.sh`：aigc28指定GPU UUID/空闲检查；Stage1与smoke模式，另有不触碰GPU的`enumerate-validation`模式。
- `scripts/audit_robotwin_behavior_effect_artifacts.py`：Stage1 epoch80通过后，以原始adapter独立重算train/validation episode与H15决策完整集合，核对导出cache每帧、train-only bank每条task归属、全部有限值、训练源码/数据/CTE/PI/统计与来源SHA。通过时写带输入SHA的审计报告。
- `scripts/run_robotwin_behavior_effect_stage2.sh`：固定的`export → preflight → stage2`入口；检查epoch80完成、源权重与审计SHA、aigc28指定GPU UUID及空闲、全新输出目录。正式Stage2参数是八卡、每卡8、累积4、global256/5000steps。当前仅备妥入口，未启动。

快照 `/mnt/100T/users/dingxin/VLA/zeva-behavior-effect-20260918`；输出 `/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-behavior-effect-20260918`。不抢占其他任务，不复用非空目录。

**正式Stage1已运行**：aigc28 GPU0，UUID `GPU-c4d9aceb-c274-fb29-e145-f6c2ef560bdc`，PID2894806。train5230/validation270 episodes，batch8、80epochs，预计每epoch654个optimizer updates（末批6条，完整无丢弃）。已核实step20：loss18.75965、clip前grad_norm26.75277，无非有限值；这不是validation通过或成功率。日志`.../robotwin-behavior-effect-20260918/stage1.log`，checkpoint目录`.../stage1/`。

## 在结果出现前固定的 gate

1. Stage1固定epoch80。validation5 action MSE比零动作、JEPA比视觉persistence、effect MSE比零effect均至少改善5%，所有值finite；逐步/整段一致与未来信息屏蔽测试通过。5%为本轮预声明工程门槛，不是论文保证。
2. Stage2固定5000末步。相同validation5状态/噪声下，H15 action MSE比正常Base改善≥3%、≥8/10任务非劣；测同任务错位phase/effect、去effect token，检查是否使用对齐信息。门槛不成立不进正式闭环。
3. 先新disjoint10×8 expert-valid开发pair，ZeVA至少多4次成功，才进入冻结原10×20一次正式pair。原样本有历史曝光，必须披露。
4. 最终≥+8/200及完整协议/视频/结果审计才算交付；不把offline proxy当成功率，失败结果完整保留。

在任何新Stage2验证结果出现前，selector将定性消融门槛落实为：aligned MSE严格低于完整同任务置换，且不高于effect-off。比较Base固定为此前正常训练Base1000，模型SHA `bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17`，不能换成较弱权重。指标必须是采样动作H15 MSE，不能用flow loss或NLL代替；独立expected-decision清单要求完整coverage。

开发闭环在任何新ZeVA评测前将seed筛选规则固定为：从absolute seed `1000000`开始，逐任务按递增顺序取最早8个expert-valid seed，并冻结对应seen instruction；须与正式10×20清单（SHA `1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b`）完全不交叉。Base使用同一个正常训练Base1000，ZeVA仅在相同seed/instruction上配对。配置见`configs/robotwin_behavior_effect_20260918.json`。当前尚未筛选或运行这80个episode；aigc29/32/15正被其他作业占满，aigc31有现存显存占用，不抢占。

## 已测/未测

aigc28真实H100/Mamba测试PASS：整段/逐步phase+effect一致、SOS/reset、未来图像/当前目标动作非泄漏、五损失mask与梯度、global/effect prefix mask、Gaussian H50 suffix、0.5推理倍率。

首次默认TF32检查最大差约0.000715，未达2e-4阈值；关闭TF32后原阈值通过，未放宽阈值。真实数据smoke首次在optimizer前因缺预训练ResNet缓存且DNS不可达失败。官方ResNet18已下载，SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`，不退回随机初始化。

真实train95数据2-step smoke已exit0：loss34.60279→32.51453，梯度范数85.8328/94.8002，实际执行backward/optimizer/EMA；这仅证明能训练，不是验证收益。测试产物不能作为正式checkpoint。

正式启动前全量dataset guard拒绝a28的部分视频副本：全50任务source仅22.7GB，对照源79.6GB。已独立对a28/a29的**固定十任务×Clean/Randomized全部20目录**逐文件内容哈希，两侧完全一致；EEF/Joint/stats也与冻结报告完全一致。其他40任务视频不在本轮训练范围，不需要伪称完整50-task副本；新task-scoped audit严格覆盖本轮所有训练/验证输入。原失败log保留为`preflight-full-dataset-guard.log`，双侧报告见本轮results目录。

评测入口已按`zeva_method=behavior_effect`接线，保留相同RoboTwin控制器/H15执行/视频协议。完整闭环验收仍须执行，不能把核心测试写成全pipeline完成。

2026-09-18后续集成：a28空闲GPU1的真实best-v1/真实图像样本测试exit0。PBD未激活时与Base动作逐位一致，激活时输出仍[1,50,16]、finite；真实full-PI backward成功，global/effect/APN均有非零梯度，peak19.43GiB。随后独立GPU1/2双卡DDP测试exit0：每卡batch1、累积2、**测试global4**，2个真实AdamW更新，effect权重变化且两rank逐位一致，peak46.91GiB。它不声称formal global256吞吐或长期稳定性通过，不保存可选checkpoint。原始报告见`docs/results/robotwin-behavior-effect-20260918/pi-runtime-smoke.json`和`pi-ddp-runtime-smoke.json`。

单元selector三组测试通过；新memory加载器拒绝非finite/错维/缺任务产物，并将50-task语言分类器候选限制为固定十任务memory范围，不用每个样本的真实task标签。Stage1进程未重启、参数未更改。后续新增采样报告生成器及三组独立枚举/置换/cache覆盖测试，合计六组纯CPU单元测试通过。报告生成器拒绝不完整checkpoint、非step5000/global256、不同Base SHA、非train-only bank、缺失/多余cache；先写决策和置换计划，再采样，部分结果单独保留，不能提升。CTE历史采用专家H15动作的离线递归cache，明确不是闭环成功率。

进一步容量检查已在a28 GPU1/2 exit0：每卡batch8、累积16、两rank，**实际global256、2次AdamW更新**，peak47.00GiB，总耗时128.18秒，effect权重更新且rank一致。测试重复真实样本，验证计算/显存容量，不证明完整dataloader吞吐、训练收益或正式八rank拓扑；未保存checkpoint，Stage1 GPU0未重启。报告`pi-global256-capacity.json`。

GPU3完整policy接口smoke已exit0：使用真实epoch5 CTE（SHA `49524b87f046c77e6ca6406fca18129f0ee49f353333d6d175e3dc864d4f47db`）、每任务一个train episode形成的十任务微型fixture memory、冻结语言分类器和真实best-v1 PI。在独立validation episode的frame0/15，在线CTE与导出式递归特征最大误差≤1.08e-6，在线/缓存条件下H50动作逐位一致，episode reset通过；显式相同噪声的effect-off采样在改变全局RNG后仍逐位一致。耗时122.76秒，peak10.18GiB，报告`cte-pipeline-smoke.json`明确`promotable=false`，没有保存memory/模型，也没有绕过正式epoch80 gate。此测试不代表全部5230条memory产物或训练后的策略验收。

**仍未验证**：报告生成器真实完整权重/数据运行；epoch80正式全量memory/cache产物；正式八rank拓扑；Stage2精确断点恢复。不能把单元测试、容量测试或早期fixture集成测试当成这些验收通过。Stage1尚须固定epoch80通过gate，才能导出正式新memory并开始Stage2；当前继续训练，无新成功率。报告命令从项目根目录运行，提供`--dataset-root --cte-checkpoint --artifacts --retrieval-checkpoint --foundation-checkpoint --checkpoint --output-dir --expected-decisions`；随后将`report.json`和`expected-decisions.json`交给独立selector，禁止用中间`rows.partial.jsonl`作选择。

独立validation5 ID枚举现已在aigc28完成：固定split270个episode，H15边界共5874个决策，清单SHA `169145aa28b70bd14396904fd0a15e02a216e46ef4d07337fb17ee84b65b1466`，来源adapter/任务配置/Stage1 manifest SHA均记录在`validation5-expected-audit.json`，完整清单`validation5-expected-decisions.json`。第一次尝试导入旧训练包装器时因CPU不可初始化Triton失败，没有生成清单；改为仅导入原始adapter metadata后重新执行exit0，未使用GPU或任何模型输出。正式Stage2入口必须见到该冻结清单及同源adapter哈希；Stage2之后报告生成器再次独立重算并逐项核对。报告命令还必须传`--expected-decisions`，selector检查清单SHA。该清单是离线验证数据索引，与正式闭环seed/成功标签无关。

Stage2预备入口已部署到aigc28。Stage1继续训练，入口的`preflight`早期拒绝检查可确认不会提前提升；实际epoch80产物尚不存在，所以完整审计结果也尚不存在。best-v1模型SHA已在远程重新核实为`7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe`。正式训练前入口要求八卡全部安全空闲；只有所有rank先写完RNG，主rank才写完整模型、adapter、optimizer并标记`COMPLETE`。validation selector还要求全部rank RNG齐全。

首次epoch5验证：action相对零动作改善93.63%，effect相对零effect改善30.11%，vision相对persistence改善3.43%，finite。当前vision未到预声明5%，但本轮只在固定epoch80决定通过与否，不提前提升、不放宽门槛，训练正常继续。

epoch10同一validation5：action 0.014655 对零动作0.312295（改善95.31%）；effect 0.005010 对零effect0.007447（改善32.73%）；direct vision 0.007167 对persistence0.007447（改善3.76%）。三者finite，vision仍未过本轮5%门槛。因为`target_next = target_current + target_effect`，effect head的低误差表明潜在表示已包含部分转移信息；direct vision head却没有同等提升。这提示绝对未来视觉预测头的参数化或优化可能是瓶颈，但两头分别使用`z`与视觉流`v`，不能据此断定原因。现有Stage1/门槛保持原样；若固定epoch80最终失败，再基于验证集做有边界的新方法实验，而非改写本轮判定。

epoch15同一validation5：action 0.012982 对零动作0.312295（改善95.84%）；effect 0.003938 对零effect0.006032（改善34.71%）；direct vision 0.005597 对persistence0.006032（改善7.21%）。三项均finite，**这一中途验证点**达到预声明阈值，但不提前提升checkpoint；正式判定仍仅在固定epoch80进行。训练进程继续运行。

上一正式Base111/20055.5%、旧ZeVA106/20053%、−2.5pp；旧输出残差500步验证改善0.00537%、preserve0改善0.2676%，均失败，不是本次新方法结果。
