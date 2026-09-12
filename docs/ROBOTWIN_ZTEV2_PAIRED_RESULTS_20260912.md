# ZeVA ZTE v2：RoboTwin 十任务配对评测

## ZTE v2 十任务完整配对结果（2026-09-12）

本节是最新结果；此前“当前 v8/v9”等段落保留为历史记录，不代表本轮方法。

| 条件 | 成功次数 | 成功率 |
|---|---:|---:|
| 普通训练 Base | 109/200 | 54.5% |
| ZeVA（ZTE v2） | 110/200 | 55.0% |
| untouched best-v1 Anchor | 101/200 | 50.5% |

ZeVA 比 Base 多成功 **1 次（+0.5 个百分点）**。这不构成稳定优势证据：
paired bootstrap 95% CI 为 **[-7.5,+8.5] pp**，exact McNemar **p=1.0**，
Base-only/ZeVA-only 成功为 **35/36**。自动验收结果为 **accepted=false**：
Base 高于同样本 Anchor，但低于预设的 57% 历史参考线；门槛没有事后放宽。

### 十任务明细

各任务每个条件均为20 episodes。

| 任务 | Anchor | 训练 Base | ZeVA | ZeVA−Base |
|---|---:|---:|---:|---:|
| beat_block_hammer | 19/20（95%） | 17/20（85%） | 17/20（85%） | 0 pp |
| blocks_ranking_rgb | 9/20（45%） | 9/20（45%） | 13/20（65%） | +20 pp |
| blocks_ranking_size | 4/20（20%） | 3/20（15%） | 7/20（35%） | +20 pp |
| handover_block | 9/20（45%） | 10/20（50%） | 9/20（45%） | -5 pp |
| hanging_mug | 4/20（20%） | 4/20（20%） | 3/20（15%） | -5 pp |
| pick_dual_bottles | 18/20（90%） | 20/20（100%） | 18/20（90%） | -10 pp |
| put_bottles_dustbin | 4/20（20%） | 10/20（50%） | 9/20（45%） | -5 pp |
| scan_object | 8/20（40%） | 11/20（55%） | 9/20（45%） | -10 pp |
| stack_bowls_three | 11/20（55%） | 10/20（50%） | 9/20（45%） | -5 pp |
| stamp_seal | 15/20（75%） | 15/20（75%） | 16/20（80%） | +5 pp |
| **合计** | **101/200（50.5%）** | **109/200（54.5%）** | **110/200（55.0%）** | **+0.5 pp** |

### 方法、选步与协议

两支从指定 best-v1 出发，均训练5000 global steps，global batch256；
冻结视觉语言backbone，Base只训action expert，ZeVA另训context/prior双残差及
Gaussian NLL模块，Stage1 ZTE/Mamba、bank、retrieval冻结。
AE LR5e-6，新模块LR5e-5，prior residual dropout0.4。没有使用正式成功率选步：
分别按最低held-out flow选择 **Base004500（0.01872689）** 与
**ZeVA005000（0.01828257）**。

ZeVA最后500步在aigc24从完整4500恢复，model/adapter/optimizer/scheduler保留，
四卡设置不变，数据副本完整哈希一致；RNG未保存且辅助库版本存在差异，因此
明确为non-bit-exact continuation，不伪称精确续训。

正式协议为Hard/Randomized、seen、三路Large_D435640×480、absolute Joint14、
chunk-start-relative EEF16、H50预测/H15执行、RGB CHW[0,1]、无IK guard/
EEF tracking。真实H15 recurrent state与task-language retrieval，无episode oracle。
从seed1000起筛选各任务首20个expert-valid seeds，三条件精确共享seed和指令；
模型RNG20260907在condition内连续消耗，episode边界重置recurrent state。
三条件共 **600个视频**，数量、成功标签、逐episode seed/指令与progress均核验通过，
所有任务正常退出。这里的视频审计指文件/结果一致性，并非逐帧人工语义复核。

历史57%来自同协议的另一个冻结manifest：200个task-seed只重合181个，
其中仅2个指令也相同；历史model在aigc29，本轮model/render均在aigc24。
因此不能将历史57%与当前54.5%的差值直接归因为训练退化；
同样本Base/Anchor/ZeVA比较有效，历史参考门槛仍保留。

### 与文档中 Ours（18-task checkpoint）的有限交集比较

本轮只评10个任务，与旧Ours有记录的18个任务仅重合6个：
beat_block_hammer、blocks_ranking_rgb、handover_block、hanging_mug、
pick_dual_bottles、put_bottles_dustbin。

这6个任务的旧Ours为 **59/120（49.17%）**，本轮ZeVA为
**69/120（57.50%）**，聚合差值 **+8.33 pp**。
旧记录缺少逐episode seed/指令，不能视作严格paired优势，也不能外推到完整18任务。
未重合的任务不按0计入。

### 证据与结论

远程权威结果目录：

`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-ztev2-selected-pair-20260912`

包含三个condition报告、paired_report.json、acceptance.json、冻结seed_manifest及
逐任务progress、status和视频。正式流程于2026-09-12 20:51:36（北京时间）完成。

**本轮训练和评测已完成，但“正常且稳定优于Base”的目标尚未验收通过。**
保留所有正式结果，不回头挑checkpoint/seed，不把这批测试成败用作训练标签；
下一步先结合训练/验证数据与注入代码做有依据的诊断，再确定独立开发实验。
