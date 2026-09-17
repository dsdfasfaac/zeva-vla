# H15-route ZeVA：RoboTwin 十任务配对正式结果（2026-09-17）

结论：**没有达到交付目标**。同一批 10 任务×20 expert-valid episodes，正常训练的 Base1000 成功 **111/200（55.5%）**，ZeVA H15-route 第500步成功 **106/200（53.0%）**；ZeVA 低 **2.5 个百分点**、少成功 5 条。目标是 ZeVA 至少高 4 个百分点，即以本轮 Base 为准至少 **119/200**；当前还差 13 条成功。不能把验证集 H15 flow loss 约 0.25% 的改善称作闭环提升。

| 任务 | Base /20 | ZeVA /20 | ZeVA−Base |
|---|---:|---:|---:|
| beat_block_hammer | 18 | 19 | +1 |
| blocks_ranking_rgb | 14 | 10 | −4 |
| blocks_ranking_size | 8 | 4 | −4 |
| handover_block | 6 | 8 | +2 |
| hanging_mug | 4 | 4 | 0 |
| pick_dual_bottles | 20 | 18 | −2 |
| put_bottles_dustbin | 9 | 9 | 0 |
| scan_object | 6 | 5 | −1 |
| stack_bowls_three | 13 | 14 | +1 |
| stamp_seal | 13 | 15 | +2 |
| **合计** | **111/200** | **106/200** | **−5/200** |

配对 discordance 为 Base-only 35、ZeVA-only 30；精确 McNemar `p=0.6201`，配对 bootstrap 95% 区间为 ZeVA−Base **[−10.5,+5.5] 个百分点**。这是单次 200 episode 估计，不证明 ZeVA 在总体上必然更差，也绝不支持已经超过 Base。区间和 p 值来自[正式 paired report](results/robotwin-h15-route-20260916/formal-complete/paired_report.json)。

## 协议与审计

正式目录：`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-h15-route-selected-000500-a29-20260917`。Base 与 ZeVA 在 aigc29 同一 8 卡环境顺序运行；固定 10 任务、每任务 20 条、从 seed 1000 开始筛出的同一批 expert-valid seeds 和实际 seen 指令。冻结 seed/instruction manifest SHA256 `1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b`。Large_D435 640×480 的 head/双 wrist 三相机，absolute Joint14，chunk-start relative EEF16，H50 预测、前 H15 执行后重规划，模型随机种子 20260907 连续流；未启用 IK guard 或 EEF tracking。Base 配置 SHA256 `a0f02f432199ba9d3bb4bdb24e3fdd59272045b108cce204e98e6f302a727248`。

[独立 completion audit](results/robotwin-h15-route-20260916/formal-complete/completion_audit.json)从逐 episode progress 重新计算成功数，核查 Base/ZeVA 的 200 对 seed 与**实际指令**完全相同、各任务 status return code 0、每个 episode 的视频文件成功标签与 progress 一致，两个条件共 400 个视频；结论 `passed=true`、`errors=[]`。此外在远端对全部 400 个 MP4 使用 `ffprobe` 检查视频流容器，未报损坏。审计器原先只支持带 Anchor 的三组评测；现已修正以识别本次预定的两组 Base/ZeVA 协议。此处 `passed` 只表示**数据完整性与配对协议**通过，并非效果达标。

Base 55.5% 略高于上轮 Base1000 的 55%，仍低于历史非同轮的 57% 参考值；历史 57% 不能混作本轮配对 Base，也不能拿来替代原始 111/200。当前 ZeVA 是固定 Base1000、冻结 ZTE/Mamba/causal bank/VLM、仅训练新增模块的 H15-route；第500步由验证集预注册规则选定，没有按这里的测试成功标签选 checkpoint 或 gate。

## 下一轮决策边界

这轮提示「H15 离线 flow 微弱改善」不足以作为闭环 +4pp 的代理指标；直接加大 context gate 的开发集检查反而使 H15 flow 变差。[后续机制诊断](ZTEV2_POST_H15_MECHANISM_20260917.md)已开始用 train95/validation5 检查有效梯度和 ZTE 的条件信息量；四个固定微批次的梯度结果**不支持 NLL 抢占裁剪预算**，全量 ZTE 错位对照仍在运行。旧 v14 **直接 output residual** 已在另一个 10×8 开发 split 得到 Base 43/80、ZeVA 38/80（远端 `eval/advantage10-output-residual-v14/step-001500/split-j/paired_report.json`），不能当作未试过的自然下一步原样重跑。不得依照本表的任务/seed 成败来筛任务、选 checkpoint、定 gate 或宣称新方案成功。若再有候选，须先通过不使用正式测试标签的独立验证门槛，再按相同冻结协议做新的配对闭环，并完整公布失败轮次。
