# ZeVA Behavior-Effect：RoboTwin 冻结配对结果

本轮达到预声明工程目标：在原冻结10任务×20 paired协议上，正常训练的Base1000为 **111/200（55.5%）**，用户授权的epoch40 CTE/PBD探索分支为 **122/200（61.0%）**，即 **+11次成功、+5.5个百分点**。目标要求至少+8/200（+4pp），正式gate通过。

这仍是一轮200 episode估计：Base-only 33、ZeVA-only 44，精确McNemar `p=0.2543`，paired bootstrap 95% CI为 **[-3.0,+14.0]pp**，区间包含0。因此结果支持“本次冻结工程gate已通过”，但不能写成统计显著或总体必然提升。该正式集合此前已有实验曝光，也必须披露上一轮失败：相同Base为111/200，旧ZeVA为106/200（53.0%，−2.5pp）。本轮没有用正式标签选择checkpoint、任务、seed或gate。

## 逐任务结果

| 任务 | Base | ZeVA | 差值 |
|---|---:|---:|---:|
| beat_block_hammer | 18/20 (90%) | 17/20 (85%) | −5pp |
| blocks_ranking_rgb | 14/20 (70%) | 13/20 (65%) | −5pp |
| blocks_ranking_size | 8/20 (40%) | 8/20 (40%) | 0pp |
| handover_block | 6/20 (30%) | 13/20 (65%) | +35pp |
| hanging_mug | 4/20 (20%) | 7/20 (35%) | +15pp |
| pick_dual_bottles | 20/20 (100%) | 18/20 (90%) | −10pp |
| put_bottles_dustbin | 9/20 (45%) | 8/20 (40%) | −5pp |
| scan_object | 6/20 (30%) | 11/20 (55%) | +25pp |
| stack_bowls_three | 13/20 (65%) | 15/20 (75%) | +10pp |
| stamp_seal | 13/20 (65%) | 12/20 (60%) | −5pp |
| **总计** | **111/200 (55.5%)** | **122/200 (61.0%)** | **+5.5pp** |

## 协议与审计

- frozen manifest SHA256：`1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b`；Base和ZeVA逐episode复用完全相同的seed与seen instruction。
- `zeva_randomized`、Large_D435 640×480三相机、absolute Joint14、chunk-start relative EEF16、PI预测H50、实际执行H15、continuous model RNG。
- Base复用此前完整审计的正常Base1000证据；其源report SHA256为`efe917df7fa4f51bb7d7b02c439c35c84ef7d43c41447e5b49cc374fb2d52506`。
- 独立completion audit重算所有progress、结果文件名、success、seed、instruction、配置与报告，`errors=[]`。
- Base和ZeVA各200个最终视频；400/400经`ffprobe`确认存在H.264视频流、分辨率640×480、大小和时长均大于0。这里不声称完成人工逐帧语义检查。
- 原始远端证据：`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-behavior-effect-20260918/eval/formal-behavior-effect-epoch040-step5000-a29-20260919`。
- 可提交证据：[结果摘要](results/robotwin-behavior-effect-20260918/epoch40-formal-paired-summary.json)、[独立completion audit](results/robotwin-behavior-effect-20260918/epoch40-formal-completion-audit.json)、[400视频ffprobe audit](results/robotwin-behavior-effect-20260918/epoch40-formal-video-ffprobe-audit.json)。

## 结果边界

这一权重来自用户授权的Stage1 epoch40 → 5000-step full PI+PBD探索分支；它在冻结validation5离线gate和disjoint 10×8开发pair通过后，才运行原10×20一次正式pair。固定epoch80 Stage1及其artifact/preflight也已通过，但epoch80 Stage2尚未训练。因此本结果不能被描述为“epoch80正式Stage2结果”，也不取代后续更大样本或新未曝光测试集上的复验。
