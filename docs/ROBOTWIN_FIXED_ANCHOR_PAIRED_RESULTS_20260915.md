# 固定 teacher 方案：十任务配对闭环结果

评测于 2026-09-15 06:49:32（北京时间）完成，运行节点 aigc28。

**结论：没有达到交付目标。普通训练 Base 与 ZeVA 均为 110/200（55.0%），ZeVA 没有总体提升；Base 也低于预设 57% 参考线。** untouched best-v1 Anchor 为 104/200（52.0%），不能用它替代普通训练 Base 来宣称 ZeVA 优势。

## 完整十任务结果

每项分母均为 20；差值为 ZeVA − Base 的百分点。

| 任务 | Anchor | 训练 Base | ZeVA | 差值 |
|---|---:|---:|---:|---:|
| beat_block_hammer | 17/20（85%） | 17/20（85%） | 19/20（95%） | +10 |
| blocks_ranking_rgb | 8/20（40%） | 10/20（50%） | 11/20（55%） | +5 |
| blocks_ranking_size | 2/20（10%） | 7/20（35%） | 5/20（25%） | −10 |
| handover_block | 8/20（40%） | 8/20（40%） | 7/20（35%） | −5 |
| hanging_mug | 6/20（30%） | 6/20（30%） | 2/20（10%） | −20 |
| pick_dual_bottles | 18/20（90%） | 20/20（100%） | 20/20（100%） | 0 |
| put_bottles_dustbin | 4/20（20%） | 5/20（25%） | 9/20（45%） | +20 |
| scan_object | 9/20（45%） | 6/20（30%） | 6/20（30%） | 0 |
| stack_bowls_three | 16/20（80%） | 13/20（65%） | 15/20（75%） | +10 |
| stamp_seal | 16/20（80%） | 18/20（90%） | 16/20（80%） | −10 |
| **合计** | **104/200（52.0%）** | **110/200（55.0%）** | **110/200（55.0%）** | **0** |

Base-only 成功 32 对，ZeVA-only 成功 32 对；精确 McNemar p=1.0，episode-paired bootstrap 95% CI 为 [−7.5, +8.0] 个百分点。这里的 bootstrap 对 episode 对进行重采样，不是 task-cluster 置信区间。小样本任务涨跌只能描述本轮观测，不能据此筛任务、选倍率或证明某类任务存在稳定优势。

## 固定模型和协议

- 普通 Base 与 ZeVA 均从训练好的 Base004500 出发，各完成预先指定的 1,000 fresh-optimizer steps；比较固定末步，不按测试结果挑 checkpoint。
- 两者共同来源为指定 `pretrained_model-best-v1`；Anchor 是未经本轮训练的该原始模型。
- Stage2 冻结 VLM、Stage1 ZTE/Mamba 和 causal bank，训练 action expert；ZeVA 另训练 action-expert 侧双残差/Gaussian prior 模块。AE LR=5e-6，新模块 LR=5e-5，global batch=256。
- 三条件使用同一组从 1000 开始筛出的 expert-valid seeds，每任务 20 个，同一实际 seen 指令、8 个逻辑 RNG 流；没有改用 episode_index oracle。
- randomized 场景语义、Large_D435 640×480、head/left/right、RGB CHW [0,1]、absolute Joint14、chunk-start-relative EEF16、H50 输出/H15 执行；不启用 IK guard/EEF tracking。

基础权重 SHA256：Base `bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17`；ZeVA `f5e4812a01e01da9936ee23a8f2c9e7d5e70037112f821e759a329d37bc4d11a`。完整配置与种子来源见 [manifest](results/robotwin-fixed-anchor-20260914/formal-complete/manifest.json)。

## 审计与限制

控制器已完成三条件结构及配对 seed/指令检查并生成 [paired report](results/robotwin-fixed-anchor-20260914/formal-complete/paired_report.json)。[acceptance](results/robotwin-fixed-anchor-20260914/formal-complete/acceptance.json) 为 false：Base ≥ 同条件 Anchor 满足，但 Base ≥ 57% 与 ZeVA > Base 均不满足。历史 57% 并非完全相同的 seed/实际指令集合，这一参考线不能被解释为统计显著的同条件掉点证明。

三条件共 600 个视频均通过独立 ffprobe 容器/视频流/640×480尺寸检查，每任务20个，summary成功数、视频文件命名与固定seed/实际指令逐项一致；[独立数据与视频审计](results/robotwin-fixed-anchor-20260914/formal-complete/paired-data-video-audit.json)的两项汇总检查均为true。不宣称完成逐帧解码或人工内容审查。dustbin 的初始化异常由既有同 frozen-seed 重试逻辑处理；Base seed1024/1035 均保留在最终结果中，没有因初始化异常换 seed。另一个通用审计脚本报告的配置文本检查问题正在独立核实，不能用视频检查通过替代所有配置检查。

## 方法判断与后续边界

这轮闭环结果与训练后 held-out 诊断一致：固定 teacher 的改动没有证明增量收益。此前 5,874 个决策上，ZeVA 的 H15 flow error 比同预算 Base 高约 1.1224%，开启残差也未优于同权重关闭残差。不能把辅助 Stage1 分数达线包装成动作增强成功，也不能单凭这一次平局断言 ZTE encoder 无效。

保留本轮失败结果，不继续原方案盲目加步数、不基于这 200 个测试样本选择新 checkpoint/任务子集。下一步必须在训练/验证数据上提出并检验明确的表征利用或优化机制假设，再决定新的训练；需要新的独立确认评测来支持未来的提升主张。监视任务不因本轮评测结束而标记用户目标完成。GitHub 发布权限仍未恢复，未宣称上传成功。
