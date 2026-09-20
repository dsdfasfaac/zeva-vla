# ZeVA PIM 原冻结 10×20 multi-attempt 正式结果（2026-09-20）

## 结果解释边界

本次三路评测是在 disjoint 10×8 开发 gate 失败后，由用户明确授权运行，必须永久标记为 `user-authorized-after-development-gate-failure`。原正式集合此前已经暴露；本次未用正式标签选择 checkpoint、任务、seed 或 gate，也未重跑或覆盖任何结果。

三路使用同一原冻结 manifest，SHA256 为 `1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b`。每路 10 任务×20 episode、seen instruction、H50 输出/H15 执行、连续模型随机流、最多 4 attempts、成功即停。Base 和 parent 失败后做 episode reset；PIM 失败后做 attempt reset，并提交前一次 BIT 到 PIM。

这里的 Base 仍是正常训练的 Base1000，没有换权重。历史 `111/200=55.5%` 是单 attempt 正式结果；本次重新执行的 Base attempt-1 是 `112/200=56.0%`，累计 4 attempts 后才是 `170/200=85.0%`。二者相差 1 次来自重新执行时的随机扩散轨迹，不应把 `170/200` 写成 Base 的单次成功率。

## 完整审计结果

独立报告：`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-pim-20260919/eval/formal-multiattempt-three-way-original10x20-user-authorized-20260920.json`，SHA256 `d56a61892f2a08e296be84fae13d1947b356aab9754cfac36a3070806da55d04`。

| 条件 | attempt-1 | attempt-2累计 | attempt-3累计 | attempt-4累计 | 实际视频 |
|---|---:|---:|---:|---:|---:|
| 正常训练 Base1000 | 112/200 (56.0%) | 141/200 (70.5%) | 165/200 (82.5%) | 170/200 (85.0%) | 382 |
| CTE+BIT+EAP parent | 126/200 (63.0%) | 153/200 (76.5%) | 166/200 (83.0%) | 176/200 (88.0%) | 355 |
| CTE+BIT+PIM+EAP | 116/200 (58.0%) | 153/200 (76.5%) | 166/200 (83.0%) | 178/200 (89.0%) | 365 |

PIM 相比 Base 最终多 `8/200=+4.0pp`，恰好达到项目目标；配对分解为 PIM-only 成功 19、Base-only 成功 11、共同成功 159、共同失败 11。PIM 相比 parent 只多 `2/200=+1.0pp`，且 attempt-1 少 10 次；因此正式结果支持“完整 CTE+BIT+PIM+EAP 相比正常训练 Base 在最多4次 attempt 的新协议上达到 +4pp”，但不支持“PIM 模块本身已稳定优于 parent”或“单 attempt 提升”。

逐任务 attempt-4 累计成功数：

| 任务 | Base | parent | PIM |
|---|---:|---:|---:|
| beat_block_hammer | 20 | 20 | 20 |
| blocks_ranking_rgb | 18 | 18 | 20 |
| blocks_ranking_size | 14 | 16 | 12 |
| handover_block | 16 | 15 | 20 |
| hanging_mug | 10 | 15 | 14 |
| pick_dual_bottles | 20 | 19 | 20 |
| put_bottles_dustbin | 14 | 17 | 16 |
| scan_object | 18 | 17 | 16 |
| stack_bowls_three | 20 | 19 | 20 |
| stamp_seal | 20 | 20 | 20 |

审计通过：600/600 episode 完整；三路 seed/instruction 完全一致；1,102 个实际 attempt 均有视频；逐 attempt 停止条件和 reset scope 正确；manifest SHA 与冻结值一致。

## 必须保留的历史披露

- 旧正式失败：Base `111/200=55.5%`，旧 ZeVA `106/200=53.0%`。
- 无 PIM 单 attempt 正式结果：Base `111/200=55.5%`，CTE+BIT+EAP `122/200=61.0%`。
- 本次 PIM 开发 gate 失败：PIM 相比 parent 最终仅 `+2/80`，且 first-attempt `-1/80`。
- 原 10×20 正式集合有历史曝光；本次正式运行由用户在开发 gate 失败后明确授权。
