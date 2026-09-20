# ZeVA PIM 原冻结 10×20 multi-attempt 正式结果（2026-09-20）

## 结果解释边界

本次三路评测是在 disjoint 10×8 开发 gate 失败后，由用户明确授权运行，必须永久标记为 `user-authorized-after-development-gate-failure`。原正式集合此前已经暴露；本次未用正式标签选择 checkpoint、任务、seed 或 gate，也未重跑或覆盖任何结果。

三路使用同一原冻结 manifest，SHA256 为 `1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b`。每路 10 任务×20 episode、seen instruction、H50 输出/H15 执行、连续模型随机流、最多 4 attempts、成功即停。Base 和 parent 失败后做 episode reset；PIM 失败后做 attempt reset，并提交前一次 BIT 到 PIM。

这里的 Base 仍是正常训练的 Base1000，没有换权重。历史 `111/200=55.5%` 是单 attempt 正式结果；本次重新执行的 Base attempt-1 是 `112/200=56.0%`，累计 4 attempts 后才是 `170/200=85.0%`。二者相差 1 次来自重新执行时的随机扩散轨迹，不应把 `170/200` 写成 Base 的单次成功率。

## 完整审计结果（修订：以每次 attempt 的独立成功率为主）

独立报告：`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-pim-20260919/eval/formal-multiattempt-three-way-original10x20-user-authorized-20260920.json`，SHA256 `c805b7bd88c46a8722bab12a278ee7b21b2ae8aa8a636fe4da5da4dd05250e11`。报告已修订为 v2；此前把 attempt-4 累计成功率当成主要成功率的解释已撤回。

每次 attempt 的独立成功率，以“该 attempt 成功数 / 实际进入该 attempt 的 episode 数”为分母：

| 条件 | attempt-1 | attempt-2 | attempt-3 | attempt-4 |
|---|---:|---:|---:|---:|
| 正常训练 Base1000 | 112/200 (56.00%) | 29/88 (32.95%) | 24/59 (40.68%) | 5/35 (14.29%) |
| CTE+BIT+EAP parent | 126/200 (63.00%) | 27/74 (36.49%) | 13/47 (27.66%) | 10/34 (29.41%) |
| CTE+BIT+PIM+EAP | 116/200 (58.00%) | 37/84 (44.05%) | 13/47 (27.66%) | 12/34 (35.29%) |

PIM 相比 Base 的独立成功率差为 attempt1 `+2.00pp`、attempt2 `+11.09pp`、attempt3 `-13.02pp`、attempt4 `+21.01pp`。PIM 相比 parent 为 attempt1 `-5.00pp`、attempt2 `+7.56pp`、attempt3 `0.00pp`、attempt4 `+5.88pp`。因此结果是混合的：PIM 对失败后的第2和第4次恢复有正信号，但第一次明显退化、第三次无增益；不能宣称每次 attempt 均达到 +4pp，也不能宣称 PIM 稳定优于 parent。

后续 attempt 的分母是前序失败留下的 survivor cohort，三路 cohort 大小并不总相同，条件成功率不是完全同质样本上的直接因果比较。必须同时报告成功数和分母。

仅作辅助诊断的 4-attempt 累计曲线如下，不作为“每次 attempt 成功率”：

| 条件 | attempt-1累计 | attempt-2累计 | attempt-3累计 | attempt-4累计 | 实际视频 |
|---|---:|---:|---:|---:|---:|
| 正常训练 Base1000 | 112/200 | 141/200 | 165/200 | 170/200 | 382 |
| CTE+BIT+EAP parent | 126/200 | 153/200 | 166/200 | 176/200 | 355 |
| CTE+BIT+PIM+EAP | 116/200 | 153/200 | 166/200 | 178/200 | 365 |

累计到4次时 PIM 相比 Base 的确是 `+8/200=+4.0pp`，但这只是“最多4次内最终解决率”，不能用来证明独立 attempt 成功率达到 +4pp。按独立 attempt 口径，项目的统一 `+4pp` 结论 **不成立**；attempt-1 仅为 `+2.0pp`，且四次结果不一致。

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
