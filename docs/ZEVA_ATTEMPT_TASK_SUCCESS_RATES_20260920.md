# ZeVA 各模型、各 Attempt、各任务成功率汇总

更新时间：2026-09-20

## 1. 口径与可比性

本文汇总四个条件：

1. **Base**：正常训练的 Base policy。
2. **CTE+BIT+EAP**：不含 PIM 的 Parent。
3. **CTE+BIT+PIM+EAP**：旧的跨-attempt PIM；PIM 在失败后的下一次 attempt 使用之前 attempt 的 BIT。
4. **CTE+BIT+Episode-PIM+EAP**：当前 Episode-PIM；BIT 是当前 H15 边界的短期状态，PIM 只聚合同一 episode 中严格更早的 BIT，不跨 attempt、也不跨 episode。

前三个条件来自原冻结 `10任务×20`、最多4次attempt的正式评测。该正式评测是在旧跨-attempt PIM 的 disjoint 开发 gate 失败后由用户明确授权继续，不能用于回头选择 checkpoint、任务、seed 或 gate。冻结 seed manifest SHA256 为：

`1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b`

Episode-PIM 来自新的、与既有集合均不相交的 `10任务×8` 单次开发评测。冻结 seed manifest SHA256 为：

`3160b039b7c571850ef8187f61818c0aaeaa0a8ebc88c8c189b1cd8fcdc86103`

多-attempt 表中的成功率采用用户指定的**独立 attempt 成功率**：

`该 attempt 成功数 / 实际进入该 attempt 的 episode 数`

因此 Attempt 2–4 的分母会逐步缩小，且对应前面 attempt 未成功的 survivor cohort；不同 attempt 的百分比不能当成同一批样本上的直接纵向比较。`— (0/0)` 表示没有 episode 进入该 attempt。本文不把“最多4次内累计解决率”写成某个 attempt 的成功率。

所有表格单元均同时给出百分比和原始计数，格式为 `成功率（成功数/进入数）`。

## 2. 原冻结 10×20 多-attempt 正式评测

### 2.1 Base

| 任务 | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 |
|---|---:|---:|---:|---:|
| **总体** | 56.0% (112/200) | 33.0% (29/88) | 40.7% (24/59) | 14.3% (5/35) |
| `beat_block_hammer` | 85.0% (17/20) | 100.0% (3/3) | — (0/0) | — (0/0) |
| `blocks_ranking_rgb` | 40.0% (8/20) | 50.0% (6/12) | 50.0% (3/6) | 33.3% (1/3) |
| `blocks_ranking_size` | 40.0% (8/20) | 25.0% (3/12) | 11.1% (1/9) | 25.0% (2/8) |
| `handover_block` | 30.0% (6/20) | 28.6% (4/14) | 60.0% (6/10) | 0.0% (0/4) |
| `hanging_mug` | 20.0% (4/20) | 6.2% (1/16) | 26.7% (4/15) | 9.1% (1/11) |
| `pick_dual_bottles` | 100.0% (20/20) | — (0/0) | — (0/0) | — (0/0) |
| `put_bottles_dustbin` | 35.0% (7/20) | 30.8% (4/13) | 22.2% (2/9) | 14.3% (1/7) |
| `scan_object` | 60.0% (12/20) | 37.5% (3/8) | 60.0% (3/5) | 0.0% (0/2) |
| `stack_bowls_three` | 75.0% (15/20) | 60.0% (3/5) | 100.0% (2/2) | — (0/0) |
| `stamp_seal` | 75.0% (15/20) | 40.0% (2/5) | 100.0% (3/3) | — (0/0) |

### 2.2 CTE+BIT+EAP

| 任务 | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 |
|---|---:|---:|---:|---:|
| **总体** | 63.0% (126/200) | 36.5% (27/74) | 27.7% (13/47) | 29.4% (10/34) |
| `beat_block_hammer` | 95.0% (19/20) | 100.0% (1/1) | — (0/0) | — (0/0) |
| `blocks_ranking_rgb` | 65.0% (13/20) | 42.9% (3/7) | 25.0% (1/4) | 33.3% (1/3) |
| `blocks_ranking_size` | 40.0% (8/20) | 33.3% (4/12) | 25.0% (2/8) | 33.3% (2/6) |
| `handover_block` | 55.0% (11/20) | 22.2% (2/9) | 28.6% (2/7) | 0.0% (0/5) |
| `hanging_mug` | 40.0% (8/20) | 33.3% (4/12) | 0.0% (0/8) | 37.5% (3/8) |
| `pick_dual_bottles` | 85.0% (17/20) | 33.3% (1/3) | 50.0% (1/2) | 0.0% (0/1) |
| `put_bottles_dustbin` | 65.0% (13/20) | 28.6% (2/7) | 20.0% (1/5) | 25.0% (1/4) |
| `scan_object` | 45.0% (9/20) | 27.3% (3/11) | 25.0% (2/8) | 50.0% (3/6) |
| `stack_bowls_three` | 70.0% (14/20) | 66.7% (4/6) | 50.0% (1/2) | 0.0% (0/1) |
| `stamp_seal` | 70.0% (14/20) | 50.0% (3/6) | 100.0% (3/3) | — (0/0) |

### 2.3 CTE+BIT+PIM+EAP（旧跨-attempt PIM）

| 任务 | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 |
|---|---:|---:|---:|---:|
| **总体** | 58.0% (116/200) | 44.0% (37/84) | 27.7% (13/47) | 35.3% (12/34) |
| `beat_block_hammer` | 90.0% (18/20) | 100.0% (2/2) | — (0/0) | — (0/0) |
| `blocks_ranking_rgb` | 70.0% (14/20) | 66.7% (4/6) | 0.0% (0/2) | 100.0% (2/2) |
| `blocks_ranking_size` | 30.0% (6/20) | 35.7% (5/14) | 0.0% (0/9) | 11.1% (1/9) |
| `handover_block` | 40.0% (8/20) | 41.7% (5/12) | 85.7% (6/7) | 100.0% (1/1) |
| `hanging_mug` | 15.0% (3/20) | 29.4% (5/17) | 33.3% (4/12) | 25.0% (2/8) |
| `pick_dual_bottles` | 95.0% (19/20) | 100.0% (1/1) | — (0/0) | — (0/0) |
| `put_bottles_dustbin` | 55.0% (11/20) | 22.2% (2/9) | 0.0% (0/7) | 42.9% (3/7) |
| `scan_object` | 35.0% (7/20) | 53.8% (7/13) | 0.0% (0/6) | 33.3% (2/6) |
| `stack_bowls_three` | 65.0% (13/20) | 42.9% (3/7) | 75.0% (3/4) | 100.0% (1/1) |
| `stamp_seal` | 85.0% (17/20) | 100.0% (3/3) | — (0/0) | — (0/0) |

### 2.4 多-attempt 总体速览

| 条件 | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 |
|---|---:|---:|---:|---:|
| Base | 56.0% (112/200) | 33.0% (29/88) | 40.7% (24/59) | 14.3% (5/35) |
| CTE+BIT+EAP | **63.0% (126/200)** | 36.5% (27/74) | **27.7% (13/47)** | 29.4% (10/34) |
| CTE+BIT+PIM+EAP | 58.0% (116/200) | **44.0% (37/84)** | **27.7% (13/47)** | **35.3% (12/34)** |

这里的粗体仅标记同一 attempt 列中的最高观测值；Attempt 2–4 的进入样本不同，不能据此单独断言模型优劣。旧跨-attempt PIM 的 Attempt 1 为 `116/200=58.0%`，低于不含PIM的 Parent `126/200=63.0%`。

## 3. 当前 Episode-PIM：新 disjoint 10×8 单次开发评测

本节三路都只执行一次，并使用完全相同的80个 seed/instruction，因此可以直接做 paired 比较。Episode-PIM 在同一 episode 内读取严格更早的 BIT；episode 起点 PIM 为空。

| 任务 | Base | CTE+BIT+EAP | CTE+BIT+Episode-PIM+EAP |
|---|---:|---:|---:|
| **总体** | 50.0% (40/80) | 52.5% (42/80) | **61.25% (49/80)** |
| `beat_block_hammer` | 87.5% (7/8) | 87.5% (7/8) | 75.0% (6/8) |
| `blocks_ranking_rgb` | 62.5% (5/8) | 62.5% (5/8) | 62.5% (5/8) |
| `blocks_ranking_size` | 25.0% (2/8) | 25.0% (2/8) | 50.0% (4/8) |
| `handover_block` | 50.0% (4/8) | 37.5% (3/8) | 50.0% (4/8) |
| `hanging_mug` | 12.5% (1/8) | 12.5% (1/8) | 37.5% (3/8) |
| `pick_dual_bottles` | 100.0% (8/8) | 100.0% (8/8) | 100.0% (8/8) |
| `put_bottles_dustbin` | 0.0% (0/8) | 12.5% (1/8) | 25.0% (2/8) |
| `scan_object` | 37.5% (3/8) | 37.5% (3/8) | 50.0% (4/8) |
| `stack_bowls_three` | 62.5% (5/8) | 75.0% (6/8) | 75.0% (6/8) |
| `stamp_seal` | 62.5% (5/8) | 75.0% (6/8) | 87.5% (7/8) |

Episode-PIM 相对同集合 Base 为 `+9/80=+11.25pp`，相对同集合 CTE+BIT+EAP Parent 为 `+7/80=+8.75pp`，通过预声明的“严格高于 Base 和 Parent”开发 gate。paired discordance 为：

- Episode-PIM 对 Base：PIM-only成功19条、Base-only成功10条，净增9条。
- Episode-PIM 对 Parent：PIM-only成功19条、Parent-only成功12条，净增7条。

该结果是 **post-formal 开发证据**，不能直接与上面的原冻结10×20正式数值做统计合并，也不能改写此前正式结论。

## 4. 证据来源

- 原冻结10×20多-attempt三方独立报告：`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-pim-20260919/eval/formal-multiattempt-three-way-original10x20-user-authorized-20260920.json`
  - SHA256：`c805b7bd88c46a8722bab12a278ee7b21b2ae8aa8a636fe4da5da4dd05250e11`
- Episode-PIM新disjoint 10×8单次三方独立报告：`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-episode-pim-20260920/eval/development-singleattempt-three-way-seed3000000.json`
  - SHA256：`17fdf8d5156c5b97673152d5f5fb6924fb7a147145e8f7c5df60112359a1a86e`
- Episode-PIM方法与训练/验证说明：`docs/ZEVA_EPISODE_PIM_20260920.md`
- 旧跨-attempt PIM正式结果说明：`docs/ROBOTWIN_PIM_FORMAL_MULTIATTEMPT_RESULTS_20260920.md`

原冻结正式历史必须同时披露：更早的正式评测为 Base `111/200`、旧ZeVA `106/200`（失败）；无PIM单次正式评测为 Base `111/200`、CTE+BIT+EAP `122/200`。这些历史结果与本文不同轮次的评测协议/运行不可混成同一 paired 样本。
