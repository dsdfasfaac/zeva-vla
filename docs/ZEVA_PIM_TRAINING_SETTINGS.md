# ZeVA PIM training settings

代码保留两条互不兼容的 PIM 训练线。统一入口是：

```bash
scripts/run_robotwin_pim_training.sh --describe
scripts/run_robotwin_pim_training.sh cross-attempt
scripts/run_robotwin_pim_training.sh within-episode
```

机器、GPU UUID、输入产物、输出目录和“不得覆盖已有 run”的检查仍由各自底层启动脚本执行。统一入口不会绕过这些保护。

## Setting A：`cross-attempt`

- BIT：当前 attempt 内的短期交互轨迹。
- PIM：同一 episode 内已经完成的 attempt 所提交的 BIT 轨迹。
- 读取：attempt 1 的 PIM 为空；后续 attempt 读取更早 attempt 的记忆。
- 写入：仅在 attempt reset 时提交完整 BIT；episode reset 清空全部记忆。
- 离线训练历史：train-only、同任务、不同 episode 的 label-free 配对，用来模拟跨 attempt 历史；前500步只训练 PIM，之后开放 PI/EAP/PIM。
- 固定训练：global256、2000步、PIM LR `5e-5`；后1500步 PI LR `1e-6`、EAP LR `1e-5`。
- 入口：`scripts/run_robotwin_cross_attempt_pim_stage2.sh`。
- 状态：作为历史消融保留。disjoint multi-attempt 开发 gate 未通过，不能表述为已验证提升。

## Setting B：`within-episode`（当前）

- BIT：当前 H15 边界的短期交互状态。
- PIM：同一 episode 内严格早于当前边界的 BIT 因果历史。
- 读取/写入顺序：先读取 `PIM[0:t)`，再预测动作，最后写入当前 BIT，避免当前目标泄漏。
- reset：只在 episode 边界清空；不存在跨 attempt 或跨 episode 记忆。
- 训练历史：train-only，同一 episode 的严格因果前缀 `[0,t)`，不使用成功标签。
- 冻结：CTE、BIT、task memory、语言检索、PI、EAP；只训练 PIM 模块。
- 固定训练：global256、2000步、PIM LR `5e-5`、容量64。
- 入口：`scripts/run_robotwin_episode_pim_stage2.sh`。
- 评测：单次闭环评测不等于“单-attempt PIM”；这里 PIM 仍然是 episode 内跨 H15 边界的长期记忆。
- 状态：新的 disjoint 10×8 单次开发集上，Base `40/80`、Parent `42/80`、Episode-PIM `49/80`，通过固定开发 gate；它仍是 post-formal 开发证据。

## 代码映射

| setting | 训练器 | policy class | 配置 |
|---|---|---|---|
| `cross-attempt` | `scripts/train_robotwin_cross_attempt_pim.py` | `ZevaCrossAttemptPIMPolicy`（兼容名 `ZevaPIMPolicy`） | `configs/robotwin_cte_eap_pim_20260919.json` |
| `within-episode` | `scripts/train_robotwin_episode_pim.py` | `ZevaEpisodePIMPolicy` | `configs/robotwin_episode_pim_20260920.json` |

机器可读总表位于 `configs/robotwin_pim_training_settings.json`。跨attempt配对产物使用入口 `scripts/build_robotwin_cross_attempt_pim_artifacts.py`。旧文件名和旧 schema 仅为 checkpoint/load 兼容保留；新代码和文档使用 ZeVA 的 CTE/BIT/PIM/EAP 命名。
