# RoboTwin 50-task 与 ZeVA 专项十任务选择

## 结论

ZeVA 的结构性优势不是单帧抓放精度，而是利用冻结的 ZTE/Mamba、causal bank、任务语言检索和真实 H15 recurrent memory，帮助 action expert 判断“当前处于哪个阶段、此前完成了什么、下一步该做什么”。因此最适合 ZeVA 的是长时序、多阶段、双臂交接、重复物体处理和接触后状态变化任务。

本轮固定以下十个任务做专项 Stage2：

1. `beat_block_hammer`
2. `blocks_ranking_rgb`
3. `blocks_ranking_size`
4. `handover_block`
5. `hanging_mug`
6. `pick_dual_bottles`
7. `put_bottles_dustbin`
8. `scan_object`
9. `stack_bowls_three`
10. `stamp_seal`

选择在看到新的 v8 闭环结果之前固定。历史 50-task audit 仅用于确认这些任务仍有 PI baseline headroom，不把同一组评测成功率当成训练标签。后续报告必须写明这是 **task-specialized 10-task model**，不能将它的十任务均值冒充 50-task benchmark 总分。

## 50 个任务的适配性

### A：强适配（18 个）

| 任务 | ZeVA 可能提供的增益 |
| --- | --- |
| `beat_block_hammer` | 抓锤、对齐、击打的阶段切换与接触结果 |
| `blocks_ranking_rgb` | 多物体排序，需要记住已完成的位置和下一目标 |
| `blocks_ranking_size` | 多物体顺序与阶段记忆，单帧容易混淆进度 |
| `dump_bin_bigbin` | 抓取、搬运、翻转倾倒的长动作链 |
| `handover_block` | 双臂交接、释放时机和接触状态 |
| `handover_mic` | 双臂交接与阶段同步 |
| `hanging_mug` | 抓取、姿态调整、挂接及接触确认 |
| `pick_diverse_bottles` | 语言目标与相似/多样瓶体消歧 |
| `pick_dual_bottles` | 双臂、多目标和已抓取状态 |
| `place_cans_plasticbox` | 重复放置时记住已处理的罐体 |
| `place_dual_shoes` | 双臂协调和左右对象/目标绑定 |
| `put_bottles_dustbin` | 多瓶重复处理，强依赖完成历史 |
| `scan_object` | 扫描进度和姿态覆盖具有历史依赖 |
| `stack_blocks_three` | 三步堆叠与当前层级记忆 |
| `stack_blocks_two` | 多阶段堆叠、接触与释放时机 |
| `stack_bowls_three` | 三步嵌套/堆叠及当前高度状态 |
| `stack_bowls_two` | 多阶段堆叠和接触状态 |
| `stamp_seal` | 抓取、对齐、下压和完成状态 |

### B：中等适配（16 个）

这些任务存在阶段或接触状态，但整体更短，视觉—动作 backbone 本身通常已能解决大部分问题：

`adjust_bottle`、`click_alarmclock`、`click_bell`、`lift_pot`、`open_laptop`、`open_microwave`、`place_bread_skillet`、`place_burger_fries`、`place_container_plate`、`place_empty_cup`、`press_stapler`、`put_object_cabinet`、`rotate_qrcode`、`shake_bottle`、`shake_bottle_horizontally`、`turn_switch`。

### C：弱适配（16 个）

这些任务主要是单目标、短时序的拾取—移动—放置；性能更受检测、6D 位姿、抓取和低层控制精度约束，memory 的边际价值较小：

`grab_roller`、`move_can_pot`、`move_pillbottle_pad`、`move_playingcard_away`、`move_stapler_pad`、`place_a2b_left`、`place_a2b_right`、`place_bread_basket`、`place_can_basket`、`place_fan`、`place_mouse_pad`、`place_object_basket`、`place_object_scale`、`place_object_stand`、`place_phone_stand`、`place_shoe`。

## 十任务选择依据

| 任务 | 关键难点 | 选择理由 | 旧 PI baseline / Ours best-v1 |
| --- | --- | --- | --- |
| `beat_block_hammer` | 多阶段、工具、接触 | phase 与 action-effect memory 都直接有用 | 55% / 80% |
| `blocks_ranking_rgb` | 颜色消歧、顺序 | task language + 已完成序列 | 35% / 45% |
| `blocks_ranking_size` | 尺寸消歧、顺序 | 强进度记忆且 baseline headroom 最大 | 10% / 40% |
| `handover_block` | 双臂交接 | 接触与释放阶段难由单帧稳定判断 | 15% / 40% |
| `hanging_mug` | 姿态、挂接 | 长动作链与挂接状态 | 15% / 30% |
| `pick_dual_bottles` | 双臂、多目标 | 任务绑定和双臂进度 | 60% / 100% |
| `put_bottles_dustbin` | 重复多物体 | 必须记住已完成对象 | 20% / 25% |
| `scan_object` | 覆盖进度 | 当前帧不完整表达扫描历史 | 25% / 40% |
| `stack_bowls_three` | 三阶段堆叠 | 当前层级、接触和释放历史 | 55% / 65% |
| `stamp_seal` | 对齐、下压 | 工具接触后的阶段切换 | 45% / 90% |

旧成功率只作为 headroom 的旁证。`Ours best-v1` 是已有 RoboTwin PI checkpoint，不是本轮 ZeVA v8 的闭环结果；真正的“ZeVA 有优势”必须由同 seeds 的 paired PI/ZeVA 闭环评测确认。

## 专项训练协议

- 初始化：原始已训好的 RoboTwin PI0.5，不从 50-task ZeVA Stage2 checkpoint 续训。
- Stage1：继续使用完整 50-task 的 ZTE/Mamba、causal bank、task-language retrieval，并全部冻结。
- Stage2 数据：仅过滤上述十任务；任务 ID 仍保留完整 50-task vocabulary，避免 bank/retrieval 错位。
- 参数：冻结 PaliGemma 视觉—语言 backbone；训练 PI0.5 action expert 与 ZeVA 双残差/action-prior 模块。
- 动作协议：模型 H50 输出、H15 执行/重规划；task-language 检索和真实 recurrent H15 phase 保持不变。
- 优化：每卡 batch 16、累积 2、8 卡，global batch 256；PI action expert LR `5e-6`，ZeVA LR `5e-5`。
- 步数：首轮 1,000 optimizer steps，约等于完整 50-task 5,000 steps 的单任务平均曝光量；100-step warmup，每 250 steps 保存并验证。
- 评测：训练完成后在十任务上使用同一份 expert-valid seed manifest，分别跑原 PI 和 ZeVA；每任务 20 episodes，报告逐任务成功率、macro average 和 paired delta。

机器可读任务清单位于 `configs/robotwin_zeva_advantage10.json`。
