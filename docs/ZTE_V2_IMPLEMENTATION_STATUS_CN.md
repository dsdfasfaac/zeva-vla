# ZTE v2 实现验收记录

日期：2026-09-11。本文记录已执行的验证；设计文档中的目标不代表已经实现或通过。

## 当前运行

`stage1-zte-v2-pilot-20260911d` 与同预算 `stage1-zte-v2-pilot-taskpaired-20260911e` 均已完成 256 steps，分别约 9 分 57 秒和 9 分 4 秒（包含两次完整 validation5）。均使用 aigc29 四张 H100（0/1/3/4），每卡 batch 8，global batch 32，warmup 32；在 step128/256 保存并跑完整 validation5。唯一训练方法变化是同任务跨 episode 配对采样。最新一轮 launcher PID `477001` 已退出，不能把仍存在的 pid 文件当成运行状态。两轮均未通过 Stage1。

该预算是学习曲线与完整验证的 pilot，最多访问 8192 个 episode 样本，远未等同于 40/80 epochs。其职责是证明训练目标可学、观察 held-out 泛化与塌缩情况；不得因此自动放行 Stage2。Stage1 ZTE 使用原有 50-task 表征数据，后续 Base/ZeVA 的训练和正式比较仍在用户选定的 10 个任务上。

## 已验证的信息路径

在 aigc29 H100、PyTorch 2.7.1+cu126、真实 Mamba CUDA 路径执行：

```bash
PYTHONPATH=/data1/dingxin/zeva-runtime-deps:src CUDA_VISIBLE_DEVICES=0 \
ZEVA_TEST_CUDA=1 python3 -m unittest openpi.zeva.transition_encoder_v2_test -v
```

结果：8 tests，全部通过（1.859 秒）。七项使用小型视觉 fixture 隔离时序机制；一项使用真实 ResNet-18 验证训练模式下 BatchNorm 与梯度，不声称测试了预训练权重的任务效果。另有 3 项 representation objectives 测试通过，验证匹配/置换 effect、增强正样本与防塌缩损失的方向。

1. 置换 after-image 不改变 forward-effect/next-action 预测，改变 post-transition causal signal。
2. 修改未来的 before/after/action 不改变过去的 phase、causal signal 或预测。
3. 颠倒 H15 动作顺序改变 pre-context。
4. 右侧 padding 不改变有效 phase 或 masked global pool。
5. 逐步回放与整段编码一致。
6. effect 预测对动作有梯度，对 after-image 没有梯度。
7. 真实视觉网络训练时固定 BatchNorm 统计，未来图像不污染当前预测，视觉参数仍有梯度。
8. 多个 episode 放在同一 batch 时，动作递归不会跨越 batch 维度；修改第二个 episode 不改变第一个。

缓存版本另外验证了 phase、causal、forward-effect、next-action、progress 和 global token 与整段编码一致（容差 `1e-5`），且三个 Mamba cache 的 tensor 数量不随 transition 数增长。8 项测试全部通过；训练用的整段 forward 数学定义保持不变。

## 首轮完整验证与采样归因

| checkpoint | 有效 validation episodes | task probe | phase order | effect cosine |
|---|---:|---:|---:|---:|
| 首轮 step128 | 1350 | 7.037% | 53.026% | 0.00877 |
| 首轮 step256 | 1350 | 7.481% | 53.099% | 0.00838 |
| task-paired step256 | 1350 | 4.148% | 52.635% | 0.00850 |

两次均未通过 Stage1。首轮仅约 0.31 epoch，不能据此否定表示结构；同时，完整采样审计定位到一个与目标函数直接冲突的问题：

| 完整 train95 sampler（26150 episodes） | 同卡内跨 episode 同任务正样本覆盖 | 不同任务负样本覆盖 | 每条 episode 恰好一次 |
|---|---:|---:|---|
| interleave + rank stride | 3.4646% | 99.9924% | 是 |
| same-task pairs | 99.8088% | 100% | 是 |

剩余约 0.19% 为各任务奇数条数的尾项，保留完整覆盖，不重复采样。复现审计：`bash scripts/run_robotwin_zte_v2.sh --run scripts/audit_robotwin_zte_sampler.py`。修正针对的是 global SupCon 缺少跨轨迹正样本的机制问题，不以闭环成功率搜索采样规则。

独立诊断使用相同 seed1000、每任务 2 条 train 原型轨迹及 2 条不重叠 validation 轨迹（共 100+100 条），没有用 validation 构建 bank：

| step256 对照 | masked-language 原型检索 | next-H15 MSE 相对任务均值改善 | effect MSE 相对零效果改善 |
|---|---:|---:|---:|
| 原采样 d | 30/100 | +2.21% | -4.59% |
| task-paired e | 39/100 | +1.76% | -6.49% |

原型检索与上表的分类头 task probe 是不同指标。检索点估计改善尚无 episode-group 显著性证据；action/effect 没有同步改善。各模型 EMA target 不同，不能直接用两者 effect 绝对 MSE 比较表征质量。两份 `representation_diagnostic_step256.json` 均为 diagnostic-only，不能放行 Stage2。

### Next-action 到 phase 的梯度审计

Luna worker 在真实 H100、step128 checkpoint 上仅对 next-action loss 反传：`phase_head.grad=None`、`post_fusion.grad=None`，`pre_fusion` 梯度范数约 0.08877。原实现下一动作只从 pre-context 预测，因此该损失没有直接训练实际导出的 phase。这是结构证据，不是已证明的成功率掉点唯一原因。

新增 `action_prediction_context=phase` 对照让下一段动作直接从导出的 normalized phase 预测；当前 after-state 在 H15 执行完毕后已可见，允许用于下一动作，但 forward-effect 仍必须隐藏 after-state。旧 checkpoint 缺省 `pre`；此变化不得隐式套用旧权重。BehaviorVLA 官方实现实际是 `action_predictor(h_a)` 加 shifted actions，不是从 `h_b` 直接预测；本修正是需要独立验证的 ZeVA 设计。

该对照预先固定 paired sampler、seed1000、256 steps、原 LR/权重和全部验证 episode。新增 head 的维度不能扰动其他公共参数的初始化随机序列；需逐张量测试。验收先验证 action loss 到 phase 的梯度、当前/未来信息边界和 Mamba cache 等价，再比较动作增量与其他表示指标，不能因代码测试通过自动宣称能力提升。

实现 commit `a10b6c9`：真实 H100/Mamba 路径 13/13 tests 通过（3.837 秒），包括同 seed 公共权重逐张量一致、新路径 phase/post 梯度、当前 after 可见与未来不可见、cache 等价。旧 task-paired step256 的 463 个 state keys 以 `strict=True` 加载通过，旧 config 缺省为 `pre`。新实验必须从初始化训练，不能把 pre checkpoint 伪装成 phase resume。

`stage1-zte-v2-pilot-phaseaction-20260911f` 已由 launcher `495287` 在 aigc29 启动，配置为上述唯一监督路径变化。日志/checkpoint/独立诊断与旧 e 对照分目录保存；训练完成后必须复核实际进程及 step256，不依赖 pid 文件判断状态。

## 已完成的真实数据 smoke

统一结果根目录：`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/`。

| 实验目录 | 配置 | 已验证结果 |
|---|---|---|
| `stage1-zte-v2-smoke-b2-20260911a` | GPU0，batch 2，2 steps | 真实 TorchCodec 加载、预训练 ResNet、backward、验证、完整 checkpoint 保存完成 |
| `stage1-zte-v2-smoke-b8-20260911b` | GPU0，batch 8，2 steps；新增对比和防塌缩目标 | 完成，无 OOM；验证统计 8 个 episode，诊断 loss 12.7925 |
| `stage1-zte-v2-smoke-ddp2-20260911b` | GPU1/3，每卡 batch 2，2 steps | DDP 完成，验证跨卡汇总 4 个 episode，诊断 loss 6.0612 |

三个实验均保存 model/optimizer/scheduler，均输出 `probe_gate_passed=false`、`probe_gate_status=diagnostic_only`、`validation_complete=false`。这些 smoke 的目标和 batch 数不同，不能比较其总 loss 来判断表征优劣。单卡 batch 8 的两个步骤与一次小验证共 16 秒，仅是短 smoke 测量，不是正式训练耗时预测。

## 验收中修正的问题

- 单 visual-key attention 会让所有 action query 得到相同输出；补回 ordered action-token residual。
- action Mamba 改为按完整 episode 的 H15 子步展开，保留跨 chunk 历史。
- 取消强制单调的 progress 公式，避免顺序指标天然满分。
- next-action auxiliary head 仅预测下一 H15，训练使用时间移位标签；PI 的 H50 契约由独立策略保持。
- 逐步状态保存 raw image，避免再进入 forward 时二次标准化。
- 视觉计算采用 microbatch 与 activation checkpoint，限制全 episode 训练的显存。
- 缺少 Mamba 时 production 配置直接报错；测试 fallback 需明确关闭 use_mamba。

## 尚未通过的验收

- 真实 Mamba cache 已通过一致性和固定内存测试；尚未接入正式 RoboTwin serving。
- 多 episode 批处理和跨卡训练已通过 smoke；两轮 pilot 均已核实完整 validation5 覆盖 1350 条 episode。
- 预训练视觉、TorchCodec 与真实 backward/保存已通过；从 step2 恢复到 step4，与连续训练的 463 个模型张量最大差异为 `1.4901161193847656e-08`，达到数值一致但不是 bitwise 相同。对应目录为 `stage1-zte-v2-resume-control-20260911c` 和 `stage1-zte-v2-resumed-20260911c`。
- global SupCon、effect-shuffle negatives 与 variance/covariance 已接入；action intervention 和完整 Stage1 对照 probe 尚未完成。effect-shuffle negatives 不能独自证明因果识别。
- 工程候选注入在指定 `pretrained_model-best-v1` 加原生 Transformers5 上已复核：零初始化 Base 差异为 0，首步 projector weight 梯度范数 95.13，冻结参数无梯度，单步 diffusion 推理有限。证据为 `zte-v2-injection-bestv1-native-smoke-20260911.json`。输入为 synthetic，内部 padded 输出 `[1,50,32]`，不等于已经验证 RoboTwin EEF16 serving。该实现将残差加到所有已有 prefix embedding，并未增加新 prompt token，因此不能声称与 BehaviorVLA 的 global-token prepend 等价。它目前只是注入工程对照，不代表已选定最终双通道接入；真实表征 correct/shuffled 的归因尚待验证。
- 没有新的正式训练完成结果或闭环成功率；v18 的 Base 43/80、ZeVA 43/80 仍是最近已完成的对应实验。

仅通过本记录中的结构测试，不能开始 Stage2，也不能声称表征已优于旧 ZTE。
