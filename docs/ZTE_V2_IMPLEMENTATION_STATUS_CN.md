# ZTE v2 实现验收记录

日期：2026-09-11。本文记录已执行的验证；设计文档中的目标不代表已经实现或通过。

## 当前运行

已在 aigc29 启动 `stage1-zte-v2-pilot-20260911d`，launcher PID `466477`，四张 H100（0/1/3/4），每卡 batch 8，global batch 32。固定 256 optimizer steps，warmup 32；在 step128/256 保存并跑完整 validation5。运行目录在下述统一结果根目录内，日志为 `train.log`。PID 仅是定位线索，每次监视必须检查实际进程，不能只相信 pid 文件。

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

- 上述逐步路径仍是前缀重算参考实现，尚非固定计算量的 Mamba cache 部署实现。
- 多 episode 批处理和跨卡训练已通过 smoke；整份 validation5 的完整覆盖还需全量运行核实。
- 预训练视觉、TorchCodec 与真实 backward/保存已通过；从 step2 恢复到 step4，与连续训练的 463 个模型张量最大差异为 `1.4901161193847656e-08`，达到数值一致但不是 bitwise 相同。对应目录为 `stage1-zte-v2-resume-control-20260911c` 和 `stage1-zte-v2-resumed-20260911c`。
- global SupCon、effect-shuffle negatives 与 variance/covariance 已接入；action intervention 和完整 Stage1 对照 probe 尚未完成。effect-shuffle negatives 不能独自证明因果识别。
- 真实 PI0.5 的零初始化等价、首次可学习梯度及 correct/shuffled 归因尚待验证。
- 没有新的正式训练完成结果或闭环成功率；v18 的 Base 43/80、ZeVA 43/80 仍是最近已完成的对应实验。

仅通过本记录中的结构测试，不能开始 Stage2，也不能声称表征已优于旧 ZTE。
