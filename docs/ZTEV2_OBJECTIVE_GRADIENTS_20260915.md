# ZTE v2：动作目标与 Gaussian NLL 的梯度关系

最新状态（2026-09-16）：下文的100步候选已完成，但没有证明残差收益；H15开启残差比关闭差0.01686%。独立预声明的1000步预算检验在aigc24因模型加载CUDA timeout失败，现以同配置重派到新空闲的aigc29，尚未确认optimizer进度；不是恢复短轮或将失败改称通过。[末步结果、固定预算与迁移证据](ZTEV2_NLL_ROUTING_BUDGET_20260916.md)。下文启动段落保留为历史记录。

2026-09-15：两臂000100的只读检查均完成，监督日志均为`AUDIT_EXIT=0`，进程已退出。使用相同seed20260915、32条固定train decision（4批×8），sample IDs及其顺序完全一致。两个报告均确认被审计的新模块参数前后未改变；没有optimizer或checkpoint写入。

## 实测

下表为4个批次中“0.01×NLL梯度范数 / flow梯度范数”的范围。

| 模块 | gate0.01 | gate0.10 |
|---|---:|---:|
| memory context encoder | 413–1,014 | 50–124 |
| task token projector | 608–1,195 | 71–146 |
| Gaussian prior head | 296,692–457,685 | 27,994–48,713 |

memory encoder两臂各有3/4批次梯度内积为负。gate0.01的cosine为−0.1999、−0.1126、+0.0507、−0.0390；gate0.10为−0.2161、−0.1463、+0.1344、−0.1783。残差投影层只收到flow梯度，NLL梯度为0，符合现有图结构。

这支持一个**局部待验证机制**：共享task/context表征收到的辅助NLL梯度远大于动作flow梯度，且部分样本方向冲突。不能把32条样本外推成全数据结论，也不能把原始梯度范数比当成Adam参数更新比例。本次没有测量preserve hinge、gate regularizer、clipping和Adam预条件的完整合成影响；它不是闭环掉点的因果证明。

原始报告：[gate0.01](results/robotwin-objective-gradient-20260915/gate001.json)、[gate0.10](results/robotwin-objective-gradient-20260915/gate010.json)。远程目录`/mnt/100T/users/dingxin/VLA/objective-gradient-20260915-ovDR64`。

## 针对性改动：仅隔离辅助NLL的共享输入梯度

新增默认关闭的`prior_nll_detach_context`训练选项，仅用于标准`zeva`路径：

- flow仍使用完全连通的task/context→Gaussian prior mean→action residual路径，context residual也保持连通。
- NLL另用相同prior head计算一次，输入task/phase/context使用stop-gradient；因此NLL继续训练Gaussian head，但不更新共享task/context模块。
- prior head无dropout，额外计算不采样新随机数；该改动旨在只改变辅助梯度路由，不改变固定权重下的flow或Gaussian分布数值。
- 默认false保留旧训练行为；VLM/ZTE/bank冻结、LR5e-6/5e-5、global256、双残差、NLL权重0.01、dropout0.4、H50/H15均不改。

## 路由验证及100步对照启动

隔离目录`/mnt/100T/users/dingxin/VLA/nll-routing-20260915-zwppCZ`中的真实权重检查已完成，`AUDIT_EXIT=0`。相同gate001/000100模型与adapter、相同32条训练样本：四批flow/NLL数值完全相同，各模块flow梯度范数相同；task/context的NLL梯度范数为0，prior head的NLL梯度范数相同。新模块参数前后不变。[原始诊断](results/robotwin-nll-routing-20260915/detached-gate001.json)、[对照核验](results/robotwin-nll-routing-20260915/routing-validation.json)。这里只记录并比较了梯度范数，不声称保存并逐元素比较了完整梯度向量。

4项直接Torch检查通过（形状、标准Gaussian NLL、新路由数值/梯度及真实policy.forward入口的轻量foundation替身测试）；它们不是完整pytest套件。该runtime没有pytest，首次隐藏全部GPU又触发了Mamba/Triton的driver初始化错误；保留失败日志，启用空闲GPU可见后4项直接检查通过。随后上述实际9GB模型诊断也完成。启动脚本10项本地和远程合同/完整性测试通过。

**已启动新路由的固定100步训练候选，尚未确认optimizer进度，也没有新成功率。** 控制器PID523813，aigc29 GPU0/1/2/3，日志`train-nll-detached-launch.log`。输出根目录`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/nll-routing-20260915/nll_detached`。预先声明的[合同](../configs/robotwin_ztev2_nll_routing_20260915.json)保持gate0.01、warmup10、global256、LR及所有物理协议不变；只启用`prior_nll_detach_context`。

从Base004500新建adapter和optimizer，不从gate001/000100续训；复用已经完成的coupled gate001/000100作为同预算开发对照。默认关闭的新选项保留旧forward；真实路由比较已证实固定权重下数值不变。末步做全5874决策同batch16/seed1000的H15/H50诊断，比较coupled control、各自off与固定Base，不能按正式成功率选步或调gate。这仍不是新训练的匹配普通Base及闭环交付，是否采用该路由还要看实测收益。
