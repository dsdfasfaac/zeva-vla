# ZeVA v2：保护 Base action expert 的梯度分路实验

状态（2026-09-16 20:48 北京时间）：新候选在 aigc29 GPU0–3 正常完成 1000/1000 optimizer steps，`001000/model.safetensors`、`zeva_adapter.pth`、`training_state.pt` 及完整验证报告均已保存，训练控制器写入 `COMPLETE` 后退出。运行目录为 `/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/gradient-route-full-20260916-a29/gradient_route_full`，独立源码快照为 `/mnt/100T/users/dingxin/VLA/gradient-route-test-20260916-PZL6Hh`。**当前路由已未通过预注册的第一道离线门槛，不进入正式闭环评测。** 对同预算训练 Base1000 的严格同噪声诊断正在进行，用于归因基础路径是否被保护，而不是重新决定失败门槛；尚无新成功率。

全 5874 条 validation5 决策、batch16、seed1000、样本顺序 SHA `a9c6a8e30aa3f7ecbfcb9ce81d6a141ef563300d41e339a22d06fb75a0418ac1` 的前 H15 flow error：residual-on `0.010095753706991673`，自身 residual-off `0.010094467550516129`。开启残差比关闭高 **0.012741%**，不满足“必须优于自身 off”。当前诊断中固定训练起点 Base004500 为 `0.010251136496663094`，但起点不是同预算训练 Base，不能代替第二道门槛。[原始全量报告](results/robotwin-gradient-route-20260916/gradient-route1000-vs-base4500-b16-a29.json)。

## 可证伪假设

旧 NLL-detached1000 的同预算全验证：ZeVA residual-on H15 误差 `0.01016413979`，自身 off 为 `0.01016624738`，普通训练 Base1000 为 `0.01009312179`。候选基础路径比 Base 差 `0.72451%`，而残差只带来 `0.02073%` 的内部改善。[旧轮证据](ZTEV2_NLL_ROUTING_BUDGET_20260916.md)。仅靠参数“少动”不成立：候选 AE 更新 RMS 反而比 Base 小 `13.24%`，但更新向量 cosine 为 `0.53855`。因此本轮直接隔离优化目标，而不再扫 gate 或重跑失败的 NLL-only 路线。

对同一个预处理 batch、同一 PI flow 噪声与时间：

- `residual-on` 的 flow、Gaussian NLL、preserve 与 gate 正则只更新 ZeVA 新模块，action-expert 参数梯度通过 hook 置零；NLL 对共享 task/context 输入保持 stop-gradient。
- `residual-off` 的当前 student 基础 flow **只更新 action expert**。这不是固定 teacher loss；固定 Base004500 仍只用于原有每 4 步一次的 preserve hinge。
- on 的 forward/backward 完成后再构造 off；on 始终处于 DDP `no_sync`，仅最后一个累积 microstep 的 off backward 同步。off 图对 ZeVA 参数有零值链接，以满足 `find_unused_parameters=False` 而不改变其梯度。
- AE 与 ZeVA 两组梯度各自按全局范数 1.0 裁剪；AE 不被新增模块的梯度范数缩放。默认未开启该选项时保留原先的联合裁剪和训练语义。

ZTE/Mamba、train95 bank、任务检索和视觉语言 backbone 冻结；task-language 检索与真实 H15 recurrent phase、action-expert 侧双残差和 Gaussian prior、H50 输出/H15 执行保持不变。训练从普通 Base004500 新建零残差 adapter 与 optimizer，AE LR `5e-6`，ZeVA LR `5e-5`，4 卡×batch16×累积4=`global256`。不恢复旧失败候选。完整固定预算、对照 SHA 和离线停止条件见[预注册合同](../configs/robotwin_ztev2_gradient_route_full_20260916.json)。

## 启动前真实检查

1. 两卡 NCCL 小模型：连续两轮、每轮两步梯度累积，跨卡平均 AE/ZeVA 梯度均为预期的 `2.0`；第一次 DDP 双 forward 的不同步问题已修复。此项只验证 reducer 协议。
2. 真实 PI0.5 + Stage1 + 一条 train 样本：Base-only 与 routed off loss 均为 `0.0017408238491043448`；覆盖五类 action-path 参数的 8 个代表张量、共 `6,881,280` 个梯度元素，最大绝对差为 **0**。全部 208 个 AE 参数触发 on/off hook；on 梯度被清零，ZeVA 38 个参数中 16 个有非零梯度。无 optimizer/权重写入。[原始 JSON](results/robotwin-gradient-route-20260916/real-gradient-smoke-a29.json)。这不是所有 4.3 亿 AE 元素逐值比对，也不是性能证明。
3. 四卡真实数据/`torch.compile` 单步：`1/1` optimizer step 正常完成，用实际 global256，启用独立组裁剪且 `save_checkpoints=false`；没有生成模型 checkpoint。[单步 manifest](results/robotwin-gradient-route-20260916/one-step-manifest.json)。

正式运行 manifest 固定 `steps=1000`、global256、分路与裁剪协议，trainer SHA 为 `71a3ea2c9be0e98ce23fa3874d2105d491fd4c224fcf31f42fb3da752370937c`。[运行 manifest](results/robotwin-gradient-route-20260916/full-run-manifest.json)、[启动 manifest](results/robotwin-gradient-route-20260916/full-launcher-manifest.json)。

## 停止与评测条件

固定末步 001000 在**同一 5874 条 validation5 决策、batch16、seed1000、相同顺序/噪声**下须同时优于自身 residual-off 和已经训练好的普通 Base1000（模型 SHA `bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17`），否则拒绝此路由，不根据测试成功标签挑 checkpoint、gate、任务或 seed。离线通过仍不等于闭环成功率改善；只有通过后才按既定十任务×20 expert-valid seed 的相同评测协议做正常 Base/ZeVA 配对闭环。最新**已完成**的正式成功率仍为 Base `110/200=55%`、ZeVA `110/200=55%`，交付条件未满足。
