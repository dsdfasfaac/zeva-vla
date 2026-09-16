# NLL 梯度隔离：短轮失败与独立预算检验

## 2026-09-16 11:15 更新：DataLoader 启动故障与对照完成

aigc29 首次尝试已加载 PI 权重，但长 `TMPDIR=.../a29-preflight-cache/tmp` 导致 multiprocessing resource sharer 创建 Unix socket 时报 `AF_UNIX path too long`，DataLoader 在首批数据前阻塞；没有训练进度或 checkpoint。这是本次启动环境配置错误，不是 ZTE/优化目标的实测失败。已核验控制器1155621及其子进程归属，仅对该训练进程树发送 SIGTERM，确认退出；失败输出目录和日志保留。

改用 `mktemp -d /tmp/zv.XXXXXXXX` 创建的短临时路径。远端实际 `resource_sharer.DupFd` 往返、4-worker Torch DataLoader 256个样本及 GPU 传输全部通过，日志 `short-tmp-dataloader-preflight.log`。模型/compile/cache/checkpoint等长期输出仍在共享盘，不向已满的 `/data1` 写入缓存。新的训练重启状态将依据实际日志记录，不将预检通过写成训练已完成。

11:19以同一个未改动的隔离源码快照重新派发，控制器 **PID1199343**，新根目录 `.../zeva-runs/robotwin-v5-h15-tasklang/nll-routing-full-20260916-a29-socketfix/nll_detached_full`；日志 `train-nll-detached-full-a29-socketfix-launch.log`。仅改变临时目录和输出目录，仍是预声明的1000步fresh初始化，不加载失败尝试的optimizer。[重启记录](results/robotwin-nll-routing-full-20260916/launch-a29-socketfix.json)。

**恢复已验证：日志出现1/1000和2/1000，至少两个optimizer steps完成，越过首批数据及首次compile。** 首步含编译耗时约210秒，不能把此时tqdm的累计ETA当成稳定训练速度。[实际训练manifest](results/robotwin-nll-routing-full-20260916/socketfix-manifest.json)确认global256、4卡×batch16×累积4、1000步/warmup100、AE5e-6/new5e-5、TorchCodec/compile和NLL输入隔离；基础权重和Stage1/bank沿袭未变。未完成终检，未产生新成功率。

Luna完成了后续启动器的fail-fast防护：在大模型SHA核验之前，实际执行临时目录下的文件系统Unix socket及`DupFd`管道往返，失败则明确退出，不静默改写TMPDIR。测试直接提取启动器中的探针，而不是复制实现；短路径通过、长路径拒绝。14项标准库测试在本地和aigc29隔离目录`tmpdir-guard-20260916-Pxrelv`通过。没有覆盖当前运行中的旧源码快照；本轮重启依赖已单独完成的实际DataLoader预检，新防护用于后续启动。

同机只读 coupled ZeVA1000 对训练 Base1000 的诊断已完成全部368批、5874决策，batch16/full/seed1000，complete=true、模型哈希和冻结权重核验通过：

| 前H15 flow error | 实测 |
|---|---:|
| 原coupled ZeVA1000，残差开启 | 0.01018691808 |
| 同一ZeVA权重，残差关闭 | 0.01018862426 |
| 同预算普通训练Base1000 | 0.01009312179 |

残差开启相对自身关闭改善0.01675%，但仍比训练Base差0.92931%。这不是新候选的结果，也不是成功率。相比旧batch8报告，批次形状改变会改变验证随机噪声；不能混用两份报告来声称模型变好或退化。后续新候选须同batch16、样本顺序、seed和同一Base权重重测。[原始报告](results/robotwin-nll-routing-full-20260916/coupled1000-vs-base1000-b16-a29.json)。以下11:15之前的派发状态为历史记录。

## 已完成的 100 步结果

2026-09-15 23:10（北京时间），`nll-routing-20260915/nll_detached/000100` 完成训练和全部 5,874 个 validation5 决策的只读诊断。模型 SHA256 为 `51d430698504d6abad6a1e8fd4cacc10a8b25d781db6f5161838b9747b1aaf9e`。该候选**没有满足预声明的残差收益条件**，不能将其标为通过。

| 前 H15 flow error，越低越好 | 原 coupled 100 步 | NLL-detached 100 步 |
|---|---:|---:|
| 开启 ZeVA 残差 | 0.01027997304 | 0.01027720701 |
| 同一模型关闭残差 | 0.01027658954 | 0.01027547475 |
| 固定起点 Base004500 | 0.01025113650 | 0.01025113650 |

新路由相对 coupled 的误差下降约 0.02691%，但残差开启仍比自身关闭高约 0.01686%，比固定起点 Base 高约 0.25432%。这些是很小的点估计差异，不是显著性或成功率结论。两份诊断采用 batch16、seed1000、相同顺序的 5,874 个决策；样本顺序 SHA 为 `a9c6a8e30aa3f7ecbfcb9ce81d6a141ef563300d41e339a22d06fb75a0418ac1`。原 coupled 的显式批次上限导致 complete 字段为 false 的历史问题已有独立全覆盖审计，不能改写原始报告；新报告使用无上限 API，complete=true。

原始证据：[新路由末步诊断](results/robotwin-nll-routing-20260915/nll_detached/validation_diagnostics.json)、[coupled 末步诊断](results/robotwin-gate-mechanism-20260915/gate001/validation_diagnostics.json)、[机制及梯度证据](ZTEV2_OBJECTIVE_GRADIENTS_20260915.md)。

## 单独声明的 1,000 步预算

100 步只覆盖约 0.23 epoch。因此，在保留短轮失败结论的前提下，另行预声明一次固定 1,000 步（约 2.26 epochs）的检验，与已有同预算普通 Base1000 和 coupled ZeVA1000 比较。不是从短轮末步恢复，不沿用其 optimizer，也不依据正式测试标签挑步数或 checkpoint。

- 同一普通 Base004500 初始化，新建零残差 adapter 和 optimizer；固定评估末步 001000。
- warmup100、每250步保存，batch16 × 4 GPU × 累积4，global256。
- 冻结 Stage1 ZTE/Mamba、bank、检索和 VLM；AE LR5e-6，新模块 LR5e-5。
- gate0.01、Gaussian NLL weight0.01、prior residual dropout0.4、memory dropout0.1；只隔离 NLL 对共享 task/context 输入的梯度，flow 两条残差路径保持连通。
- TorchCodec、torch.compile、H50输出/H15执行、语言检索、实际递归观测、相机与动作协议均不变。
- 预先指定普通 Base 对照：`fixed-anchor-pair-20260914/baseline/001000`，模型 SHA `bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17`。
- coupled ZeVA 对照：同根目录 `zeva/001000`，模型 SHA `f5e4812a01e01da9936ee23a8f2c9e7d5e70037112f821e759a329d37bc4d11a`。

合同：[robotwin_ztev2_nll_routing_full_20260915.json](../configs/robotwin_ztev2_nll_routing_full_20260915.json)。若固定末步 H15 不能同时优于自身 residual-off 和同预算训练 Base，则不将该改动送入正式评测，也不继续延长该路由的训练预算。离线改善仍不等于闭环成功率改善。

## 运行迁移与实际状态

2026-09-16：aigc29 八卡被其他任务占用，未终止或抢占它们。aigc24 的既有数据副本重新完成全内容哈希，source、EEF index、Joint14 index、stats 四组件与原 aigc29 冻结报告完全相同，路径归一后的 adapter 内容也一致。实际 adapter SHA 为 `864e7e3a3cea2966c4e4e2d984d033ac87008017d11a7fa17fd2d6cd0d459eb1`；绝对路径不同，不伪装成原 adapter 哈希，也不声称跨主机 bit-exact。

隔离源码与证明目录：`/mnt/100T/users/dingxin/VLA/nll-routing-full-20260915-YLzLnA`。证据包括 `dataset-identity-a24.json`、`launch-preflight.json` 和 `runtime-a24-uuid.log`。Torch2.7.1+cu126 四卡 CUDA 计算、逐卡 UUID 和 TorchCodec 实际视频 `[3,480,640]` 解码检查均通过。

发现故障 GPU 导致 CUDA ordinal 与 nvidia-smi index 不一致后，启动器改为先将选择的物理卡解析成 UUID，再做空闲检查和 CUDA 选择，并拒绝重复 UUID。12 项标准库合同/回归测试在本地及远端通过；这不是完整模型 pytest 套件。实际选用 nvidia-smi 物理卡 1、2、3、5；不使用有其他任务的卡0或故障卡4。

首次控制器 PID **3664248** 在 aigc24 派发，日志 `train-nll-detached-full-launch.log`；输出根目录 `.../zeva-runs/robotwin-v5-h15-tasklang/nll-routing-full-20260915/nll_detached_full`。10:38 多个 rank 在 `PI05Policy.__init__ → self.model.to(config.device)` 出现 CUDA launch timeout，进程已退出；未产生训练 manifest 或 checkpoint，也未确认有效 optimizer step。小矩阵预检通过并不能保证完整模型可运行。不把该启动故障解释为方法无效；原目录和日志保留，没有 GPU reset 或其他任务终止操作。

对照诊断的首次 GPU6 小计算预检也遇 CUDA timeout，未启动诊断；GPU7 预检成功，但独立诊断环境漏设 `LIBRARY_PATH`，Triton 编译报 `cannot find -lcuda`，仍未产生诊断结果。训练启动器已有该 linker 路径设置，这不是训练 CUDA timeout 的已证根因；两类故障分别记录。

aigc29 随后已空闲，重派之前在物理卡0–3逐卡验证 UUID、67,108,864 元素的 CPU float32→GPU BF16 传输、有限值与矩阵计算，并验证真实视频解码。新训练控制器 **PID1155621**，仍用同隔离源码，日志 `train-nll-detached-full-a29-launch.log`，输出根目录改为 `.../zeva-runs/robotwin-v5-h15-tasklang/nll-routing-full-20260916-a29/nll_detached_full`。使用原 aigc29 数据与原 adapter SHA `8ac54abcec7704b0111b7c28be3fb3a18e27e0ebe8e3dcf8ddff948b36100f8f`，重新 fresh 初始化；配置和预算不变，不称为 exact resume。此记录写入时尚未确认 optimizer 进度。[重派证据](results/robotwin-nll-routing-full-20260916/launch-a29-recovery.json)。

同机物理卡4已派发只读 coupled ZeVA1000 对普通训练 Base1000 的 batch16/full/seed1000 诊断，**PID1155676**，日志 `coupled1000-vs-base1000-b16-a29.log`，输出同名前缀 JSON。该进程单独补齐 `LIBRARY_PATH=/usr/local/cuda/lib64/stubs`，不与训练共享 GPU；尚无终检结果。正式成功率仍是已完成轮次的 **Base55%、ZeVA55%**，不是该新候选的结果。

对照诊断也需要统一 batch16/full/seed1000：旧 full 对照报告的 batch8 不能直接与新候选混比。最终还需将候选与训练 Base1000 用同一验证批次和噪声比较；启动器默认附带的固定起点 Base004500 诊断不能代替此项。
