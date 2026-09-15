# ZTE v2：初始注入强度机制对照

状态：2026-09-15 19:19北京时间，两臂均完成100步及全5874决策验证。gate010控制器已退出并记录COMPLETE。**增大初始gate的短程收益假设未获支持，不据此加长训练或进入正式闭环。** 上一轮完整闭环仍为Base 55%、ZeVA 55%，没有新的成功率。

## 完整双臂结果

| 同样本、同噪声验证指标 | gate=0.01 | gate=0.10 |
|---|---:|---:|
| H15 residual-on flow error | 0.010279973 | 0.010290603 |
| H15 current residual-off | 0.010276590 | 0.010286857 |
| H15 固定Base004500 | 0.010251136 | 0.010251136 |
| on相对off误差变化 | +0.03292% | +0.03641% |
| on相对固定Base误差变化 | +0.28130% | +0.38500% |
| H50 residual-on | 0.019746169 | 0.019749021 |
| H15 context残差相对范数 | 0.000433618 | 0.003303655 |
| H15 prior残差相对范数 | 0.000004811 | 0.000040662 |

gate0.10的context/prior残差范数分别为0.01臂的7.62/8.45倍，但H15 error比0.01臂高0.10341%。两臂on均略差于各自off和固定Base。该结果只说明在**这组固定100步设置**下没有观察到放大初始化的收益；小差异没有统计显著性证明，更不能据此否定ZTE编码器或推断闭环成功率。

两臂训练manifest除gate初值、输出目录及同内容task文件的隔离路径外一致；trainer/policy源码SHA、foundation、ZTE、bank、live queries、retrieval同源。验证ordered sample SHA、seed、batch16、adapter SHA、368批和5874决策均一致，固定teacher H15值也完全相同。0.01臂的显式cap标记例外见下节独立覆盖审计；0.10臂`protocol.complete=true`。原始报告：[0.01](results/robotwin-gate-mechanism-20260915/gate001/validation_diagnostics.json)、[0.10](results/robotwin-gate-mechanism-20260915/gate010/validation_diagnostics.json)。本实验没有新训练的匹配普通Base，不能用固定004500替代最终公平对照。

## 下一步边界

停止扩大gate或据此直接加步数。限定检查是：在固定训练样本上，分别测量flow和加权Gaussian NLL对新memory/task/prior模块的梯度范数及方向夹角，检查辅助目标与动作目标的局部梯度关系；loss数值大小本身不能证明梯度主导。

**2026-09-15梯度检查已启动，尚无结果。** 同一seed20260915从train集合确定32条decision，4批×8，两臂000100分别加载；保留phase noise/memory/prior dropout，比较同一次forward图上的H50 flow与0.01×Gaussian NLL梯度。覆盖task projector、memory encoder、action prior、两种残差投影及router；未使用正式成功标签。仅为局部梯度诊断，不模拟global256的optimizer更新，不包含preserve hinge、gate regularizer、clipping或Adam预条件，不能单凭梯度比值宣称某项损失控制了实际更新。

执行目录`/mnt/100T/users/dingxin/VLA/objective-gradient-20260915-ovDR64`，aigc29 GPU0/1；两臂监督进程PID387689/387694，日志gate001.log/gate010.log，成功后各写一个新JSON。未创建optimizer、未写checkpoint，检查前后新模块参数逐张量不变；autograd.grad使用eager且关闭gradient checkpointing以兼容多目标求导，冻结VLM/ZTE和训练模式保持。4项CPU/stdlib统计测试在本地和远程通过，实际模型梯度尚待验证。没有启动新策略训练、Stage1重训或已取消的扩展frozen-probe研究。

## 完整性标记拦截及恢复

首臂训练100/100正常结束，`000100`权重、adapter及training_state已保存。只读验证遍历368/368批次，但控制器285790随后退出：诊断CLI的`complete`字段只在`--eval-batches 0`时为true；原launcher给了1000000的显式上限，因此即使遍历全数据仍标false。这不是loss发散、权重损坏或实际只测了部分样本。

已独立核对四项H15/H50 on/off统计及两个paired统计均有5874个有效样本，H15共88110步、H50共293700步，且没有optimizer或checkpoint写入。[原始报告](results/robotwin-gate-mechanism-20260915/gate001/validation_diagnostics.json)的`complete=false`保持不变，另附[实际覆盖审计](results/robotwin-gate-mechanism-20260915/gate001/coverage-audit.json)，不伪造原控制器COMPLETE标记。该臂H15 on=0.010279973、off=0.010276590、固定Base=0.010251136；等待第二臂后才比较初始化假设，不把单臂点估计当方法效果结论。

修正仅将**后续只读诊断**调用设为`--eval-batches 0`，训练budget/数据/模型/噪声设置不变。未重训首臂、未修改原始指标。旧隔离源码保留；新目录`/mnt/100T/users/dingxin/VLA/gate-mechanism-recovery-20260915-KU3sjc`，trainer与policy字节哈希和旧目录一致，9项本地及远程测试通过。只启动尚未开始的gate010，控制器PID321328，日志`train-launch-gate010.log`，沿用相同输出根目录与四卡；这是恢复实验编排，不是optimizer resume。两臂结束后仍需核对ordered sample SHA、batch/noise、lineage及配对诊断，不能只依赖总COMPLETE文件。

真实权重预检在 aigc29 GPU0 完成，PID256838已退出。隔离源码/日志目录 `/mnt/100T/users/dingxin/VLA/gate-mechanism-20260915-L9BZ2g`。两种初始gate分别通过实际Base004500/ZTE加载、零残差输出等价与H15递归；0.10另通过compiled teacher独立性。两个子进程均exit0，[完整终态](results/robotwin-gate-mechanism-20260915/completion.json)和两份模型报告已归档。测试输入是synthetic zero images，不能替代闭环。

训练控制器 PID285790，在同一aigc29上按gate001→gate010依次运行，每臂4卡0/1/2/3、global256，末步再做单卡全验证诊断。日志为上述隔离目录的 `train-launch.log`；模型输出根目录 `/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/gate-mechanism-20260915`。已有产物一律拒绝覆盖，不将目录创建或模型加载视为optimizer已更新。

工程校验：远程7项静态合同测试通过；本地增加实际执行的诊断终态反例测试后共8项通过（拒绝不完整验证、样本数/H15协议错误和写入optimizer的诊断）。真实既有完整诊断报告也通过终态校验。Luna在收尾时触及使用额度，主agent完成最终审查、完整性检查和启动；没有声称Luna完成了远程训练验证。未改生产trainer/policy或历史实验。

## 待检验问题

当前双残差的 flow 路径没有意外 detach。两个投影层零初始化，因此第一步上游 context/prior 没有经投影传来的 flow 梯度；投影更新后该路径才打开。这是安全初始化的预期行为，不是已证实的实现缺陷。context 从第一步仍能通过 Gaussian NLL 学习。

只检验一个假设：初始 gate 从 0.01 改为 0.10，是否让新分支在固定短预算内产生更有用的动作条件信息。Adam 会部分抵消恒定梯度缩放，不能把 gate 十倍等同于参数有效更新十倍；残差范数增大也不等于预测更好。旧 checkpoint 的推理时 prior 放大没有解决收益问题，本实验改变的是训练初始化，而不是再做测试集倍率搜索。

## 固定设计

- 两臂为 `gate001`、`gate010`，均从同一 Base004500 加载 PI 权重，以它作为固定 teacher；新建零投影、adapter 和 optimizer，不读取旧 ZeVA adapter。
- 每臂固定 100 optimizer steps、warmup 10、末步保存；同 seed=1000、训练数据、批量与其余设置。采用相同随机数初始化和数据顺序机制，不承诺跨进程/设备 bit-exact。
- 每卡 batch16 × 4卡 × 累积4 = global256；AE LR5e-6、新模块5e-5；冻结 VLM、ZTE、bank/retrieval。Gaussian NLL、prior dropout0.4、memory dropout0.1、双残差、H50训练/输出及H15执行不变。
- 两臂固定末步在全部 validation5（预期5874决策）上比较相同样本/噪声的 H15/H50 flow、各自 residual-off、固定 Base004500，以及残差相对范数。正式闭环的成功标签不进入训练或选参。
- 若范数增加但 H15 预测没有改善，不支持“放大初始 gate 可以解决收益”的假设；单次小差异只作开发线索，不能宣布统计显著。
- 这是两支 ZeVA 的短机制实验，**没有同时训练新的匹配预算普通 Base**，所以不能作为最终 Base/ZeVA 公平成功率交付。若继续完整训练，仍须给普通 Base 匹配训练预算，并独立确认闭环收益。

## 执行前检查

aigc29 的原始 adapter 路径存在；八卡仅观察到 Xorg 小额显存占用，无计算进程。只读 runtime probe 已实际完成一次训练源视频解码：`[3,480,640]`、uint8，CUDA 数值有限，subprocess exit0；Torch2.7.1+cu126、Transformers5.5.4、Mamba2.2.6.post3、TorchCodec0.5+cu128。该单视频检查不是全数据完整性证明，也不是模型 smoke。

`/data1` 只余约2.2GB，输出和编译缓存写到共享 `/mnt/100T`，不清理他人文件。实际模型加载、zero-init等价、H15状态更新及compiled teacher检查已完成；启动脚本再次核对GPU、端口和权重/产物SHA。使用新隔离源码目录，不覆盖历史产物或生产runtime。
