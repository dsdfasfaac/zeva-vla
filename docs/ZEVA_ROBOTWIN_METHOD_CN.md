# ZeVA：BehaviorVLA 对齐的 CTE + effect

更新：2026-09-18。用户指定以 BehaviorVLA 为骨架，增加 effect 预测和 effect token。本页取代旧双门控、冻结 PI 输出残差等方法纵览；[历史实验完整保留](ZEVA_ROBOTWIN_METHOD_HISTORY_TO_20260918.md)。

## 方法纵览

CTE 沿用 BehaviorVLA VBE 的三流因果 Mamba：视觉流、**上一段实际执行动作**流、可学习行为 token 流。各流作因果时间建模，再在同一时刻交互。当前隐状态 `z_t ∈ R256` 表示执行阶段，task head 输出128维检索 key。新增 effect head 从 `z_t` 预测下一次 H15 边界的视觉特征差；它不是已被证明的因果识别量。

Stage2 信息进入 PI0.5 的路径：

1. 冻结 task-language 检索器从指令预测任务，在**新 CTE 的 train95 memory**中检索全局行为 `g`，本 episode 保持稳定。
2. `Linear(g)` 和 `Linear(predicted_effect_t)` 两个 token 放在 **VLM prefix 最前方**，与图像、语言同属 prefix attention block。
3. Gaussian APN 将 `g` 展开为8个 waypoint，以当前 `z_t` 作 attention query，预测 H50×16 mean/std。
4. mean 经零初始化线性层，**相加到 action expert 的 noisy-action embedding**，不是最终 EEF 动作。训练每样本40% residual dropout，推理固定乘0.5。
5. PI原始 flow 解码仍输出H50；执行前H15后，用真实已执行动作及新图像更新 CTE。

这条路线不使用旧 context/prior 双门控、BIT/PIM context、输出动作修正、teacher hinge 或 episode oracle。旧 CTE/bank/adapter 不能当作新 checkpoint 加载。

## Stage1

保留官方四个目标：action reconstruction、next-vision JEPA、global task contrastive、local temporal distinctiveness。只增加 effect MSE：

`effect_target_t = stopgrad(EMA_visual(o_{t+1}) − EMA_visual(o_t))`

`L = 0.1 L_action + 0.2 L_vision + 2 L_global + L_phase + 0.2 L_effect`

effect head 只有 `z_t` 输入，不输入当前 expert action 或未来图像。未来图像仅作监督，部署使用同一个预测 effect。

- 三路640×480 RGB分别缩放224×224、映射[-1,1]后融合，保持相机顺序。
- 动作采用 baseline mean/std；上一段有序 H15×EEF16 展平投影，不将未执行H50放入历史。
- action loss 对15个时间位置平均、对16个动作坐标求和；effect权重0.2与JEPA同量级，是本轮预声明而非调测试集得到。
- ImageNet预训练 ResNet18、4层三流Mamba、dim256；80epochs、batch8；vision LR1e-5，其他LR1e-4；AdamW decay1e-4、clip1、EMA0.99。
- 固定十任务train95、完整H15序列、右padding+mask，同任务成对且每epoch无放回完整覆盖。
- BatchNorm固定预训练running statistics，避免跨时间batch statistics泄漏；EMA始终eval。
- 固定epoch80；每5epoch验证，不按正式闭环结果选权重。

## Memory、Stage2、评测

Stage1过验证gate后重新导出 train-only key/value memory、真实逐步 train/validation H15 phase/effect cache。旧的**语言任务分类器**只复用指令→任务分类，不把旧ZTE坐标当新CTE key；预测任务映射到新CTE task key，再在该任务memory中top5 cosine/softmax检索。

Stage2 **全量训练PI0.5＋PBD**，冻结新CTE/memory/语言检索；global256，PI LR5e-6、新模块5e-5，固定首轮5000steps。目标为原始H50 flow loss＋0.01 Gaussian NLL（动作维求和，batch/time求平均），不再只训练action expert。保存完整model.safetensors、adapter、optimizer、每rank RNG及manifest/SHA。没有验证过exact-resume前不宣称无损续训。

固定末步在validation5做Base对照和对齐/错位/去effect消融，过gate后才进入新disjoint10×8开发闭环，再按冻结10×20配对协议正式评测。不能用正式success labels选checkpoint、任务、seed或gate；正式测试已有历史曝光必须披露。

不存在自动追加的“Stage3再微调”。语言检索必须在Stage2前对齐，并在训练/部署保持一致；若需改检索，另立有验证依据的方案，不复制官方训练期episode oracle。

## 来源与明确差异

依据 [BehaviorVLA官方仓库](https://github.com/iLearn-Lab/ICML26-BehaviorVLA)，commit `0dbabc7e79791a325c4e76acde0ddfd7a18e8326`，Apache-2.0。CTE/PBD是工程命名，不代表共有结构由我们首创。

| 项目 | 本路线 |
|---|---|
| 三流Mamba、四个loss、APN、global prefix、Gaussian prior embedding residual | 沿用官方结构 |
| effect head/loss/prefix token | 本次新增 |
| LIBERO单图/7D/H10 | 适配三图/16D/H50，实际执行H15 |
| released Stage2局部token为stateless | 保留用户要求的真实H15 recurrent状态 |
| released Stage2 episode_index全局查询 | 改用训练/部署一致的task-language检索 |
| quantile normalization | 保留RoboTwin mean/std |
| 旧B₀语言拼接与post-effect Mamba流 | 移除，回到官方可学习行为token；语言用于PI和检索 |

应称“BehaviorVLA对齐＋effect的RoboTwin适配”，不能说所有细节完全一样。effect目标与JEPA相关，并不自动保证互补，须以消融和闭环检验。

## 证据状态

新路线无成功率结果。上一正式Base111/200=55.5%、旧ZeVA106/200=53.0%仍是失败。之后两个旧输出残差试验均未过验证gate，无新闭环结果。见[机制记录](ZTEV2_POST_H15_MECHANISM_20260917.md)和[本轮工程记录](ZEVA_BEHAVIOR_EFFECT_20260918.md)。

新Stage1在aigc28 GPU0运行，已完成epoch10并继续epoch11；train5230/validation270、固定80epochs/batch8。epoch10 action/effect验证改善95.31%/32.73%，direct vision改善3.76%尚未到5%，不提前提升或改门槛。真实PI双卡global256容量测试、早期CTE→语言检索→PI的H15在线/cache一致性测试已通过；后者仅微型train-only fixture，不是正式memory或成功率。Stage2和新闭环尚未开始。
