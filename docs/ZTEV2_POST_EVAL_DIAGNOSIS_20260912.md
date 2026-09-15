# ZTE v2：配对评测后的诊断边界与下一步

更新：2026-09-15。本文区分已测事实、未测假设和计划，不改变已完成实验。

**最终：固定teacher配对闭环已完成，Base110/200=55.0%、ZeVA110/200=55.0%、Anchor104/200=52.0%，验收失败。** 三条件均10任务×20 episodes；结束时间09-15 06:49:32北京时间。ZeVA−Base=0个百分点，discordant pairs=32/32，paired bootstrap 95% CI=[−7.5,+8.0]个百分点，McNemar p=1.0。Base超过同条件Anchor，但低于预设57%参考线；不能用Anchor替代训练Base宣称增强成功。完整任务表与审计限制见[正式结果](ROBOTWIN_FIXED_ANCHOR_PAIRED_RESULTS_20260915.md)，原始report/acceptance/manifest已取回`results/robotwin-fixed-anchor-20260914/formal-complete/`。以下“正在评测”均为历史快照。

这次没有通过交付目标，不继续原方案盲目加步数，也不根据正式任务涨跌改选seed/任务/倍率。下一步只在训练/验证数据上检验明确机制假设，再决定新训练；Stage1辅助分数达线既不能保证增强，也不能被此次平局反证为编码器无效。三条件600视频现均通过独立ffprobe/尺寸/计数检查，固定seed及实际指令完全配对，证据已并入formal-complete；不把ffprobe等同于全帧解码。另一个通用审计脚本的配置文本检查错误仍需澄清，不用数据审计覆盖该问题。

**09-15 03:57北京时间：Base十任务已全部完成，110/200=55.0%；状态为running_anchor，ZeVA尚未开始。** 普通训练Base仍低于预设57%参考线，不能宣布达到“正常且ZeVA更好”的交付目标。原始[Base report](results/robotwin-fixed-anchor-20260914/baseline-live/report.json)已回收；各任务依次为hammer17、RGB10、size7、handover8、mug6、dual bottles20、dustbin5、scan6、bowls13、stamp18，分母均20。不依据已看到的正式结果更换checkpoint/seed，继续既定Anchor和ZeVA闭环。

运行中dustbin seed1024曾在expert初始化`TASK_ENV.play_once()`中发生`IndexError`，既有客户端记录同一frozen seed重试1/20后继续，最终该任务完成20次、5次成功；这是被处理的初始化异常，不是新增的policy推理崩溃。未现场修改源码或替换该seed；最终seed/指令/视频一致性继续按既定审计执行。

**09-15 01:38北京时间：正式Base闭环已运行，完成2/10任务，其余任务继续；ZeVA尚未开始。** `beat_block_hammer`=17/20（85%），`pick_dual_bottles`=20/20（100%）；每个完成任务均有20条episode结果和20个非空视频文件。仅检查文件存在/非空，不冒充完整视频解码与内容审计。控制器PID1283028仍存活，其他任务日志仍在推进。以上是部分任务结果，不能合并外推为十任务总体成功率或ZeVA增益；继续固定顺序评测，不据此更换权重/seed。部分summary快照保存于`results/robotwin-fixed-anchor-20260914/baseline-live/`。

**2026-09-15 01:01北京时间：aigc28正式评测控制器已启动，通过资源及固定输入检查，正在启动模型服务；尚无新成功率。** 控制器PID1283028、PPID1，运行入口`launch_paired_formal_eval.sh`；日志为release下`formal-launch-a28.log`，已打印`Pinned models, frozen task/seed/instruction manifests and ports verified`。正式目录`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-fixed-anchor-pair-20260914`现已创建，不能再写“目录不存在”，也不能仅凭目录创建宣称episodes已开始。

8卡CUDA检查actual UUID全部匹配、finite、exit0；完整ZeVA001000实际模型报告两次[50,16] finite、commit15后transition_count=1，耗时280.14秒。[完整模型报告](results/robotwin-fixed-anchor-20260914/selected-model-full-a28-0wgpsdfj/report.json)、[8卡证据](results/robotwin-fixed-anchor-20260914/cuda-health-a28-20260914T160117/)、[汇总proof](results/robotwin-fixed-anchor-20260914/renderer-formal-health-proof-a28.json)。模型probe的SSH在结果回收前超时，退出码未知，未伪造为0；其最终报告/日志已持久化，PID1254111与父进程1254109已消失，16:58:34UTC再次确认8卡0MiB/0%。放行依据为完整功能报告、进程结束及重新检查资源；renderer的真实exit0仍是独立硬条件。新a28入口没有豁免任何不可见PID。此后不修改运行中的评测源码或公共依赖。

**迁移恢复进展（09-14 15:47–15:59 UTC）：aigc28资源与renderer通过，完整模型仍在验证，正式评测未启动。** root确认a28全8卡0MiB/0%、compute-apps为空；随后完整进程预检无C/G任务。GPU6按UUID绑定，shared RoboTwin、`/etc/vulkan/icd.d/nvidia_icd.json`下输出finite640×480，renderer PID1250866实际GPU6 C+G、干净exit0（GPU0另有7MiB枚举G上下文）。[完整证据](results/robotwin-fixed-anchor-20260914/renderer-health-a28-20260914T154905/)。a32默认进程表虽空，精查仍有每卡3.8GiB和不可见PID1182878，未判定为空闲；跳板间歇失败不影响已经取得的a28原始证据。

新增a28固定UUID的8-slot placement，保持全部checkpoint/seed/指令和H50/H15协议；仍要求无非Xorg进程、显存低于1GiB、8卡计算健康与实际模型两次推理/H15递归proof，17项CPU测试通过。新增可选`MODEL_EXTRA_DEPENDENCIES`用于隔离依赖定位，默认仍是原路径；本次a28原依赖目录已存在，没有改公共环境。初始独立import探针虽打印正确Torch2.7.1cu126/Mamba/TF5.5.4，但90秒未退出，不能计作成功；完整selected-model测试改为直接运行既有脚本，保留600秒上限及真实退出码要求，不据此宣称根因或放行评测。

**20:39 heartbeat：一次隔离依赖恢复尝试仍失败，正式评测未启动。** a31在20:40仍有普通训练PID1795058占用全部8卡（约41.5GiB、100%），另有旧PID2369486记录；不抢占。Luna将a31的Vulkan loader1.3.204与NVIDIA EGL vendor JSON复制到全新临时overlay，仅设置该测试进程的库路径；未更改系统驱动、正式配置或阻塞文件。a24 GPU2/6无非Xorg任务、UUID与tiny计算通过后，单次renderer测试于20:43在初始化阶段报`vk::PhysicalDevice::createDeviceUnique: ErrorDeviceLost`，真实exit1，无图像输出、无成功的PID落卡证明。[环境和SHA](results/robotwin-fixed-anchor-20260914/renderer-vulkan-overlay-probe-a24-20260914T124010/evidence/env.txt)、[错误日志](results/robotwin-fixed-anchor-20260914/renderer-vulkan-overlay-probe-a24-20260914T124010/evidence/renderer.stderr.log)、[退出码](results/robotwin-fixed-anchor-20260914/renderer-vulkan-overlay-probe-a24-20260914T124010/evidence/exit-code.txt)。

诊断限制：本次使用shared RoboTwin venv，而非a24正式的`/data1/.../.venv_robotwin`，不能冒充严格单变量A/B。两者已核对SAPIEN版本和libsvulkan2二进制相同，但不等于全部依赖相同。结论仅为这条隔离恢复路径没有成功，不宣称排除所有用户态因素或确认硬件根因。另查16:14曾成功的a24日志也有missing loader/GLVND警告，因此警告本身不充分解释失败。没有部署overlay到正式入口，没有继续重复尝试或新增训练。

**19:54 heartbeat复核：正式输出目录仍不存在。** 19:56根任务SSH成功，a31仍有PID1768793的普通计算任务和PID2369486的驱动记录，不把0%利用率/约4.3GiB显存当空闲；a24除GPU4为N/A外各卡34MiB/0%，仍未解除renderer健康阻塞。其余6节点本轮查询均成功且有高负载计算任务。18:46失败renderer的[原始日志](results/robotwin-fixed-anchor-20260914/renderer-recheck-a24-PwlmaG/renderer.log)与[nvidia物理落卡证据](results/robotwin-fixed-anchor-20260914/renderer-recheck-a24-PwlmaG/nvidia-during.log)现已成功回收。

附加只读诊断发现：a24失败日志有system libvulkan和GLVND ICD缺失警告，系统ldconfig未列出libvulkan，a31成功日志没有这些警告。正确设置`VK_ICD_FILENAMES`不会自动补齐Vulkan loader或EGL vendor JSON。正在核对两环境的user-space依赖差异；这不是已确认的DeviceLost根因，不据此解除安全阻塞，也不改系统驱动或模型/场景协议。

**19:21 heartbeat：aigc31现有明确的高负载训练占用，aigc24仍未通过renderer健康复核；没有新评测结果。** Luna在19:22:15取得的a31原始输出显示，PID1733831（`/mnt/100T/xielele/openpi-cleaned-droid-venv/bin/python`）占用全部8卡，每卡约29.2GiB、99–100%利用率。旧PID2369486仍是`/proc=present`、`kill -0=Operation not permitted`，**不是ESRCH**；Luna首份摘要再次误写为消失，经索取原始行已纠正，不采用该摘要作为解除依据。a24 GPU1/2/3/5/6/7仅Xorg、34MiB/0%，但健康阻塞仍在；a32/29/15/01/14有计算任务，a28本轮SSH reset未取得快照。没有停止或抢占任何任务。

根任务本轮向a24拉取原始证据、向a31新建只读连接均立即返回 `Connection closed by 36.189.234.168 port 33`（exit255），而Luna上述连接成功；因此只能记录新连接失败，不能宣称跳板全面中断。未改变认证/SSH配置。Luna本轮未保存原始快照文件，证据仅存在工具返回记录，不提供虚构的文件链接。

上一轮18:46的aigc24复核也没有解除健康阻塞：物理GPU6使用正确UUID和正式ICD，生成finite640×480帧后仍发生 `vk::DeviceLostError`、exit134，renderer PID3877344被核实在GPU6。因此不能再用16:14的成功证明当前renderer健康。远端原始目录为 `/mnt/100T/users/dingxin/VLA/fixed-anchor-pair-20260914-P8hRUO/renderer-recheck-a24-PwlmaG/`，本轮因SSH失败尚未取回，不提供不存在的本地原始证据链接。已收取此前待返回的tiny CUDA结果：GPU2/6的actual UUID均匹配，Torch2.7.1+cu126且finite/exit0；这只证明微型CUDA计算通过，不能抵消renderer退出失败。保持两台机器的安全阻塞文件，不清理他人进程、不重置GPU；截至最后一次成功检查正式输出目录不存在，无新成功率。

**18:33 heartbeat后最新：aigc31完整模型/递归/渲染及8卡计算均通过，但正式启动被资源保护拦截；没有运行episode。** 既定ZeVA001000完整模型加载和两次真实推理通过，输出均[50,16] finite、commit15后transition_count=1，耗时284.34秒，使用Torch2.7.1cu126与实际UUID6。[真实模型报告](results/robotwin-fixed-anchor-20260914/selected-model-smoke-a31-pEWPcm/report.json)。renderer PID1697338已由独立nvidia PIDS核实在GPU6/BA:00.0，640×480且干净rc0；8卡tiny CUDA均actual UUID匹配、finite，约5.54秒。已准备一卡一slot，仍为同样8个逻辑RNG流、固定checkpoint/200组seed及实际指令；19项相关CPU测试通过。

**重要纠正：不能把PID2369486当作已确认退出的进程。** 首次launcher PID1720516在创建正式输出前退出，日志为`GPU 0 has non-Xorg processes; launch cancelled: 2369486`。随后严格检查得到`/proc/2369486`目录存在、status文件不可见，`os.kill(pid,0)`为`EPERM(errno1)`，不是`ESRCH`。此前Luna由ps不可见/kill-zero失败推导“进程不存在”不充分，且被本次内核结果否定；本次不推断具体owner或内核原因，不绕过guard。计算健康通过不等于这些显存可自由占用。a31入口增加安全阻塞记录，需确认GPU分配/共用授权或进程确实消失后再启动；未终止进程或重置设备。仍在检查其他节点可用资源，无新成功率。以下资源推断按本段修正。

17:37 heartbeat后：aigc31模型迁移阻塞已定位为ABI不匹配——系统model Torch2.5.1cu124无法导入现有Mamba编译扩展（undefined symbol），而a24模型用Torch2.7.1cu126；renderer单独Torch2.10通过不代表model可运行。已授权Luna创建全新的shared依赖overlay，优先复制a24的匹配Torch及必要依赖，不更改公共环境或编译现有Mamba。入口新增可选`MODEL_DEPENDENCY_OVERLAY`，保持源码/native TF5优先，记录manifest。a31专用placement要求独立成功proof与overlay路径完全匹配，UUID固定物理2/6，只豁免已确认不存在的历史PID2369486，内存门槛6GiB用于容纳已测3.8GiB残留；任何新活跃C/G进程均拒绝，绝不清理旧句柄。6项placement、4项runtime、5项UUID映射测试通过。正式评测尚未启动。

迁移场景一致性也已只读核对：a24的`/data1/dingxin/robotwin-formal-eval/RoboTwin`与shared RoboTwin的`envs`（忽略`__pycache__`）和整个`task_config`分别`diff -qr`均无差异、rc0。因此候选shared renderer并非未经比对的新场景配置；仍须完成新model runtime真实加载/推理验证。a31本地/data1仅余37GiB，依赖和评测产物使用shared盘（当前约11TiB可用）。

**16:45 heartbeat后的最新状态：UUID固定设备、正式ICD的aigc24恢复验证通过，但启动前GPU2/6已被他人渲染任务占用，未启动正式评测。** 16:09–16:14的完整证据现已回收：两卡tiny matmul的Torch UUID分别匹配物理2/6且finite/rc0；renderer PID3674669实际PCI BA:00.0、nvidia GPU6 C+G、finite640×480帧、干净退出rc0。[原始证据](results/robotwin-fixed-anchor-20260914/renderer-formal-health-20260914T1534/)、[核验汇总](results/robotwin-fixed-anchor-20260914/renderer-formal-health-proof.json)。目录名1534不是实际运行时间，以metadata的16:09–16:14为准。UUID机制已加入模型与renderer launcher，并保留数字物理映射和全部8个逻辑RNG流；5项映射、3项runtime、4项client测试通过。专用入口同时检查UUID对应关系及非Xorg进程，不能把0%/低显存的G任务当空闲。健康恢复不等于资源已预留，旧blocker仍保留到正式恢复交接。

备选aigc31：原驱动记录PID2369486无对应OS进程，历史worker持有句柄不等于正在训练；因此不做清理，先用极小计算验证是否能与残留分配共存。物理2/6 UUID限定的tiny CUDA finite通过（系统Torch没有UUID属性，未将该输出单独作为物理身份复核）。随后使用现有`/etc/vulkan/icd.d/nvidia_icd.json`（与a24相同内容）和shared RoboTwin Python，UUID6 renderer输出PCI BA:00.0、finite640×480帧，进程干净rc0，证据在`release/renderer-health-a31-FlDVZb`；5秒PIDS采样未捕获renderer PID，不宣称已有完整nvidia PID独立证明。shared renderer Python为3.10/Torch2.10cu128，与a24模型runtime尚未核对，正在做只读依赖审查，不直接迁移模型。以下资源状态为历史记录。

16:06核对补充：Luna确认15:20那次renderer smoke未设置正式的`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`。故那次DeviceLost/Vulkan fallback不能直接代表完整正式环境失败；tiny CUDA失败与实际落卡偏移仍是需要解决的独立观测。当前aigc24 GPU4仍ERR，其余卡0–1MiB/0%且无可见进程。保持health blocker，安排一次GPU UUID固定物理身份、正式ICD/当前client hook完全对齐的验证：先有限时tiny CUDA，两卡任一失败即停止；只有均通过才验证renderer与干净退出。未启动正式评测。

**最新资源状态（15:13–15:22）：aigc24 的 nvidia-smi 恢复返回，但计算/渲染未恢复验证通过，正式评测仍未启动。** GPU4显示ERR/N/A，其余卡状态表只有Xorg。随后有超时限制的smoke中，`CUDA_VISIBLE_DEVICES=6`、`cuda:0`实际renderer PID3615310位于PCI DB:00.0/物理GPU7，而非BA:00.0/GPU6；生成640×480帧后以DeviceLostError退出，rc134。可见帧不等于健康，脚本硬编码的`physical_gpu_index=6`和`passed=true`也不是成功依据。CUDA可见序号2、6下的tiny matmul都报launch timeout；未记录其实际PCI/UUID，不能将日志中的physical_gpu标签当作已核实物理身份。故障后的枚举变化是待核实假设，不是确定根因。当前smoke的ICD环境与正式launcher的一致性仍需核对，未据此断言正式ICD配置错误。

原始smoke已归档于[恢复后验证目录](results/robotwin-fixed-anchor-20260914/renderer-device-smoke-20260914T1520-gpu6-rerun2/)，远端此次日志误放在旧`eval-release-ztev2-20260912`下的独立新子目录，未启动或覆盖旧正式结果。专用入口新增`gpu-health-blocked.json`硬拦截；旧的成功选卡proof不能解除本次阻塞。须以正式runtime完成新的finite CUDA、真实物理GPU身份、renderer干净退出验证，再显式记录解除。未重置GPU、停止他人进程或启动正式任务；无新成功率。以下资源快照为历史记录。

最新运行状态（09-14 13:31 heartbeat 后核查）：显式 SAPIEN 选卡已通过真实 client hook smoke。`CUDA_VISIBLE_DEVICES=6` + `ZEVA_SAPIEN_RENDER_DEVICE=cuda:0`，通过 `sapien.core.SapienRenderer()` 创建的 PID3447001 经 nvidia-smi 确认在物理 GPU6 / PCI BA:00.0，输出480×640×4帧。证据转录见[选卡验证](results/robotwin-fixed-anchor-20260914/renderer-device-smoke.json)；原独立日志未保存，不能冒充原始日志。4项client测试、4项映射测试和3项只读runtime测试本地通过，修复已部署隔离release，未改公共RoboTwin。

本轮启动前发现新的资源健康阻塞：aigc24 SSH/hostname正常，但全局 nvidia-smi 超过2分钟不返回，单独GPU2/6查询也在8秒超时后强制结束（rc137）；进程出现D态，其他任务的查询同时挂起。尚不能确定驱动或硬件根因。未重置GPU、未停止他人任务，正式输出目录仍不存在。专用入口增加GPU查询超时并失败即停止；正在只读检查其他授权节点，不能声称正式闭环已经开始。训练与checkpoint不变，无新成功率。

14:07 heartbeat复核：aigc24的限时GPU查询仍以rc137结束，正式目录仍不存在。Luna上一轮完成其余7台授权节点的完整进程检查，未发现至少2张无占用的GPU：31每卡约3.8GiB且同一计算进程占据8卡；32/29/15/28约75–80GiB，01/14约51GiB。不能把31利用率0%视为无人占用。当前保持等待安全资源，不重置驱动、不抢占他人进程；这不是新的模型或评测结果。

14:08追加快照：aigc31每卡仍约3809MiB/0%，但默认nvidia-smi进程表变为`No running processes found`，与上一轮PIDS查询结果不一致。已交Luna核实PIDS/compute-apps及设备使用者，未据此宣布卡空闲或启动新进程。

14:40 heartbeat：aigc24仍以rc137结束限时GPU查询，正式目录仍不存在。aigc31追加检查确认PIDS/compute-apps中的2369486为`[Not Found]`且`/proc/2369486`不存在；另有8个PPID1、约52天的`pt_data_worker`持有全部GPU设备句柄。因此此前“同一计算进程占用”应精确理解为驱动残留记录，不能当作活跃训练证明，也不能仅凭默认进程表为空宣布可用。未清理这些历史进程或重置设备；继续检查其他授权节点的资源释放。

## 最新：固定teacher方案未证明增量收益，准备固定末步闭环

两路终检于09-14 02:03均完成735 batches/5874决策。checkpoint、adapter、Stage1/bank/live/retrieval、样本顺序SHA、seed1000、batch8、源码及预处理协议核对一致，两路student residual-on/off统计完全相同；两种teacher的冻结权重均逐张量核验相同。传回本地曾遇SSH/SFTP挂起，终止本次传输进程并通过带超时的rsync恢复，未影响已完成的训练/验证。

| 固定同样本、同噪声的 H15 路径 | 样本平均 flow error |
|---|---:|
| 训练起点 Base004500 | 0.010180731304 |
| 同预算新 Base001000 | 0.010030957870 |
| 新 ZeVA001000，关闭残差 | 0.010142331943 |
| 新 ZeVA001000，开启残差 | 0.010143543594 |

ZeVA相对训练起点改善0.3653%，但新Base改善1.4711%，ZeVA比新Base差1.1224%；开启残差比同权重关闭残差略差0.01195%，逐样本胜率48.47%。H50 residual-on=0.018861809745、off=0.018864864483，H50微小正收益不能替代实际执行H15上的负点估计。prior NLL=11.60375。未计算episode级置信区间，不能宣称统计显著；这些不是成功率。

**预设动作收益检查：相对固定起点不回退满足；普通Base相对起点不回退满足；residual-on应优于off不满足，且ZeVA不及同预算Base。** 因而不支持“仅换固定teacher足以提供ZTE增益”，不继续该方案的盲目加步数/倍率搜索。它仍未证明Stage1编码器无效。保留预定step1000做正常配对闭环，不能把负的离线点估计直接解释成成功率下降，也不根据正式标签改选checkpoint。

固定产物：ZeVA model SHA=`f5e4812a01e01da9936ee23a8f2c9e7d5e70037112f821e759a329d37bc4d11a`，adapter=`cbf0100994d43d7faa2aba23d4f38ae665331f565df53636d6435654c25a1c57`，新Base model=`bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17`。[对固定起点原始报告](results/robotwin-fixed-anchor-20260914/vs-fixed-anchor.json)、[对同预算Base原始报告](results/robotwin-fixed-anchor-20260914/vs-matched-base.json)。

闭环将复用原`eval/formal-ztev2-selected-pair-20260912/seed_manifest.json`（SHA=`1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b`）中的200组seed/实际seen指令，不重筛；本地已有原manifest，哈希再次一致。维持10tasks、Large_D435640×480、demo_randomized语义的`zeva_randomized`配置、H50/H15、视频审计。拟同时重跑untouched Anchor核验正常性，Base门槛仍为max(同条件Anchor,57%)；不放松。当前Luna正在检查空闲renderer并做隔离初始化smoke；正式闭环尚未启动。

配置预检首次SSH调用超时，但后续经aigc24共享路径确认远端已成功生成3个配置及完整manifest（mtime09-14 08:10），没有重跑或覆盖目录。新Base/ZeVA的model SHA与完整诊断报告吻合，train-only bank/语言及物理协议通过检查。[配置预检原始manifest](results/robotwin-fixed-anchor-20260914/eval-staging/robotwin_eval_ztev2_staging_manifest.json)，远端目录为`RUNROOT/eval/formal-fixed-anchor-pair-20260914-staging`。正式launcher增加可选`READ_ONLY_RUNTIME=true`和`NATIVE_TRANSFORMERS_RUNTIME`，复用已验证共享overlay，不改公共runtime symlink；3项隔离shell分支/语法测试通过。默认历史分支保持不变，新开关不改推理数学或模型权重。

09-14中午资源复核：不能把`--query-compute-apps`为空当作GPU空闲。aigc24多张卡上有他人的RLBench **G类图形进程**，显存仅约141MiB但利用率100%；未停止或占用这些进程。确认可用的是物理GPU2/6；aigc29和其余授权H100均有训练占用。Luna已完成按slot显式`MODEL_GPU_IDS`/`RENDER_GPU_IDS`映射，4项映射测试及3项只读runtime测试通过。计划在aigc24同机使用映射`2,6,2,6,2,6,2,6`，保留8个独立model RNG流及原任务分配，不把逻辑slots改为2；每张卡需承载4个模型/renderer，启动后必须监测实际显存，不能预先保证性能。

专用入口为`scripts/robotwin_eval/launch_fixed_anchor_pair_20260914.sh`，已部署于同一隔离release；它要求新输出目录、再次确认GPU2/6空闲、核对seed/task/model哈希和19300–19307端口。当前仍在完成SAPIEN实际物理选卡核验，**正式入口尚未执行**；不能凭CUDA环境变量设置就声称不会落到其他卡。前次普通相机smoke验证了640×480渲染，但不替代PID→物理GPU的核验。此前中断保留了映射文件，恢复后未重复启动评测。

随后物理选卡smoke发现真实问题：`CUDA_VISIBLE_DEVICES=6`下PID3399952的SAPIEN **G进程实际位于GPU0/PCI18:00.0**，而GPU6/BA:00.0只有Xorg；smoke自然退出。这证明仅传CUDA映射不足以约束Vulkan renderer。正式入口继续暂停；Luna正在验证显式SapienRenderer设备参数/最小process-local钩子，不能改公共RoboTwin runtime。专用入口将要求保存的`renderer-device-smoke.json`证明与显式设备选择一致，否则拒绝启动。不要运行旧的未验证CUDA-only映射。

历史记录（2026-09-14 01:40核查）：两支均完成1000步。Base于01:24:59、ZeVA于01:30:46完成，两个launcher进程退出，`COMPLETE`及`latest.json step=1000`均确认；每支model.safetensors为9354050752字节，均有optimizer/scheduler training_state，ZeVA另有adapter。训练循环含验证/保存耗时分别45分07秒、50分42秒，另有启动开销。[Base完整manifest与末步记录](results/robotwin-fixed-anchor-20260914/baseline/manifest.json)、[ZeVA完整manifest与末步记录](results/robotwin-fixed-anchor-20260914/zeva/manifest.json)已归档。末步旧口径H50 validation：Base自身flow=0.02059016；ZeVA flow=0.02085952、同次固定teacher=0.02076325（该次ZeVA略差），NLL=11.61310。两支validation RNG状态不同，不能将两个flow直接当同噪声paired比较，也不能据此回头挑其他checkpoint。

预设末步的完整只读终检已启动：隔离目录`/mnt/100T/users/dingxin/VLA/fixed-anchor-pair-20260914-P8hRUO`下，aigc29 GPU0/PID2767372输出`vs-fixed-anchor.json`（teacher=原Base004500），GPU1/PID2767373输出`vs-matched-base.json`（teacher=本轮Base001000）；同名`.log`记录过程。两路均使用ZeVA001000、batch8、seed1000、eval_batches0覆盖完整5874决策，分别给出H15/H50、current residual-off和固定teacher结果。它们不创建optimizer或修改checkpoint。当前仍在加载，没有终检结论或新闭环成功率。Luna同时只读核对后续正式评测的固定seed/实际instruction复用与启动协议，不重筛测试样本。

历史进度（2026-09-14 01:09核查）：固定teacher匹配训练正常运行。aigc29 GPU0–3为ZeVA（launcher PID2701483），GPU4–7为普通Base（PID2701482），各4卡、global256、1000新optimizer steps。Base已超过576步（000500完整checkpoint可用），ZeVA正在500步验证/保存（上次确认完整为000250，不能把当时空的000500目录当成完整产物）。稳态训练约Base1.9秒/步、ZeVA2.1–2.3秒/步，不包含阶段性验证/保存。active flow/NLL有限；Base的prior/retrieval NaN是禁用项，不是训练发散。实际manifest确认两支初始化相同、AE430098464参数/LR5e-6；ZeVA另有2910532参数/LR5e-5，teacher=`independent_frozen_base_action_path`、每4步抽样并同噪声重放，scheduler每optimizer step推进一次。真实compiled anchor/zero-init/H15预检及6项launcher契约测试均通过。训练输出根为`/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914`，各分支`train.log`、`manifest.json`和checkpoint写在其`baseline/`、`zeva/`下。隔离代码与外层启动日志位于`/mnt/100T/users/dingxin/VLA/fixed-anchor-pair-20260914-P8hRUO`，分别`baseline-launch.log`、`zeva-launch.log`。不要启动第二份，也不要将此新实验伪装成旧ZeVA resume。

## 当前结论：全量验证已完成

**最新执行状态（2026-09-13 23:56 核查）：四组完整消融均已结束。** aigc29 GPU0/1/2/3上的context-only（PID2655485）、prior-only（2655489）、prior×50（2655493）和同机原始1/1对照（2655497）均完成735 batches，结果文件时间约23:35，进程已退出。每组5874决策、batch8、seed1000，OMP/MKL各8线程，checkpoint及冻结Base不变；同机对照用于控制从aigc24迁移带来的环境差异。没有启动新训练或新的闭环评测。

### 同机四路消融结果

| 配置（context倍率 / prior倍率） | 样本平均 H15 flow，越低越好 | 相对关闭双残差改善 | 样本平均 H50 flow |
|---|---:|---:|---:|
| 原始对照 1 / 1 | 0.010179412551 | +0.05914% | 0.018888637424 |
| context-only 1 / 0 | 0.010177855380 | +0.07443% | 0.018888257444 |
| prior-only 0 / 1 | 0.010185945779 | −0.00500% | 0.018895527348 |
| prior 放大 1 / 50 | 0.010180360638 | +0.04983% | 0.018888372928 |

四组的checkpoint、adapter、fixed teacher、全部lineage、诊断源码哈希及除干预倍率外的protocol逐项相同；均`complete=true`、5874决策。样本顺序SHA一致，关闭双残差的H15均值均为`0.010185436345636845`，固定Base的H15均值均为`0.010180731303989887`。新同机原始对照的H15/H50均值还精确复现了先前aigc24报告。这里的均值一致支持对照可比性，不冒充逐样本误差的bit-exact核验。

**可支持的结论：当前权重下，弱小的离线收益主要来自context分支；prior单独没有降低平均H15误差，加入context后反而略抵消其收益。简单把prior放大50倍没有改善H15，不能将“gate太小”当成充分解释。** H50放大后略好、H15略差，进一步说明部署前15步必须单列观察。差值很小，未计算episode级置信区间，不能声称统计显著，也不能推出ZTE信息本身无效。未经该倍率训练的压力测试不等价于重新训练后的结果。

原始报告：[原始对照](results/robotwin-ztev2-20260912/same_host_control.json)、[context-only](results/robotwin-ztev2-20260912/context_only.json)、[prior-only](results/robotwin-ztev2-20260912/prior_only.json)、[prior×50](results/robotwin-ztev2-20260912/prior_strength_test.json)、[运行计划](results/robotwin-ztev2-20260912/plan.json)。这些结果不能用作正式测试选checkpoint或直接决定部署倍率；不据此删除用户指定的prior路线。下一步收敛到prior监督/梯度路径的最小训练改动审查，而不是继续扫gate。

### 下一轮：固定 Base teacher 的匹配训练（2026-09-14，已提交启动）

Luna的监督/梯度审查没有发现NLL遗漏反传：NLL监督prior及上游context表示，flow监督action expert和两路注入投影；prior dropout只作用于注入而不关闭NLL。没有据此修改NLL、倍率或Stage1。已确认的可改机制是preservation当前默认使用随student更新的residual-off对照；它不是独立Base能力锚点。

计划让普通Base和ZeVA都从既定`baseline/004500`出发，各再训练1000 optimizer steps、warmup100、每250步存档，固定step1000为比较候选。ZeVA使用同一004500作为immutable preservation teacher并初始化新的零残差adapter；不加载旧ZeVA005000 adapter。两支均保留指定best-v1起源、共享历史4500步和同等后续AE预算，optimizer重新建立，明确是**新实验，不是旧ZeVA续训**。冻结ZTE/bank/VLM、AE LR5e-6、新模块LR5e-5、global256、双残差/Gaussian NLL、dropout0.4、H50输出/H15执行及门控初始化均保持。共有初始化和预算变化意味着不能把它与旧实验的差异解释为纯单因素teacher因果效应。

**边界：现有hinge是`relu(student_flow - teacher_flow.detach())`，只对较差样本增加专家标签的flow梯度权重，不是teacher动作蒸馏，更不保证策略能力被保留。** 固定teacher是否有益必须实际测量。训练后分别报告fixed anchor、current residual-off及同预算新Base的样本平均H15/H50；不拿H50替代H15，不以正式成功标签选步或倍率。若没有residual正增益或出现能力退化，记录不支持此假设，不把该轮包装成成功。1000步是有界验证预算，不代表已承诺训练充分或可交付。

真实模型预检位于`/mnt/100T/users/dingxin/VLA/fixed-anchor-pair-20260914-P8hRUO`，smoke PID2691401已完成，报告`passed=true`：共同Base初始化的零残差on/off均bit-exact、H15 transition=1；`torch.compile`下扰动student后teacher不变，调用后student权重正确恢复。模型SHA为既定`2f106633…`，Stage1/完整bank/live来源匹配。[原始smoke报告](results/robotwin-fixed-anchor-20260914/compiled-anchor-smoke.json)。预检使用合成图像/动作，只验证前向实现，不证明真实样本训练反传或任务表现。aigc29 `/data1`仅约2.2GB空余，新增checkpoint和编译缓存全部置于`/mnt/100T`；不清理或覆盖他人文件。随后提交两支训练，状态见顶部。[预注册配置](../configs/robotwin_ztev2_fixed_anchor_pair_20260914.json)固定预算、lineage和末步候选；6项标准库launcher契约测试在本地及aigc29均通过，不将它们冒充真实训练反传测试。

输出目录：`/mnt/100T/users/dingxin/VLA/diagnostics-ztev2-ablation-20260913-fGari2/full-fourway`，包含四组各自的`.pid`、`.log`，完成后写同名`.json`；`plan.json`固定实验设置。单次启动脚本有空闲GPU检查、输出目录拒绝覆盖与独立日志，不是另建监视任务。

Luna已确认已有a29/a24数据全量identity reports的四组件指纹完全相同，报告自身SHA与恢复provenance一致；当前a29 adapter SHA=`8ac54abcec7704b0111b7c28be3fb3a18e27e0ebe8e3dcf8ddff948b36100f8f`匹配原证明。此次只复核报告和adapter，未重读80GB原始数据。显式使用a29原路径`/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data`，不修改历史manifest里的a24路径。

a29未安装pytest（测试命令未执行，不计通过）；8项隔离CPU Tensor/控制流检查在Torch2.7.1+cu126上通过。真实prior×50预检也通过：模型/adapter/fixedBase SHA一致、冻结路径逐张量一致、样本顺序SHA=`a9c6a8e30aa3f7ecbfcb9ce81d6a141ef563300d41e339a22d06fb75a0418ac1`；H50 replay差值0。[两条预检记录](results/robotwin-ztev2-20260912/ablation-preflight-aigc29-prior50-b2x1.json)仅用于实现核验，不能外推总体收益。

固定 ZeVA005000 / Base004500 的 **5874 个验证决策、735 batches** 已全部完成（结果文件时间 09-13 01:52；验证循环17分48秒），两模型冻结路径逐张量一致。模型、adapter、ZTE、bank、retrieval 和 normalization SHA 与既定产物匹配。[完整原始报告](results/robotwin-ztev2-20260912/step5000-diagnostics-full-base4500-b8.json)。

| 相同样本、相同 flow noise 的 H15 指标 | 平均误差 |
|---|---:|
| ZeVA 当前 action expert，关闭双残差 | 0.0101854363 |
| ZeVA 当前 action expert，开启双残差 | 0.0101794126 |
| 独立训练的固定 Base004500 | 0.0101807313 |

双残差带来的平均相对改善仅 **0.0591%**，对固定 Base 的平均相对改善仅 **0.0130%**。这不是成功率，也没有据此声称统计显著。残差开启相对关闭的样本胜率为50.85%，相对固定Base为46.75%；同一episode内的决策有关联，不能当成5874次独立闭环试验。

H15 context/prior 残差相对 noisy-action embedding 的**重建范数比**分别为 `0.0034690`（约0.347%）和 `0.0000149977`（约0.00150%），prior 约为context的1/231。retrieval accuracy=99.8469%，prior NLL=6.76919。这说明目前接入带来的动作预测增量非常小，但不独自证明ZTE没有信息，也不证明增大gate一定有效。

原验证 `flow` 对batch均值再平均，末批只有2条；新诊断按样本平均，因此本次原指标与新H50聚合不作精确等价声明。单批无末批加权差异的真实smoke已验证差值0。当前数据未提供显式action padding mask，诊断使用`implicit_all_action_steps_valid`；不能据此断言轨迹末尾没有重复填充。

上述预先指定的验证集分支消融已完成，结果见顶部；不是正式测试选参，也未变更部署权重。保持同5874决策、batch8、seed1000、固定Base及所有物理协议。[预先固定的消融设置](../configs/robotwin_ztev2_validation_ablation_20260913.json)。放大后变差也不能单独证明ZTE无信息，因为当前权重并非在该幅度训练。

历史资源与实现状态（09-13下午，已被顶部启动记录覆盖）：Luna连续遭遇transport错误，root接手完成只读门控开关，默认1/1不替换方法；异常退出也恢复原方法。8个本地隔离控制流用例通过，当时未作完整tensor/runtime测试。8台授权H100当时全部高负载，未抢占他人作业；随后释放资源、完成真实预检并启动消融。没有新训练。

## 历史过程：2026-09-13 验证进度

01:32 更新：真实 ZeVA005000 的单批 smoke 已完成，模型/adapter/ZTE/bank/retrieval SHA 与正式选定产物完全匹配。两条真实验证决策上，原 H50 flow 与 raw replay 均为 `0.0061195502`，差值为零；H15 residual-on=`0.0051624347`，current residual-off=`0.0053198677`。同批 H50 略差而 H15 略好，仅说明两种观测不能互相替代，**不能从两条样本外推总体增益**。H15 context/prior 相对范数的重建均值分别约 `0.002976` / `0.00001348`，不是直接捕获的 BF16 累加后差值。完整 smoke JSON 已[归档](results/robotwin-ztev2-20260912/step5000-diagnostics-smoke-b2x1.json)。

独立只读 CLI 与加载器已完成，相关远程测试累计 **16 passed in 15.34s**。首次固定 Base 对照加载被原 `load_foundation_anchor` 的起点一致性保护拒绝，未进行验证：该方法要求先在 teacher 权重上建立 anchor。修复为先逐张量确认两模型的冻结路径相同，再加载 Base004500、建立 anchor、恢复 ZeVA005000；未修改 policy 或训练权重。失败日志保留在原隔离目录，不伪装成成功。

完整 **5874 个 validation5 决策**的只读测量已在 aigc24 GPU6 启动，batch8、单进程、不 compile、无 optimizer；同时分别比较 current residual-off 与固定 Base004500。新隔离目录为 `/mnt/100T/users/dingxin/VLA/diagnostics-ztev2-fixedteacher-20260913-a5evvD`，日志 `step5000-full-base4500-b8.log`，完成后独占写入同名前缀 JSON。此时仍在加载，尚未声称完整测量已完成。正式测试成功标签不输入此工作流，未启动新训练。

- 默认关闭的诊断实现已完成初版；远程 aigc24 隔离目录 `diagnostics-ztev2-20260913-lubDbq` 中，5 项新诊断测试与 5 项既有 Stage2 v2 测试实际通过（13.13 秒），不是本地缺 PyTorch 导致的 skip。生产源码与已有 checkpoint 未覆盖。
- 代码审查发现可选 raw-forward 的失败可能因 rank 而异，而其 gather 是条件调用；已限定诊断为单进程，避免多卡条件 collective 挂起。普通训练默认路径不受影响。
- 加入单进程保护回归测试后，远程复测为 **11 passed in 9.30s**。
- 核对实际 handoff `PI05Policy.forward`：历史 flow 截取 EEF16 后对 H50/动作维取均值，并不使用 `action_is_pad`。新增 valid-only H15/H50 是另行标注的诊断统计，不能在有 padding 时冒充历史指标的精确重放；这项发现本身还不证明闭环掉点由 padding 引起。
- 只读 checkpoint 验证入口正在实现，尚未得到真实权重上的 H15 或注入幅度数值。context/prior 分量根据实际投影、gate、confidence 和 embedding dtype 重建；不声称已经直接捕获了 BF16 累加后的有效差值。需要先核对重放结果，再据结果决定训练策略。

## 已测事实

- 固定的正式配对结果：Base 109/200，ZeVA 110/200，Anchor 101/200。ZeVA 多成功一次，不构成稳定优势证据；完整结果见[评测记录](ROBOTWIN_ZTEV2_PAIRED_RESULTS_20260912.md)。不据此回头选择 checkpoint、任务、seed 或指令。
- 当前 Stage1 为 step4096。1350 条验证轨迹上，task=96.8889%、order=88.6432%、effect cosine=0.375731、language consistency=0.851242，均超过配置参考线 50%、80%、0.05、0.80。这些是辅助诊断，不是 BehaviorVLA 官方标准。
- `train_robotwin_zte_v2.py` 将 `probe_gate_status` 写为 `diagnostic_only`，并固定写入 `probe_gate_passed=false`；不能把该布尔值解释为某个测量不及格。order 来自独立 progress head，不能代替实际导出 phase token 的质量测量。
- Stage2 没有指定固定 anchor 权重时，`_matched_baseline_flow` 调用当前 student 的 foundation，关闭 ZeVA 注入作为对照。action expert 同时更新，因此这个对照不是独立训练 Base，也不是固定能力锚点。
- 当前正式选步使用 held-out H50 flow。部署实际执行 H15；二者契约没有冲突，但既有汇总没有单独解释前 H15 的预测收益。

## 尚不能下的结论

- 约 1% 的 sigmoid gate 不足以证明注入太弱：还需测量投影后、乘 confidence/gate 后的残差相对 action embedding 的幅度。
- ZTE 辅助分数达到参考线，不足以证明它提供了 PI 尚未掌握的动作信息；同样，闭环增益不足也不能单独证明编码器无效。
- 某个最后训练 batch 的 NLL 与全验证集 NLL 不能直接当成泛化差距；需要同口径、同样本统计。
- H50 输出/H15 执行是用户指定协议，不把它作为待修复错误，也不改为 H10 或 H15 输出。

## 历史实施要求：先增加验证观测（已完成）

该阶段由Luna修改本地Stage2 trainer和专门测试，随后在隔离远程副本完成验证；未覆盖生产源码、未启动新训练。原实施要求：

1. 不改变历史 loss、模型权重、门控、优化器、训练步数或默认验证输出。
2. 分别记录 H50 与前 H15 的 flow error，明确 action padding 与 EEF16 维度处理；不从已归约的标量推测 H15。
3. 当前 student residual-off 与固定 teacher 单独命名；没有加载并核验固定 teacher 时明确 unavailable，不能冒充已测 Base 对照。
4. 测量或明确标注重建的 context/prior 残差和相对幅度，不用 gate 数值代替。额外 forward 必须恢复 RNG 和临时注入状态。
5. 单测完成后，先对既定 ZeVA005000 在独立验证样本上做只读重放；保留 checkpoint SHA、样本、噪声与运行时来源。正式 rollout 成功标签不进入这个测量。

上述验证现已完成，数值见本文顶部。下一轮训练须根据实际观测确定单一可解释改动，继续冻结 ZTE/bank/VLM，保留 AE LR5e-6、新模块 LR5e-5、global256 和现有双残差/Gaussian prior 路径。

若采用额外 action-expert continuation，必须给普通 Base 匹配的额外训练预算与数据；只给 ZeVA 增加 AE 更新不能称为表征增益的干净隔离实验。未启动新的 Stage1、Stage2 或 Stage3。
