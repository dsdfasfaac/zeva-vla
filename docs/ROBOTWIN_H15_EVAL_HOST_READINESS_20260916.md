# H15 候选正式评测节点准备情况（2026-09-16）

本记录只涉及正式闭环评测的节点健康，不含任务成功率。Stage2 H15 训练、只读选模、选定模型 H50/H15 serving smoke 均已完成；**正式 Base/ZeVA 配对评测于 2026-09-17 09:45 在 aigc29 启动，尚无成功率**。任何 renderer 产生一帧图像都不等于节点可用于正式评测：还须真实物理 GPU 映射、完整进程 `exit=0`、当前资源空闲与模型两次 H50/一次 H15 recurrent 功能检查。

| 节点 | 只读资源检查与小型渲染检查 | 当前判定 |
|---|---|---|
| aigc28 | 8 卡均由他人的 PI 训练 PID943637 占用约76 GiB、100% util；先前曾完整完成同协议600视频评测 | 暂不可用，不中断他人训练 |
| aigc29 | 原先只有补 Vulkan loader 时仍无法创建 renderer。进一步定位为节点缺少 `libEGL.so.1`；在我们的隔离目录补齐 loader+EGL 后，Vulkan 枚举出 8 卡，8 张卡逐一经正式 SAPIEN 钩子验证有限帧、PID/物理 UUID/PCI 映射和干净退出。先前占用 GPU6 的 RLBench 作业已自然结束；正式启动前 8 卡均约37–38 MiB、0% util、无 compute 作业 | **已通过完整健康检查，正式评测运行中** |
| aigc24 | GPU6 UUID `GPU-2e145d56-…`、PCI `BA:00.0`、仅 Xorg；正式 ICD 和显式 `cuda:0` 可生成 finite 640×480×4 帧，**但进程退出码255，删除 camera/scene 时 `vk::DeviceLostError`**。增加显式销毁顺序、隔离使用 aigc28 的 Vulkan loader 各复现一次；`nvidia-smi` 确认探针 PID 确在该 GPU 上 | 不可用；帧级结果不能冒充完整健康证明 |
| aigc31 | 全卡约3.8 GiB不可归属残留；既有健康阻断记录对 PID2369486 的所有权/可用性无法确证 | 不绕过健康阻断，不用于正式测试 |

一帧探针来自 [`smoke_renderer_placement.py`](../scripts/robotwin_eval/smoke_renderer_placement.py)，通过现有 evaluator 的 `_configure_sapien_renderer` 钩子而非自建另一套设备选择。第一次脚本在帧后立即写了 `passed=true`，aigc24 随后 `DeviceLost`，证明该字段不足以表示完整健康；已修改为只写 `frame_passed` 与两个明确的 `process_exit_verified=false`、`physical_pid_mapping_verified=false`，完整证明必须在父进程观察干净退出后另行形成。[a24 帧级原始 JSON](results/robotwin-h15-eval-host-20260916/a24-gpu6-provisional-frame.json)保留原样，不改写历史产物。

对 aigc29 做了分层定位：初次仅从 aigc28 隔离复制 `libvulkan.so.1`（SHA256 `f2c637267fc08e343f3d617252e1ddcaf035dc714728fbdc35ce9ebd8fc4a453`）仍失败。[只读 Vulkan 枚举探针](../scripts/robotwin_eval/vulkan_device_probe.py)显示 aigc29 的 `vkCreateInstance=-9`、0 张卡，而 aigc28 为 8 张。直接查询同一 NVIDIA ICD 二进制的 `vk_icdGetInstanceProcAddr("vkCreateInstance")`，aigc29 返回空，aigc28 非空；`strace` 发现 aigc29 缺 `libEGL.so.1`。将 aigc28 的 `libEGL.so.1` **仅复制到我们自己的隔离运行目录**（SHA256 `69816a7062d7d624144472f5ef71f5b23732b9003df7472662920602e62805db`）并在子进程临时设置 `LD_LIBRARY_PATH` 后，该函数非空、Vulkan 成功枚举 8 卡。未修改系统库、共享 RoboTwin 或他人作业。

[八卡 renderer 健康父进程审计](results/robotwin-h15-route-20260916/renderer_health_a29_all8_20260916/health_proof.json)逐卡启动与正式 evaluator 相同的 `_configure_sapien_renderer` 钩子，检查 480×640×4 finite 帧、对应 GPU UUID/PCI、探针 PID 在该物理 GPU、子进程 `exit=0`；8/8 全部通过。选定 ZeVA 的[合成 serving smoke](results/robotwin-h15-route-20260916/selected_model_serving_smoke.json)另验证连续两次 H50×EEF16 输出及提交 H15 后恰有一次 recurrent transition。健康证明不包含任务成功标签；不能冒充任务成功率。

随后在 aigc24 的空闲 GPU6 用同一正式选卡钩子做了两次 opt-in 析构定位：原环境与加入上述隔离 loader 的环境都能取到完整帧，且 GPU UUID/PID 映射正确，但都在清理 camera/scene 时抛 `vk::DeviceLostError`、退出码255。因此问题不是探针在正常退出时误报，也不能仅通过补 loader 解决；没有将这两份只到帧级的 JSON 记为评测节点通过。

另查 work_env.md 所列 aigc32、aigc15、aigc01、aigc14，八卡均有其他用户进程常驻占用约74–78 GiB/卡；即使 util 为0，也不能视为空闲卡或抢占。这些节点未尝试渲染/评测。

正式评测使用[固定选模与节点健康双重预检的启动脚本](../scripts/robotwin_eval/launch_h15_selected_formal_a29.sh)，aigc29 8 卡每卡一组模型服务/renderer，固定 10×20 seed manifest SHA `1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b`、seen 指令、原门控 Base1000 与 ZeVA500；原 launcher 仍负责视频和逐任务结果。当前状态只可写为运行中；最终目标必须有真实闭环 Base/ZeVA 配对结果，不能以离线 H15 指标代替。
