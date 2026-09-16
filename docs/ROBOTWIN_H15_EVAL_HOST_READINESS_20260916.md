# H15 候选正式评测节点准备情况（2026-09-16）

本记录只涉及正式闭环评测的节点健康，不含任务成功率。Stage2 H15 训练仍在 aigc29 GPU0–3 运行，未完成。任何 renderer 产生一帧图像都不等于节点可用于正式评测：还须真实物理 GPU 映射、完整进程 `exit=0`、当前资源空闲与模型两次 H50/一次 H15 recurrent 功能检查。

| 节点 | 只读资源检查与小型渲染检查 | 当前判定 |
|---|---|---|
| aigc28 | 8 卡均由他人的 PI 训练 PID943637 占用约76 GiB、100% util；先前曾完整完成同协议600视频评测 | 暂不可用，不中断他人训练 |
| aigc29 | GPU4/5/7 当前空闲，GPU6 有他人 RLBench 采集 PID1740656。GPU7 使用正式 `/etc/vulkan/icd.d/nvidia_icd.json`、显式 `cuda:0`、固定 UUID 的一帧 smoke 报 `failed to find a rendering device`。加入隔离复制的 Vulkan loader 后仍同样失败 | GPU7 不可用；不干扰占用中的 GPU6 |
| aigc24 | GPU6 UUID `GPU-2e145d56-…`、PCI `BA:00.0`、仅 Xorg；正式 ICD 和显式 `cuda:0` 可生成 finite 640×480×4 帧，**但进程退出码255，析构时 `vk::DeviceLostError`** | 不可用；帧级结果不能冒充完整健康证明 |
| aigc31 | 全卡约3.8 GiB不可归属残留；既有健康阻断记录对 PID2369486 的所有权/可用性无法确证 | 不绕过健康阻断，不用于正式测试 |

一帧探针来自 [`smoke_renderer_placement.py`](../scripts/robotwin_eval/smoke_renderer_placement.py)，通过现有 evaluator 的 `_configure_sapien_renderer` 钩子而非自建另一套设备选择。第一次脚本在帧后立即写了 `passed=true`，aigc24 随后 `DeviceLost`，证明该字段不足以表示完整健康；已修改为只写 `frame_passed` 与两个明确的 `process_exit_verified=false`、`physical_pid_mapping_verified=false`，完整证明必须在父进程观察干净退出后另行形成。[a24 帧级原始 JSON](results/robotwin-h15-eval-host-20260916/a24-gpu6-provisional-frame.json)保留原样，不改写历史产物。

对 aigc29 缺少系统 Vulkan loader 的假设做了隔离检查：从 aigc28 系统库复制 `libvulkan.so.1` 到我们自己的 `/mnt/100T/users/dingxin/VLA/runtime/zeva-vulkan-loader-a28-20260916`，SHA256 `f2c637267fc08e343f3d617252e1ddcaf035dc714728fbdc35ce9ebd8fc4a453`；aigc29 通过临时 `LD_LIBRARY_PATH` 可加载该库，但 GPU7 的渲染错误未消失。未修改系统库、共享 RoboTwin 或他人作业，也未宣称解决根因。

后续只在有通过完整健康验证的空闲节点上启动正式评测。优先等待既有有效节点 aigc28 空出，或在不占用他人 GPU 的条件下定位 aigc29 渲染设备问题。训练/离线选模不因 renderer 暂不可用而中断；但最终目标必须有真实闭环 Base/ZeVA 配对结果，不能以离线 H15 指标代替。
