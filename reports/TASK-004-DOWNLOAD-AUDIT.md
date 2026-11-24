# TASK-004 历史下载事实审计

本次整改开始时，检查了 Windows 与 WSL 中的项目下载命令和进程，未发现仍在运行的项目下载器。本轮读取已有缓存、回执、网络日志和实验输出，新增文件下载为 0 字节。保留已有数据、缓存与断点。之前按项目负责人指令删除的 ZIP 已无法再次读取，以下 ZIP 完整性证据来自当时保存的 SHA-256、成员 CRC 和清单回执。[机器可读审计](../results/vicad_refinement/evidence/historical-download-audit.json) · [本轮零下载计划](../results/vicad_refinement/evidence/rectification-local-plan.json)

版本为 `05abc5d03167354a7ef98541236b5c0bd00abadf`。下表时间为北京时间，单位为字节。日期未单列的各行发生于 2026-11-24。

| 类别 | 完整归档字节 | 已知起止时间 | 记录的源和路由 | 可核验的本地分块 |
|---|---:|---|---|---|
| Bird | 15,506,344,175 | 11-23 约 21:25 至 23:45:35 | 官方入口，Git LFS 回退至 `us.aws.cdn.hf.co`。代理环境有记录，逐连接路由未核实。原 Xet 回退无 ZIP 字节落盘。另一次探测收到 8,261,662 字节，无法确定是否为重复流量。 | 停止记录显示完整归档在本地，初始保留分块字节未核实。 |
| Motor | 14,955,220,087 | 07:19:00 至 07:34:06 | 官方入口及 Git LFS CDN，逐连接路由未核实。 | 停止记录显示完整归档在本地，初始保留分块字节未核实。 |
| Teapot | 15,758,841,927 | 07:34:10 开始。08:29:44 暂停后离线组装，08:36:43 启动评价前已核验。 | 停止前旧官方路径，逐连接路由未核实。暂停后的组装与核验完全离线。 | 停止时 235 段，15,758,841,927 字节。本地组装新增下载 0 字节。 |
| Tube | 16,709,077,850 | 08:35:15 至 09:07:45 | `hf-mirror.com` → 境外 `cas-bridge.xethub.hf.co`。观测到 WLAN，VPN 接口计数无增量。 | 完整镜像流程有回执，首包前保留分块总量未核实。 |
| Dice | 16,681,648,758 | 09:07:45 至 09:46:10 | 同上。 | 同上。 |
| Holder | 17,471,748,248 | 09:46:10 至 10:22:55 | 同上。 | 同上。 |

六档完整载荷合计 **97,082,881,045 字节**。停止时保留的首轮三类落盘载荷合计 **46,220,406,189 字节**。镜像阶段三类归档载荷合计 **50,862,474,856 字节**。旧日志记录有重试、探测和中断，但缺少逐请求完整字节计数。六类各自的真实新增传输量、重复传输总量和 VPN 供应商账单字节均未核实。Windows 接口计数只描述观测窗口内的本机行为。ModelScope 探测为 404，HF-Mirror 的三个最终数据端点均为境外 CDN。[停止前载荷与探测](../results/vicad_refinement/evidence/data-transfer.json) · [原镜像与路由观测](../results/vicad_refinement/evidence/mirror-probe.json) · [镜像守卫日志](../results/vicad_refinement/evidence/mirror-download.log)

完整性回执包括 [Bird](../results/vicad_refinement/evidence/Bird.json)、[Motor](../results/vicad_refinement/evidence/Motor.json)、[Teapot](../results/vicad_refinement/evidence/Teapot.json)、[Tube](../results/vicad_refinement/evidence/Tube.json)、[Dice](../results/vicad_refinement/evidence/Dice.json)、[Holder](../results/vicad_refinement/evidence/Holder.json)。原始暂停记录保存在本机 `D:/vicad-syn/task-004/download-paused.json`。时间根据原执行与镜像守卫日志整理，约略值和代理可见性缺口已注明。

**事前审批未能核实（六类）。** 没有找到逐文件和累计预算的真实事前授权，也没有找到单独放行境外 CDN 的记录。项目负责人在反馈 VPN 流量后给出的国内镜像原则和继续指令属于已知历史操作背景。历史审批索引为空。请项目负责人单独决定是否接受现存数据和实验继续保留，以及如何处理未核实审批的事实。该决定属于事后处置。
