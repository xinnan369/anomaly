# 文件获取与审批

默认使用已有缓存、归档、分块、权重及 wheel。禁止直接用线上安装器、`pip`、`curl`、`git lfs pull`、Hugging Face 工具、原始下载命令或会触发模型自动拉取的入口来绕过审批。历史实验的命令记录保留供追溯。依赖的传递依赖必须按文件列入同一任务计划。安装时仅使用已验证的本地 wheelhouse，例如 `python -m pip --isolated --disable-pip-version-check install --no-index --find-links <本地wheelhouse> -r tools/environment/requirements-wsl2.lock`。先确认锁文件及包的 URL 引用均不会触发在线访问。[强制下载规则](../AGENTS.md)

## 离线计划

`python -B -m tools.download_policy <plan.json>` 默认仅检查本地文件和分块并输出字节预算，无网络访问。使用固定 `TASK-NNN` 任务 ID，将同一任务的全部依赖、归档与权重列入同一清单。计划内逐文件包括 `name`、`version`、`size_bytes`、`sha256`、`mirror`、`redirect_chain`、`route_status`、`purpose`、`destination`、`file_budget_bytes`，另有累计 `task_budget_bytes`。最终端点未知时标记 `unverified`，不得凭入口推定最终地区。未知大小先报告，已知大小的批准计划须保留原始和剩余字节预算。先检查本地再向负责人提交审批；无网络授权时，连小范围探测也不执行。

`prepare_release.py` 与 `stage_refinement.py` 默认只打印草案。`--stage-local` 验证现有回执及本地树，或从本地完整归档或分块组装，校验 SHA-256、CRC、发布清单和实例隔离。`prepare_car.py` 从本地已解压数据准备 Car 样例。`tools/audit/` 只读固定的本地资源，缺项时报错。

## 身份与预算

有真实批准时，负责人须独立配置 Windows `~/.codex/vicad-download/owner.allowed_signers`（OpenSSH allowed signers，身份 `vicad-project-owner`），用其自行持有的密钥对执行计划的**原始字节**生成 SSH 签名（命名空间 `vicad-download-v1`）。计划须写入 `owner_approval_record`，指向真实的决定记录。凡任一入口或重定向非国内来源，相关文件还须有不同的 `owner_foreign_exception_record`，指向另行作出的明确授权。端点以无查询参数的完整 HTTPS 主机及路径逐跳绑定，须填写 `route_status: verified` 和获准的物理网卡 `direct_interface_index`。仅在项目负责人允许此次传输时，使用 Windows 原生 Python 执行 `python -B -m tools.download_policy <已签计划.json> --execute --signature <负责人签名文件>`。审批密钥、签名和信任文件不得提交。

程序在每次请求前核对 DNS 候选地址的实际 Windows 出口是否仍为签名批准的物理网卡，禁用请求级代理及自动重定向，拒绝白名单外的未获准来源。受控入口锁定每任务累计预算，按实际文件摘要记录单文件预算。发送分段请求前持久化保守预算预留；重启、失败、重试均保留预留，不自动重试。账本同时记录 UTC、预留范围、已完成分块和文件核验。预算预留表示最坏情况下的请求载荷，网络账单另行核对。校验严格的 Range 区间与文件 SHA-256，ZIP 和 wheel 另校验 CRC。WSL 只做离线准备，原生 Windows 维护单一账本。该控制覆盖仓库的脚本入口与运行规范，操作系统的独立网络程序另由工作站管理。负责人维护信任文件与账本。本轮未配置审批信任、未做真实传输验证。[本轮离线计划](../results/vicad_refinement/evidence/rectification-local-plan.json)
