# 验收记录

| 日期 | 交付 | 自检范围 | 验收状态 |
| --- | --- | --- | --- |
| 2026-11-19 | VICAD-Syn 文档初始化 | 文件清单、Markdown 内容与相对链接、Git 差异空白检查 | 待项目负责人验收 |
| 2026-11-19 | TASK-001 初审 | 十章节和 12 项核查基本合格。审查要求补齐脚本、复跑证据、逐类计数和状态更新。[初审意见](https://github.com/xinnan369/anomaly/pull/1#pullrequestreview-5969795350) | CHANGES_REQUESTED |
| 2026-11-19 | TASK-001 初审整改 | [整改报告](../reports/TASK-001-REPORT.md)及[复跑日志](../reports/evidence/TASK-001/test-log.txt)已提交至[原 PR #1](https://github.com/xinnan369/anomaly/pull/1)，待独立复查。 | CHANGES REQUESTED - AWAITING RE-REVIEW |
| 2026-11-19 | TASK-001 最终验收 | 项目负责人正式验收通过。PR #1 已合入 `main`，合并提交为 `d2da56c11fef084f47b90b201b0f905ec8e29c3e`。未解决事项继续记录于审计报告第 9 节。[审计报告](../reports/TASK-001-REPORT.md) · [PR #1](https://github.com/xinnan369/anomaly/pull/1) · [合并提交](https://github.com/xinnan369/anomaly/commit/d2da56c11fef084f47b90b201b0f905ec8e29c3e) | ACCEPTED |
| 2026-11-19 | TASK-002 第一阶段 | WSL2 环境固化、RTX 3090 CUDA 环境、Car/Dinomaly 单步训练、留出正常实例推理及 ADEval CUDA 合成测试。验收提交 `90fe550c28d00191ac67c0739d6097b6472fdc4b`，Review ID `5970938786`。[审查记录](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-5970938786) · [执行报告与待办](../reports/TASK-002-REPORT.md) | TASK-002 第一阶段：独立验收通过（APPROVED）。任务整体尚未完成，PR #2 保持开放。 |
| 2026-11-19 | TASK-002 第二阶段执行记录 | 空环境重装、RTX 3090 两种基线单步更新和留出正常实例推理、ADEval CUDA 合成测试，以及固定发布清单复核。[第二阶段记录](../reports/TASK-002-REPORT.md#第二阶段执行记录) · [PR #2](https://github.com/xinnan369/anomaly/pull/2) | PARTIAL。正式实验及可视化待完成，尚未申请第二阶段独立验收。 |
| 2026-11-19 | TASK-002 第二阶段审查及整改 | Review `5974527521` 要求修复物件身份和重复输出分组，验证冻结检查点，完成 Car/Dinomaly 闭环后再推进七组。身份和检查点回归通过，正式训练已启动。[审查](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-5974527521) · [整改证据](../reports/TASK-002-REPORT.md#第二阶段审查整改与正式实验) | CHANGES_REQUESTED。整改交付仍为 PARTIAL，等待完整实验证据。 |
| 2026-11-22 | TASK-002 第二阶段正式结果交付 | 四类归档凭据、8 组 100 epoch 与冻结检查点评价、指标和诊断、Doll 恢复记录、图表及发布材料已自检。[最终报告](../reports/TASK-002-REPORT.md#第二阶段最终结果) · [PR #2](https://github.com/xinnan369/anomaly/pull/2) | AWAITING RE-REVIEW。第二阶段实验交付完成，独立验收待复查。 |
| 2026-11-22 | TASK-002 最终整改审查 | [Review 6009448364](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-6009448364) 对提交 `f224a27e8020c0d98100acd082196846f1cc761b` 要求 Dinomaly/Doll 新连续运行、冻结复评和八组验证。 | CHANGES_REQUESTED。 |
| 2026-11-22 | TASK-002 最终整改交付 | 7/8 组连续训练和冻结评价有通过记录，最终整改存在阻塞。[报告](../reports/TASK-002-REPORT.md#第二阶段最终结果) · [核验 JSON](../results/baselines/metrics/verification.json) · [PR #2](https://github.com/xinnan369/anomaly/pull/2) | PARTIAL。 |
| 2026-11-22 | TASK-002 最终研究审查 | [Review 6011031984](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-6011031984) 接受七组和故障证据作为阶段成果，要求验证零分母来源并单独运行稳定性变体。 | CHANGES_REQUESTED。 |
| 2026-11-23 | TASK-002 最终研究交付 | 七组原版已复核，原版 Dinomaly/Doll 保留故障结论，稳定版完成冻结评价及复评。[报告](../reports/TASK-002-REPORT.md#第二阶段最终结果) · [PR #2](https://github.com/xinnan369/anomaly/pull/2) | 实验完成，待独立验收。 |
| 2026-11-23 | TASK-002 最终验收与合并 | 七组 OFFICIAL_CLEAN、Dinomaly/Doll 原版 OFFICIAL_FAILED、Dinomaly-stabilized NUMERICALLY_STABILIZED、历史 RECOVERED_DEVIATION。普通 merge 与当时 main 为 d1ddbb9d698114ccb616ad96c23565f7f2b1e275。[Review 6018691109](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-6018691109) · [合并提交](https://github.com/xinnan369/anomaly/commit/d1ddbb9d698114ccb616ad96c23565f7f2b1e275) | ACCEPTED_WITH_EXCEPTION / 有限例外验收通过。 |
| 2026-11-23 | TASK-003 执行交付 | 4/4 类正式结果通过，Car 核心消融 A0–A4 完成。[执行报告](../reports/TASK-003-REPORT.md) · [独立核验材料](../results/vicad_syn/evidence/verification.json) | 实验完成，待独立验收。 |

执行证据见[报告目录](../reports/README.md)。后续记录验收日期、任务 ID、报告与提交链接、验收结论和待修正项。
