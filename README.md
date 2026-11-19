# VICAD-Syn

VICAD-Syn 研究多视角、多光照条件下的工业视觉异常检测。项目拟使用 M²AD 数据集，重点审计并研究 M²AD-Synergy 协议。Car、Cube、Doll、Ring 是第一阶段候选类别，类别名称、可用样本与数据设置以官方发布为准。

基础模型候选为 DINOv2 ViT-B/14。光照条件化特征预测、双上下文预测稳定性、预测不确定性建模和可靠性加权异常融合均为**待验证的研究假设**。计划硬件为 NVIDIA RTX 3090 或 RTX 4090（均为 24 GB 显存），预计研究周期为 2～3 个月，投稿目标为 SCI 二区优先、三区备选。

## 当前状态

- **TASK-001 数据集与实验协议审计已完成并通过项目负责人验收（ACCEPTED）。** [审计报告](reports/TASK-001-REPORT.md) · [PR #1](https://github.com/xinnan369/anomaly/pull/1)
- **TASK-002 第一阶段已独立验收（APPROVED），第二阶段部分完成（PARTIAL）。** WSL2 空环境重装和 RTX 3090 上两种官方基线的单步复验已通过。四类别完整数据校验与八组正式实验仍待完成，RTX 4090 未实测。[执行报告](reports/TASK-002-REPORT.md#第二阶段执行记录) · [PR #2](https://github.com/xinnan369/anomaly/pull/2)

## 项目文档

- [协作规则](AGENTS.md)
- [当前任务](instructions/CURRENT_TASK.md)
- [项目章程](docs/PROJECT_CHARTER.md)
- [研究进度](docs/RESEARCH_PROGRESS.md)
- [研究决策](docs/RESEARCH_DECISIONS.md)
- [验收记录](docs/ACCEPTANCE_LOG.md)
- [执行报告目录](reports/README.md)
