# VICAD-Syn

VICAD-Syn 研究多视角、多光照条件下的工业视觉异常检测。项目拟使用 M²AD 数据集，重点审计并研究 M²AD-Synergy 协议。Car、Cube、Doll、Ring 是第一阶段候选类别，类别名称、可用样本与数据设置以官方发布为准。

基础模型候选为 DINOv2 ViT-B/14。光照条件化特征预测、双上下文预测稳定性、预测不确定性建模和可靠性加权异常融合均为**待验证的研究假设**。计划硬件为 NVIDIA RTX 3090 或 RTX 4090（均为 24 GB 显存），预计研究周期为 2～3 个月，投稿目标为 SCI 二区优先、三区备选。

## 当前状态

- **TASK-001 数据集与实验协议审计已完成并通过项目负责人验收（ACCEPTED）。** [审计报告](reports/TASK-001-REPORT.md) · [PR #1](https://github.com/xinnan369/anomaly/pull/1)
- **TASK-002 已有限例外验收（ACCEPTED_WITH_EXCEPTION）。** 七组原版通过，Dinomaly/Doll 原版故障和独立稳定版分别留存。RTX 4090 未实测。[报告](reports/TASK-002-REPORT.md) · [合并](https://github.com/xinnan369/anomaly/commit/d1ddbb9d698114ccb616ad96c23565f7f2b1e275)
- **TASK-003 已验收（ACCEPTED）。** VICAD-Syn v1 为可复现的研究原型，四类结果均低于可用原版基线。[报告](reports/TASK-003-REPORT.md) · [结果与图表](results/vicad_syn/README.md)

- **TASK-004 待独立验收。** 完成失效诊断与正常开发，已见类别 4/4，独立确认 6/6。[执行报告](reports/TASK-004-REPORT.md)

## 项目文档

- [协作规则](AGENTS.md)
- [当前任务](instructions/CURRENT_TASK.md)
- [项目章程](docs/PROJECT_CHARTER.md)
- [研究进度](docs/RESEARCH_PROGRESS.md)
- [研究决策](docs/RESEARCH_DECISIONS.md)
- [验收记录](docs/ACCEPTANCE_LOG.md)
- [执行报告目录](reports/README.md)
