# 当前任务

[TASK-001](TASK-001.md) 已完成，项目负责人正式验收通过（**ACCEPTED**，2026-11-19）。[审计报告](../reports/TASK-001-REPORT.md) · [验收记录](../docs/ACCEPTANCE_LOG.md)

当前任务仍为 **TASK-002 · M²AD 官方基线复现与实验环境固化**，整体尚未完成。

**TASK-002 第一阶段：独立验收通过（APPROVED）。** 验收范围为 WSL2 环境固化、RTX 3090 CUDA 环境、Car/Dinomaly 单步训练、留出正常实例推理及 ADEval CUDA 合成测试。[执行报告](../reports/TASK-002-REPORT.md) · [审查记录](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-5970938786)

**TASK-002 第二阶段：PARTIAL。** 已在全新 WSL2 Python 3.11.11 环境按 87 项锁文件重装，并在 RTX 3090 实测两种基线的单步更新、留出训练正常实例推理和 ADEval CUDA 合成测试。四类档案完整性、八组正式训练与评估、结果分析和图表待完成。RTX 4090 未实测。继续使用开放的 PR #2，满足第二阶段验收标准后再提交独立验收。[第二阶段记录](../reports/TASK-002-REPORT.md#第二阶段执行记录) · [恢复命令](../tools/environment/README.md#第二阶段完整基线) · [PR #2](https://github.com/xinnan369/anomaly/pull/2)
