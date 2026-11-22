# 当前任务

[TASK-001](TASK-001.md) 已完成，项目负责人正式验收通过（**ACCEPTED**，2026-11-19）。[审计报告](../reports/TASK-001-REPORT.md) · [验收记录](../docs/ACCEPTANCE_LOG.md)

当前任务仍为 **TASK-002 · M²AD 官方基线复现与实验环境固化**，整体尚未完成。

**TASK-002 第一阶段：独立验收通过（APPROVED）。** 验收范围为 WSL2 环境固化、RTX 3090 CUDA 环境、Car/Dinomaly 单步训练、留出正常实例推理及 ADEval CUDA 合成测试。[执行报告](../reports/TASK-002-REPORT.md) · [审查记录](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-5970938786)

**TASK-002 第二阶段：8/8 组正式训练与评价完成，等待独立复查。** RTX 3090 上的两个基线均完成 Car、Cube、Doll、Ring 各 100 epoch（30,000 步），冻结检查点评价与结果图表已交付。Dinomaly/Doll 从第 37 epoch 恢复。RTX 4090 未实测。当前仍为 TASK-002，待项目负责人确认验收。[最终报告](../reports/TASK-002-REPORT.md#第二阶段最终结果) · [证据索引](../results/baselines/README.md) · [PR #2](https://github.com/xinnan369/anomaly/pull/2)
