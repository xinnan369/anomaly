# TASK-002 基线与稳定性证据

**ACCEPTED_WITH_EXCEPTION / 有限例外验收通过。** 七组原版已复核，原版 Dinomaly/Doll 保留故障结论，稳定版完成冻结评价及复评。[最终报告](../../reports/TASK-002-REPORT.md#第二阶段最终结果)

## 正式结果

- 原版表含 7 个 OFFICIAL_CLEAN 与 1 个 OFFICIAL_FAILED。Dinomaly/Doll 原版指标留空。[原版 CSV](metrics/baseline-summary.csv) · [类别比较及资源](metrics/model-comparison.csv)
- NUMERICALLY_STABILIZED 单独记录。旧恢复版保留为 RECOVERED_DEVIATION。[稳定版 CSV](metrics/stabilized-doll.csv) · [恢复版 CSV](metrics/doll-recovered-history.csv)
- 论文图分开显示原版和变体，所有分数来自固定的单种子协议。[原版指标 PNG](figures/official-only-metrics.png) / [PDF](figures/official-only-metrics.pdf) · [Doll 稳定版与原版 INP-Former PNG](figures/doll-variant-metrics.png) / [PDF](figures/doll-variant-metrics.pdf) · [稳定版诊断](metrics/stabilized-doll-diagnostics.csv)

## 正式证据

- [原版舍入与有界回放](evidence/doll-zero-origin.json) · [稳定版运行与文件哈希](evidence/doll-stability-run.json) · [日志节选](evidence/doll-stability.log) · [CUDA 融合身份回归](environment/stability-identity-regression.json)
- [正式运行和预测数组哈希](metrics/run-index.json) · [逐组核验](metrics/verification.json) · [四类已审计归档](data_audit/verified-archives.json)
- [原版失败现场](evidence/doll-clean.json) · [旧恢复证据](evidence/doll-recovery.json) · [运行命令](../../tools/environment/README.md#dinomaly-stabilized-数值稳定性实验)

## 旧图表和恢复版快照

前次混有恢复版的图表和诊断留作溯源。论文引用使用上方正式结果。[历史图表与诊断固定快照](https://github.com/xinnan369/anomaly/tree/00dd7f4f01a598337c9650203b3d8b28d8fd850e/results/baselines) · [旧指标图](figures/official-metrics.png) · [旧资源图](figures/training-resources.png) · [旧视角图](figures/view-diagnostics.png) · [旧光照图](figures/illumination-diagnostics.png) · [旧热力图](figures/reference-heatmaps.png)

## 环境和入口预检

- [运行环境和安装检查](environment/runtime.json) · [87 项实装依赖](environment/freeze.txt) · [两种模型单步与 ADEval CUDA](environment/smoke-and-evaluator.json) · [合成身份回归](environment/identity-regression.json) · [冻结保护测试](environment/checkpoint-validation.json)
- [正式入口八组预检、官方配置及缺档拦截](environment/preflight.json)

## 历史快照

[首次下载快照](data_audit/download-snapshot.json) · [续传过程](data_audit/restored-archives.json) · [远端 ZIP 中央目录](data_audit/remote-index.json) · [预检](environment/preflight.json) · [正式运行早期快照](metrics/formal-execution.json)
