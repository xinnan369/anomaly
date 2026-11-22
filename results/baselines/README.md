# TASK-002 第二阶段基线证据

**TASK-002 最终整改 PARTIAL。** 7/8 组连续训练和冻结评价有通过记录，最终整改存在阻塞。[最终报告](../../reports/TASK-002-REPORT.md#第二阶段最终结果)

## 指标与前次图表

- [逐组指标与资源总表](metrics/baseline-summary.csv) · [类别对比](metrics/model-comparison.csv)（前次快照）
- 前次含恢复版的诊断与图表材料 · [视角和光照诊断](metrics/diagnostics.csv) · [正常件逐光照分数](metrics/normal-scores-by-light.csv) · [热力图样本索引](metrics/heatmap-inputs.csv)
- 前次图表（PNG / PDF） · [官方指标](figures/official-metrics.png) / [PDF](figures/official-metrics.pdf)、[训练资源](figures/training-resources.png) / [PDF](figures/training-resources.pdf)、[视角](figures/view-diagnostics.png) / [PDF](figures/view-diagnostics.pdf)、[光照](figures/illumination-diagnostics.png) / [PDF](figures/illumination-diagnostics.pdf)、[热力图](figures/reference-heatmaps.png) / [PDF](figures/reference-heatmaps.pdf)

## 正式证据

- [八组机器可读核验](metrics/verification.json) · [Doll 连续训练与复评证据](evidence/doll-clean.json) · [日志节选](evidence/doll-clean.log) · [注意力分母诊断](evidence/doll-forensic-replay.json)
- [Doll 恢复版历史指标](metrics/doll-recovered-history.csv)为 RECOVERED_DEVIATION。原失败、恢复和已有正式评价的文件路径及哈希见索引。新 Doll 无正式评价文件。
- [官方融合身份回归](environment/final-identity-regression.json) · [单元测试及八组校验日志节选](evidence/finalization.log)

- [八组运行、配置、检查点和数组哈希索引](metrics/run-index.json)。完整日志、权重、数据、原始预测数组及 GPU 采样保存在外部的数据根 `D:/vicad-syn/task-002-stage2`。该索引给出各文件的相对路径与 SHA-256。
- Dinomaly 日志节选 · [Car](evidence/dinomaly-Car.log)、[Cube](evidence/dinomaly-Cube.log)、[Doll 历史恢复](evidence/dinomaly-Doll.log)、[Ring](evidence/dinomaly-Ring.log)
- INP-Former 日志节选 · [Car](evidence/inpformer-Car.log)、[Cube](evidence/inpformer-Cube.log)、[Doll](evidence/inpformer-Doll.log)、[Ring](evidence/inpformer-Ring.log)
- [Doll 原失败和续跑凭据](evidence/doll-recovery.json) · [实际恢复脚本](evidence/doll-recovery-script.py.txt)
- [四类完整归档凭据](data_audit/verified-archives.json) · [固定发布清单统计](data_audit/manifest-summary.json)
- [复现与恢复命令](../../tools/environment/README.md#第二阶段完整基线)

## 环境和入口预检

- [运行环境和安装检查](environment/runtime.json) · [87 项实装依赖](environment/freeze.txt) · [两种模型单步与 ADEval CUDA](environment/smoke-and-evaluator.json) · [合成身份回归](environment/identity-regression.json) · [冻结保护测试](environment/checkpoint-validation.json)
- [正式入口八组预检、官方配置及缺档拦截](environment/preflight.json)

## 历史快照

[首次下载快照](data_audit/download-snapshot.json) · [续传过程](data_audit/restored-archives.json) · [远端 ZIP 中央目录](data_audit/remote-index.json) · [预检](environment/preflight.json) · [正式运行早期快照](metrics/formal-execution.json)
