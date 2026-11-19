# TASK-002 第二阶段基线证据

本目录保存已复验的环境、发布元数据、入口预检和正式运行状态。阶段二状态为 **PARTIAL**。正式结果表中的空白指标表示尚未测量。[第二阶段报告](../../reports/TASK-002-REPORT.md#第二阶段执行记录)

## 环境和入口预检

- [运行环境和安装检查](environment/runtime.json) 与 [87 项实装依赖](environment/freeze.txt)
- [两种模型单步复验、ADEval CUDA 和合成回归](environment/smoke-and-evaluator.json)
- [正式入口八组预检、官方配置及缺档拦截](environment/preflight.json)

## 数据审计

- [固定发布清单统计](data_audit/manifest-summary.json) 与 [四类远端 ZIP 中央目录](data_audit/remote-index.json)
- [完整归档下载快照](data_audit/download-snapshot.json)

## 正式实验

- [八组状态、资源与指标总表](metrics/baseline-summary.csv)
- 完整数据、权重、训练日志及原始预测数组保存在仓库外。正式运行产生的单模型输出位于外部目录 `runs/dinomaly/` 与 `runs/inpformer/`。图表在正式评价后生成。[下载、训练与恢复命令](../../tools/environment/README.md#第二阶段完整基线)
