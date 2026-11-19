# TASK-002 第二阶段基线证据

本目录保存已复验的环境、发布元数据、入口预检和正式运行状态。阶段二状态为 **PARTIAL**。正式结果表中的空白指标表示当前未取得有效正式评价。[整改与实测进度](../../reports/TASK-002-REPORT.md#第二阶段审查整改与正式实验)

## 环境和入口预检

- [运行环境和安装检查](environment/runtime.json) 与 [87 项实装依赖](environment/freeze.txt)
- [两种模型单步复验、ADEval CUDA 和合成回归](environment/smoke-and-evaluator.json)
- [正式入口八组预检、官方配置及缺档拦截](environment/preflight.json)
- [固定发布的实例编号与合成身份回归](environment/identity-regression.json)
- [检查点路径、实际中途元数据及冻结保护测试](environment/checkpoint-validation.json)

## 数据审计

- [固定发布清单统计](data_audit/manifest-summary.json) 与 [四类远端 ZIP 中央目录](data_audit/remote-index.json)
- [首次完整归档下载快照](data_audit/download-snapshot.json)与[续传和完整性复验证据](data_audit/restored-archives.json)

## 正式实验

- [八组状态、资源与指标总表](metrics/baseline-summary.csv)
- [正式运行快照与完整日志位置](metrics/formal-execution.json)及 [Car/Dinomaly 训练节选](metrics/car-dinomaly-train-excerpt.log)
- 完整数据、权重、原始日志与预测数组保存在仓库外。训练和评价分别写到 `runs/<模型>/<类别>/` 和 `evaluations/<模型>/<类别>/`。后处理日志为外部数据根的 `logs/finalize-stage2.log`，只在八组完整通过时生成正式图表、资源对照表、视角和光照诊断及实际热力图。[下载、训练与恢复命令](../../tools/environment/README.md#第二阶段完整基线)
