# TASK-004 · VICAD-Syn 失效机制、正常开发与验证

[执行报告](../../reports/TASK-004-REPORT.md) · [预登记](../../instructions/TASK-004.md) · [TASK-003 已验收结果](../vicad_syn/README.md)

## 已见类别诊断

- 原有四类 A1/A4 的 Synergy 复评与正式历史结果逐值一致。[复评分数](metrics/v1-replay.csv) · [复评凭据](evidence/replay.json)
- 缺陷尺寸、掩膜跨光照交集、局部残差及不确定性排序、全局加权影响。[定位诊断](metrics/v1-localization.csv) · [定位机制 PDF](figures/v1-localization-mechanisms.pdf) · [Car 空间分辨率 PDF](figures/car-v1-patch-resolution.pdf)

## 正常样本开发

- 全部 24 项配置、主体隔离和预登记条件。[候选总表](metrics/development-matrix.csv) · [逐类别与情景](metrics/development-cells.csv) · [开发凭据](evidence/development.json) · [冻结凭据](evidence/freeze.json)
- 同坐标结构扰动对照、方向性污染、各特征深度及信号。[扰动幅度](metrics/contamination-summary.csv) · [方向投影](metrics/normal-direction-summary.csv) · [特征深度](metrics/normal-layer-summary.csv)
- [结构图 PDF](figures/method-structure.pdf) · [开发与污染 PDF](figures/normal-development-and-contamination.pdf) · [分层残差与可靠性 PDF](figures/car-synthetic-feature-and-reliability.pdf) · [共享特征投影 PDF](figures/car-feature-prediction.pdf)




## 冻结正式评价

**整改完成，待独立复验。** 已见类别 4/4，独立确认 6/6。

[逐种子指标与成本](metrics/formal.csv) · [类别摘要](metrics/formal-summary.csv)

[与已验收基线对照](metrics/accepted-baseline-comparison.csv) · [探索性图 PDF](figures/exploratory-official-comparison.pdf)

[预留类别图 PDF](figures/confirmatory-by-seed.pdf)

## 复现与文件核验

- [历史镜像入口、境外最终端点和路由观测](evidence/mirror-probe.json)

- [实验工件和日志哈希](evidence/external-artifacts.json) · [验证清单](evidence/verification.json) · [冻结时间及提交](evidence/freeze.json) · [官方归档传输](evidence/data-transfer.json)
- [CPU/CUDA 实测](evidence/multilayer-smoke.json) · [测试日志](evidence/test-log.txt) · [官方融合身份回归](evidence/identity.json) · [运行环境](evidence/environment-runtime.json)
- 完整预测数组、原始图像、缓存和权重保持在仓库外。训练透过正常样本拟合，合成指标与已见测试集的探索性指标都独立于确认结果。

## 审查后离线证据

- [44 条官方指标复评](metrics/official-replay.csv) · [20 组重新汇总](metrics/formal-recomputed-summary.csv) · [冻结时间及 18 个配对检查点](evidence/formal-audit.json)
- [六类配对逐种子差值](metrics/confirmatory-paired-seeds.csv) · [六类等权宏平均](metrics/confirmatory-macro.csv)
- [作者公开参考](metrics/published-baseline-reference.csv) · [本机已验收基线对照](metrics/accepted-baseline-comparison.csv)。作者参考用于描述性对照。预留六类没有本机原版对照训练。
- [本轮离线计划](evidence/rectification-local-plan.json) · [六类历史事实与审批缺口](../../reports/TASK-004-DOWNLOAD-AUDIT.md)
