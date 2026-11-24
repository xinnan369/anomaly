# TASK-005 结果索引

**PARTIAL。** 仅显示已经通过评价或逐阶段核验的记录。状态和缺失值都随实际运行更新。原始数据、检查点、特征缓存及数组保存在仓库外，原实验目录保持只读。

- [执行报告与恢复入口](../../reports/TASK-005-REPORT.md) · [预登记协议](../../instructions/TASK-005.md)
- [本地资产审计](evidence/resource-audit.json) · [既有证据与本轮自检](evidence/verification.md)
- [十二组官方基线状态](tables/official-progress.csv) · [十类可用本机指标](tables/local-results.csv) · [分覆盖范围宏平均](tables/coverage-macro.csv)
- [B0–B9 与扩展种子矩阵](tables/ablation.csv) · [逐类配对种子差值](tables/paired-seed-summary.csv)
- [作者公开基线，分列参考](../vicad_refinement/metrics/published-baseline-reference.csv) · [TASK-002 本机基线](../baselines/README.md)
- [同覆盖范围的本机基线比较](tables/same-coverage-baseline.csv) · [逐指标提升和下降计数](tables/paired-change-counts.csv)
- [历史 B6/B0 物件级区间](tables/historical-paired-ci.csv) · [配对区间图](figures/historical-paired-bootstrap.pdf) · [三种子稳定性图](figures/historical-seed-stability.pdf)
- [Bird 缓存／端到端时延](evidence/bird-runtime.json) · [Bird 时延图](figures/bird-runtime.pdf)
- [模型结构图与原研究材料](../vicad_refinement/README.md)

TASK-004 的历史预留类别成绩仍沿用其确认性定义。新增消融、区间与部署时延均作为补充或探索性证据。B0 为 TASK-004 使用同一检查点的配对 v1。TASK-003 独立训练结果见原目录。[TASK-003](../vicad_syn/README.md)
