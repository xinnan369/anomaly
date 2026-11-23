# TASK-003 · VICAD-Syn 结果与证据

**实验完成，待独立验收。** [实验报告](../../reports/TASK-003-REPORT.md) · [固定协议和命令](../../instructions/TASK-003.md)

## 正式结果

- A0–A4 的原始指标和检查点绑定。[核心指标](metrics/summary.csv) · [A2/A3 追加指标](metrics/ablation-extension.csv) · [扩展核验](evidence/ablation-extension.json)
- 原版 Dinomaly、原版 INP-Former、Dinomaly-stabilized 与 VICAD-Syn 分列。Doll 原版 Dinomaly 的指标留空。[基线对照](metrics/baseline-comparison.csv)
- 四类的光照和视角统计来自完整正式评价。[条件诊断](metrics/condition-profiles.csv) · [定位诊断](metrics/localization.csv)

## 科研图表

- 原版、独立变体和 VICAD-Syn。[PDF](figures/baseline-comparison.pdf) · [PNG](figures/baseline-comparison.png)
- Car A0–A4。[PDF](figures/car-ablations.pdf) · [PNG](figures/car-ablations.png)
- Car 光照与视角诊断。[PDF](figures/car-condition-profiles.pdf) · [PNG](figures/car-condition-profiles.png)
- Car 特征、残差及可靠性图。[PDF](figures/car-feature-and-reliability.pdf) · [PNG](figures/car-feature-and-reliability.png)

## 核验与复现

- [核验与预测数组索引](evidence/verification.json) · [外部实验文件与日志哈希](evidence/external-artifacts.json)
- [CPU/CUDA 烟测](evidence/smoke.json) · [CUDA 官方融合身份回归](evidence/identity-synthetic.json) · [执行日志节选](evidence/pipeline-excerpt.log) · [最终测试](evidence/final-tests.log)
- [实装依赖](evidence/environment-freeze.txt) · [GPU 与驱动](evidence/gpu-environment.csv) · [扩展优先级封存](evidence/ablation-extension-plan.json) · [冻结时序](evidence/preregistration-timing.json)

- [预测与不确定性成本分解](evidence/uncertainty-cost.json)
- [可视化样本和权重绑定](evidence/car-figure-source.json)
