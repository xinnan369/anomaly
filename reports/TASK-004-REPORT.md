# TASK-004 · VICAD-Syn 失效诊断与验证

**状态　ACCEPTED_WITH_DOCUMENTED_HISTORICAL_EXCEPTION。** [最终验收](../docs/ACCEPTANCE_LOG.md) [任务与协议](../instructions/TASK-004.md) · [结果索引](../results/vicad_refinement/README.md)

TASK-003 通过独立验收。它提供可复现的研究原型和真实负结果。PR #4 由批准提交 `4982849e4907e22df1dd036adfdfdb9b4ac3e750` 普通合并到 `498d99652d3153f50a3113c7a8b363d088449c56`。[Review 6022362483](https://github.com/xinnan369/anomaly/pull/4#pullrequestreview-6022362483) · [合并提交](https://github.com/xinnan369/anomaly/commit/498d99652d3153f50a3113c7a8b363d088449c56)


## 探索性复查 · 四类已见样本

从已验收的 A1/A4 读取冻结预测、掩膜与身份，核对文件字节数及 SHA-256，以固定官方 Synergy 入口只读复评。四类八项的三个分数与已验收总表逐值一致。定位分析只在有非空官方缩放掩膜的视角和图像中进行。patch AUROC 在每类最多 10 万个阳性及等量阴性 patch 上以固定 seed 42 等量抽样，描述边缘分离能力。共享物件内的观测相关，未用于统计独立性推断。

| 类别 | 有掩膜视角（全部十光照可见） | 中位数掩膜像素 | 被覆盖 patch 的掩膜面积比例 | 残差 patch AUROC | 稳定性／认知 patch AUROC | A4 掩膜内响应下降视角 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Car | 209 | 15 | 5.40% | 0.503 | 0.331 / 0.192 | 75.60% |
| Cube | 325 | 9 | 5.44% | 0.748 | 0.630 / 0.625 | 67.38% |
| Doll | 391 | 64 | 12.51% | 0.632 | 0.463 / 0.227 | 51.41% |
| Ring | 312 | 2 | 1.23% | 0.728 | 0.547 / 0.529 | 60.26% |

Car A4 的掩膜内外均值为 0.134412 与 0.134098，差值仅 0.000315。A1 的差值为 0.001499，A4 进一步降低了约 52.63% 阳性视角的区域对比度。Car 的低 AUPRO 与残差几乎不具区分度、缺陷区域远小于一个 patch 及全局权重削弱掩膜响应一致。Ring 在占据率更低时仍能达到较高定位分数，分辨率解释存在类别交互。占据率衡量掩膜对 14×14 网格的充满程度，这项量不作为像素定位能力上限。第 12 块偏语义的说法作为假设，由固定正常参考评分上的深度对照检验。

1,237 个含掩膜视角的十张官方评价掩膜均逐像素完全相同，每个阳性视角的跨光照掩膜交并比为 1。九个源光照均具备相同位置异常的评价标注。方向性传播及其对残差的作用由下一节的合成对照估计。两类不确定性对部分类别具有信号，对 Car 的排序方向与残差的常规异常方向不一致。其 patch AUROC 是单变量诊断，独立增益通过同一预测器上的单信号与双信号消融核查。

## 机制与方法

冻结 ViT-B/14 reg4，取第 4、8、12 块的 LayerNorm 后输出，剔除前 5 个 CLS／register 位置，再对每个 768 维 patch 向量做 L2 归一化。三块均保留 16×16 网格。FP16 缓存每张图 1,179,648 字节，以 FP32 恢复与重新归一化。用同一输入真实样本在 CPU 和 RTX 3090 上比较，第 12 块输出与原版逐元素一致。

设归一化特征为 x，归一化向量预测为 p，r(x,p)=||x−p||²。A 取块 12 的原版学习预测。B 将 A 与块 4、8 各自对九个其他光照的归一化均值预测残差等权平均。C 对每个块使用训练正常件同视角、同光照、同 patch 的归一化均值原型，然后跨三块等权平均。D 为 (B+C)/2。同一光照预测不读取目标块。C 使用待测物件之外的原型。跨视角只沿用官方物件标量聚合。

局部可靠性在同一 patch 的十光照中，计算每种信号相对有效光照均值的非负比值，取负号，双信号相加，再作光照维 softmax。无有效光照时输出零。保护模式使用 w = 0.75·w_local + 0.25/N_valid。对非负残差，保护图在逐 patch 上至少保留有效等权评分的 25%。候选按 10w 缩放各光照的 patch 贡献，经上采样后进入官方平均融合与 CUDA ADEval。无修改的整图 v1 分支继续使用原版权重。等权融合在所有有限输入上保留简单平均口径。

合成扰动使用冻结位置生成规则，在一个、四个或全部光照上叠加同坐标局部逆序纹理并按强弱混合。每类为六个物理留出对象各两个视角。光照增益对照无结构掩膜。受影响光照在目标 patch 比较模型预测与干净时的输出移动。带方向的投影为 (Δ预测·Δ目标)/||Δ目标||²，正值表示输入扰动方向的一部分进入预测。正常条件原型在同一视角下固定。原版九光照、九光照逐通道中位数和四光照子集共用相同的目标掩膜。

## 正常开发与冻结

以 seed 42 在四类中各取 24 个训练正常件和 6 个留出正常件。对每个留出实例的 000°、180° 视角固定生成四个结构扰动与两个控制情景。像素 AP 等权汇总 192 个阳性视角情景。正常与光照控制记录超过同场景干净分数 P99 的比例，空 AP 保留为空值。本阶段未使用官方测试掩膜。

| 特征评分 | 等权 | v1 整图 | patch 双信号 | 局部保护 | 仅稳定性 | 仅认知 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_final | 26.11 | 26.09 | 27.68 | 27.57 | 28.44 | 26.28 |
| B_multilayer | 32.66 | 31.07 | 32.51 | 32.80 | 33.89 | 31.76 |
| C_normal | 58.15 | 58.01 | 58.73 | 58.66 | 58.75 | 58.35 |
| D_combined | 63.44 | 63.27 | 64.99 | 64.54 | 65.02 | 63.26 |

表内为宏平均像素 AP（%）。预登记规则选中 **D_combined / patch_stability**（65.02%）。原版 v1 A4 对应的代理值为 26.09%。加入正常参考的收益高于从等权改为局部稳定性加权的收益，这一排序来自正常图像合成条件。

冻结回执时间 `2026-11-23T14:06:50.847065+00:00`（UTC），代码提交 `91ec34558507cb6d9a23aeed6dc51ca51d21e50a`，回执 SHA-256 `41fb52634e61a5023cd8172c83d9982f075eedfc65a6cf807bdca8286b4e503f`。冻结前执行正常代理矩阵和 v1 的只读探索复查。新候选的 Synergy 评价由该回执约束。

[全部候选](../results/vicad_refinement/metrics/development-matrix.csv) · [类别与情景汇总](../results/vicad_refinement/metrics/development-cells.csv) · [冻结回执](../results/vicad_refinement/evidence/freeze.json)

## 合成污染与特征深度

| 条件 | 九光照学习预测 | 正常条件原型 | 九光照中位数 | 四光照子集 |
| --- | ---: | ---: | ---: | ---: |
| one_weak | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| four_strong | 0.2439 | 0.0000 | 0.1090 | 0.2824 |
| all_weak | 0.7840 | 0.0000 | 0.8051 | 0.7729 |
| all_strong | 0.8286 | 0.0000 | 0.8513 | 0.8180 |

单光照扰动中，四种条件的受扰目标均无预测变化，这与目标排除一致。四光照共同扰动时，中位数把九光照学习预测的带方向响应从 0.244 降至 0.109。所有光照共享扰动时，直接预测、中位数和子集均有较大正投影。强扰动时九光照为 0.829，正常原型为零。这支持共同结构沿预测传播的机制。实际 M²AD 缺陷的具体形态和照明响应未被合成干预全面覆盖。

单光照增益的阴性对照没有结构掩膜。相对同视角干净 P99，v1 和冻结候选的超阈比例均值分别为 1.020% 与 1.015%，与干净对照的约 1.000% 接近。该控制只覆盖登记的单光照增益，尚不能代表更大范围的照明变化。

| 类别 | 第 4 块 AP | 第 8 块 AP | 第 12 块 AP |
| --- | ---: | ---: | ---: |
| Car | 56.98% | 55.12% | 49.25% |
| Cube | 45.25% | 65.06% | 63.23% |
| Doll | 39.66% | 39.82% | 35.72% |
| Ring | 63.01% | 59.96% | 54.37% |

三块使用相同的正常原型评分和合成掩膜。Car 第 4 块比第 12 块高 7.73 个百分点。四类中均有一个中间块高于末块，最佳层随类别变化。这支持多层互补，尚无法把差异因果归于语义偏向。特征图使用共享 PCA 投影，颜色仅用于同一组三张特征图的相对对照。

[污染方向](../results/vicad_refinement/metrics/normal-direction-summary.csv) · [逐层定位](../results/vicad_refinement/metrics/normal-layer-summary.csv) · [机制图](../results/vicad_refinement/figures/normal-development-and-contamination.pdf)

## 数据下载与流量复核

本轮在 Windows 和 WSL 检查项目下载进程，未发现运行项。未启动下载、探测或新训练。本地原始发布数据、缓存、断点和既有科学结果保留。六类归档完整载荷合计 97,082,881,045 字节。历史传输的新增、重复和 VPN 供应商计费字节均不可从现存日志完整复原。Teapot 235 段全部来自已有本地缓存，离线组装新增下载为 0 字节。HF-Mirror 的三个数据端点为境外 CDN，保存的 WLAN 和 VPN 接口观测只描述本机网络行为。六类事前审批均未能核实。负责人随后同意保留已有数据和实验成果，继续保留事前审批未能核实的结论。[历史事实审计](TASK-004-DOWNLOAD-AUDIT.md) · [本轮零下载计划](../results/vicad_refinement/evidence/rectification-local-plan.json) · [审批入口使用说明](../tools/DOWNLOADS.md)

## 冻结后的正式结果与限制

### M²AD-Synergy 评价

指标依次为 O-AUROC / I-AUROC / AUPRO（%）。同一类别同一 seed 的新候选和 v1 A4 复跑共享叶层预测器检查点。已见类别也保留 v1 的已验收原始指标。evaluator 读取完整官方划分，校验物件和视角身份。

| 证据范围 | 类别 | Seed | v1 A4 配对复跑 | 冻结候选 |
| --- | --- | ---: | --- | --- |
| exploratory | Car | 42 | 53.70 / 49.83 / 13.20 | 65.10 / 61.62 / 72.12 |
| exploratory | Cube | 42 | 63.00 / 55.46 / 66.99 | 88.30 / 70.01 / 92.15 |
| exploratory | Doll | 42 | 60.00 / 49.06 / 35.56 | 50.90 / 55.93 / 74.13 |
| exploratory | Ring | 42 | 60.50 / 56.51 / 51.06 | 90.10 / 75.05 / 87.29 |
| confirmatory | Bird | 42 | 60.30 / 52.32 / 45.70 | 70.10 / 63.94 / 83.38 |
| confirmatory | Bird | 43 | 59.70 / 52.04 / 45.31 | 69.70 / 64.10 / 83.31 |
| confirmatory | Bird | 44 | 64.30 / 53.00 / 46.06 | 70.90 / 63.91 / 83.48 |
| confirmatory | Motor | 42 | 49.10 / 51.35 / 49.88 | 90.10 / 73.15 / 94.63 |
| confirmatory | Motor | 43 | 49.50 / 51.38 / 47.75 | 91.30 / 73.12 / 94.67 |
| confirmatory | Motor | 44 | 48.80 / 51.40 / 48.54 | 90.10 / 72.34 / 94.63 |
| confirmatory | Teapot | 42 | 81.00 / 59.81 / 19.89 | 90.70 / 78.48 / 90.17 |
| confirmatory | Teapot | 43 | 80.90 / 59.22 / 19.82 | 89.60 / 77.95 / 90.84 |
| confirmatory | Teapot | 44 | 80.70 / 59.87 / 20.80 | 90.00 / 78.51 / 90.72 |
| confirmatory | Tube | 42 | 64.40 / 50.38 / 15.97 | 83.60 / 63.14 / 65.99 |
| confirmatory | Tube | 43 | 67.30 / 51.18 / 16.27 | 84.40 / 63.06 / 66.10 |
| confirmatory | Tube | 44 | 62.60 / 50.64 / 16.83 | 83.30 / 62.83 / 66.37 |
| confirmatory | Dice | 42 | 41.80 / 50.40 / 47.28 | 98.30 / 80.52 / 79.24 |
| confirmatory | Dice | 43 | 42.00 / 50.39 / 47.83 | 98.00 / 81.01 / 79.09 |
| confirmatory | Dice | 44 | 41.50 / 50.48 / 48.47 | 98.20 / 81.24 / 79.33 |
| confirmatory | Holder | 42 | 79.49 / 58.23 / 20.64 | 97.14 / 70.63 / 82.93 |
| confirmatory | Holder | 43 | 78.47 / 57.32 / 21.44 | 97.14 / 70.61 / 82.93 |
| confirmatory | Holder | 44 | 78.57 / 57.75 / 21.49 | 97.04 / 70.97 / 82.99 |

[逐组指标与资源](../results/vicad_refinement/metrics/formal.csv) · [类别均值及样本标准差](../results/vicad_refinement/metrics/formal-summary.csv)

已见类别完成 4/4。与 TASK-002 已验收原版对照，Dinomaly/Doll 故障维持空指标，Dinomaly-stabilized 单列。历史恢复实验继续留在 TASK-002 档案。该段结果属于 exploratory。 [固定基线身份与差值](../results/vicad_refinement/metrics/accepted-baseline-comparison.csv) · [探索性结果图](../results/vicad_refinement/figures/exploratory-official-comparison.pdf)

| 已见类别 | 编码测试图 (s) | 拟合与原型 (s) | 配对预测 (s) | 拟合峰值 (GiB) | 预测峰值 (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Car | 781.9 | 169.1 | 304.2 | 5.60 | 1.35 |
| Cube | 685.9 | 142.7 | 325.4 | 5.60 | 1.35 |
| Doll | 708.7 | 159.0 | 304.0 | 5.60 | 1.35 |
| Ring | 719.4 | 145.0 | 268.4 | 5.60 | 1.35 |

冻结主干 86,583,552 个参数。叶层预测器 478,656 个可训练参数。每类正常参考原型 FP32 存储 270 MiB。表中编码计入图片读取、三个块的特征计算及缓存写入。预测计入两种评分共享的读盘、残差和信号计算。训练与预测峰值均为当前阶段的 PyTorch 分配峰值。RTX 4090 未实测。

## 独立确认和科学边界

独立确认覆盖 6/6 个此前未用于本项目模型选择的类别。各次冻结运行均使用 42、43、44 三个种子。冻结候选和配对 v1 仅改变评分及融合路径，按类别重新在同一发布的正常训练物件上拟合。该设计验证流程向未见类别的迁移，样本覆盖限于 M²AD。样本标准差反映同一测试集上跨训练种子的波动。

| 类别 | 方法 | O-AUROC 均值 ± SD | I-AUROC 均值 ± SD | AUPRO 均值 ± SD |
| --- | --- | ---: | ---: | ---: |
| Bird | selected | 70.23 ± 0.61 | 63.98 ± 0.10 | 83.39 ± 0.08 |
| Bird | v1_A4 | 61.43 ± 2.50 | 52.45 ± 0.49 | 45.69 ± 0.38 |
| Dice | selected | 98.17 ± 0.15 | 80.92 ± 0.37 | 79.22 ± 0.12 |
| Dice | v1_A4 | 41.77 ± 0.25 | 50.42 ± 0.05 | 47.86 ± 0.60 |
| Holder | selected | 97.11 ± 0.06 | 70.73 ± 0.20 | 82.95 ± 0.03 |
| Holder | v1_A4 | 78.84 ± 0.56 | 57.77 ± 0.46 | 21.19 ± 0.48 |
| Motor | selected | 90.50 ± 0.69 | 72.87 ± 0.46 | 94.64 ± 0.02 |
| Motor | v1_A4 | 49.13 ± 0.35 | 51.38 ± 0.02 | 48.72 ± 1.08 |
| Teapot | selected | 90.10 ± 0.56 | 78.32 ± 0.31 | 90.58 ± 0.36 |
| Teapot | v1_A4 | 80.87 ± 0.15 | 59.64 ± 0.36 | 20.17 ± 0.55 |
| Tube | selected | 83.77 ± 0.57 | 63.01 ± 0.16 | 66.15 ± 0.19 |
| Tube | v1_A4 | 64.77 ± 2.37 | 50.73 ± 0.41 | 16.35 ± 0.44 |

Bird 的三种子平均改进为 +8.80 / +11.53 / +37.70 个百分点。按 M²AD 预印本表 2 的 224 输入参考，Bird 的 Dinomaly 为 75.1 / 74.9 / 86.9，INP-Former 为 80.0 / 67.2 / 84.1。三项 Bird 平均结果仍低于这两个作者公布的基线。已发表参考单独标记，当前试验没有在确认类另行训练这两个原版基线。Holder 的预印本与固定发布有一件异常物件差异。[作者公开结果](https://arxiv.org/html/2505.10996v1#S3.T2) · [作者参考数据](../results/vicad_refinement/metrics/published-baseline-reference.csv)

已见四类 AUPRO 和 I-AUROC 均随新候选上升。Doll 的 O-AUROC 由 60.00% 降至 50.90%，说明像素定位增益没有自动转化为物件级判别。六个预留类支持把共同异常传播及正常参考评分作为论文的机制贡献。相对配对 v1 的提升在此发布内跨类一致。相对公开原版基线的优势有限，Dice 的 O-AUROC 领先最高参考 0.07 个百分点，Motor 的 AUPRO 领先 0.44 个百分点。这些差异适合描述性比较。Holder 的发布差异也限制与论文绝对分数的解释。研究方向具有机制型 SCI 论文价值，现有证据不足以支持普遍领先的检测方法主张。

[确认覆盖与归档状态](../results/vicad_refinement/metrics/confirmatory-coverage.csv) · [各类归档审计](../results/vicad_refinement/evidence/data-transfer.json)

当前确认范围中，三项均值分别在 6/6、6/6、6/6 类高于配对 v1。按作者公布两者中分数较高的一项对照，改进方法分别在 1/6、0/6、1/6 类超过该参考。全部类别和种子同时保留。冻结对比同时改变了评分表示和融合。局部权重的单独贡献由正常代理消融估计。公开对照缺少同一环境的逐种子基线，论文性能结论应逐指标限于当前发布及已跑范围。

### 独立确认资源

| 类别 | 测试编码 (s) | 平均拟合 (s) | 平均配对预测 (s) | 拟合最高峰值 (GiB) | 预测最高峰值 (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bird | 686.4 | 62.2 | 94.1 | 6.15 | 1.89 |
| Motor | 707.3 | 72.8 | 134.0 | 6.15 | 1.89 |
| Teapot | 737.8 | 86.8 | 155.4 | 6.15 | 1.89 |
| Tube | 734.8 | 82.4 | 100.8 | 6.15 | 1.89 |
| Dice | 712.7 | 70.9 | 118.3 | 5.97 | 1.89 |
| Holder | 698.5 | 73.0 | 103.8 | 6.15 | 1.89 |

测试特征缓存每类只计算一次。拟合及配对预测按三个种子平均，峰值取三个种子中的最高值。完整逐种子数据、评估耗时和哈希索引见正式 CSV。

## 环境、命令与自检

Windows 主机使用 Python 下载。拟合、诊断和评价使用 WSL2 Ubuntu 22.04、Python 3.11.11、PyTorch 2.2.2+cu121、CUDA 12.1、ADEval 1.1.0、RTX 3090。首次多层提取发现固定上游的接口只接受末层数量，工程烟测修复后，用同一固定原版 block 循环选取第 4、8、12 块。CPU/CUDA 再核验末层结果逐值相同及主干无梯度。首次运行没有形成代理或正式指标。官方 CUDA ADEval 的中间除法可出现空分母警告，实际报告指标均通过有限性检查。Windows 环境的报告导出因缺少 sklearn 未运行，使用锁定 WSL 环境完成导出。

运行核对覆盖新旧 56 项单元测试、官方融合身份回归、目标光照排除、训练与开发／测试物件隔离、逐组融合身份、冻结时序、检查点和预测关联、相对链接、Git 差异空白及敏感信息与超大文件。阶段命令按下列入口执行。

```bash
python -B -m vicad_refinement.run develop --inputs /mnt/d/vicad-syn/task-002-stage2 --output /mnt/d/vicad-syn/task-004/run-v2
python -B -m vicad_refinement.run freeze --inputs /mnt/d/vicad-syn/task-002-stage2 --output /mnt/d/vicad-syn/task-004/run-v2
python -B -m vicad_refinement.run explore --inputs /mnt/d/vicad-syn/task-002-stage2 --output /mnt/d/vicad-syn/task-004/run-v2 --category Car
python -B -m vicad_refinement.run confirm --inputs /mnt/d/vicad-syn/task-002-stage2 --output /mnt/d/vicad-syn/task-004/run-v2 --holdout /mnt/d/vicad-syn/task-004/holdout --category Bird
```

实际逐类探索命令依次替换 Car、Cube、Doll、Ring。使用 Windows Git worktree 时，为冻结命令传入映射至 WSL 的 GIT_DIR 与 GIT_WORK_TREE。原回执存在时从该点恢复。Bird 使用 D 盘归档。Motor、Teapot、Tube、Dice、Holder 的归档准备切换到 C 盘独立数据根，该选择保留 D 盘上的缓存空间与 40 GiB 的预留。以下 Tube 命令是旧阶段的历史记录，禁止直接执行。现行流程只从既有许可路径和签名计划进入。

```powershell
python -B -m tools.vicad_syn.stage_refinement --root C:/Users/xinnan/vicad-data/task-004-holdout --manifest-cache D:/vicad-syn/task-002-stage2/cache --category Tube --transport mirror --workers 12
```

数据版本见 [官方发布](https://huggingface.co/datasets/ChengYuQi99/M2AD/commit/05abc5d03167354a7ef98541236b5c0bd00abadf)、[论文](https://arxiv.org/html/2505.10996v1)及[固定上游](https://github.com/hustCYQ/M2AD/tree/a606aa0e1614f23dad70da26430dd66095192141)。完整实验位于 `D:/vicad-syn/task-004`。原版指标、原预测和模型保持原路径。 [核验清单](../results/vicad_refinement/evidence/verification.json) · [图表、证据和命令索引](../results/vicad_refinement/README.md) · [PR #5](https://github.com/xinnan369/anomaly/pull/5)

## Review 6033672223 整改与只读复评

正式 CSV 保留 44 行。按 scope、类别和方法重算 20 组均值及样本标准差，逐值与原摘要一致（容差 1e-10 个百分点）。四类探索性设定各只有一个种子，样本标准差留空。冻结回执为北京时间 2026-11-23 22:06:50，SHA-256 为 `41fb52634e61a5023cd8172c83d9982f075eedfc65a6cf807bdca8286b4e503f`。六类 18 个配对检查点均有早于该次测试的预检，且晚于冻结、绑定同一冻结摘要。每对中的 v1_A4 和选定方案使用同一检查点。使用固定版源代码和 CUDA ADEval 重新读取 44 套 SHA-256 匹配的已保存融合数组。三项指标逐项差值均为 0.0 个百分点。[正式复评](../results/vicad_refinement/metrics/official-replay.csv) · [汇总复算](../results/vicad_refinement/metrics/formal-recomputed-summary.csv) · [冻结与绑定核对](../results/vicad_refinement/evidence/formal-audit.json)

重读固定发布的本地清单。Dice 正常训练 29 件、3,480 张。Holder 测试 69 件（正常 20、异常 49）、8,280 张，对应 828 个融合视角。Holder 仍保留与公开论文的发布差异。跨划分的完整实例键、图像路径及类别加实例编号交集均为 0。裸编号的跨类别重复为 8。[类别覆盖](../results/vicad_refinement/metrics/confirmatory-coverage.csv)

| 指标 | 配对 v1 宏平均 ± SD（%） | 冻结改进版宏平均 ± SD（%） | 配对差 ± SD（百分点） | 六类升／降 |
|---|---:|---:|---:|---:|
| O-AUROC | 62.80 ± 0.16 | 88.31 ± 0.05 | +25.51 ± 0.13 | 6／0 |
| I-AUROC | 53.73 ± 0.14 | 71.64 ± 0.01 | +17.91 ± 0.14 | 6／0 |
| AUPRO | 33.33 ± 0.33 | 82.82 ± 0.10 | +49.49 ± 0.27 | 6／0 |

每个种子对六类等权，三个种子宏值再取均值及样本标准差。该 SD 描述训练种子波动。同一测试集重复评估的种子结果无法估算对其他工业域的泛化误差。已见四类中，O-AUROC 3 类升、1 类降（Doll），I-AUROC 与 AUPRO 均 4 类升、0 类降。[逐类逐种子差值](../results/vicad_refinement/metrics/confirmatory-paired-seeds.csv) · [逐种子宏值](../results/vicad_refinement/metrics/confirmatory-macro.csv)

- **机制证据。** 正常代理与可控结构扰动支持跨光照共享异常被预测器重建、导致预测残差减弱的解释。多层特征、正常原型和局部融合的单独效果只在正常代理实验中估计。正式确认同时改变这些因素，现有正式比较只估计整个组合的效果。
- **已验证的性能结论。** 在 M²AD 固定发布上，六个此前未用于选型的类别、全部三个种子均报告。冻结组合在三项六类别宏指标上均高于配对 v1，正式核查已复现。
- **描述性比较。** 作者公开参考与本机复现分列。确认类没有同环境的原版基线。按两条作者公开基线的较高项对照，O-AUROC、I-AUROC、AUPRO 分别 1／0／1 类领先，剩余 5／6／5 类落后。最多只支持小幅个别领先。已见四类本机对照保留原版失败与数值稳定变体身份。[作者参考](../results/vicad_refinement/metrics/published-baseline-reference.csv) · [本机对照](../results/vicad_refinement/metrics/accepted-baseline-comparison.csv)
- **论文建议。** 可形成围绕跨光照异常传播及局部保护机制的论文。若要把与原版基线的差异作为方法领先证据，需同环境原版基线及新的独立评估，优先复验预留类别及正式因子拆分。当前范围不足以支持 SOTA。

### 本轮环境与实际命令

WSL2 Ubuntu 22.04、锁定 Python 3.11.11、PyTorch 2.2.2+cu121、CUDA ADEval 1.1.0、RTX 3090。复评无新增种子；只重读探索种子 42 与确认种子 42、43、44 的历史结果。原协议、24 项正常代理、已见四类、六类确认原记录和 TASK-003 结果均保留。本轮镜像、签名、累计预算、重试、断点、重定向、路由的验证仅为模拟。Windows 路由应用、真实签名信任文件及任何下载均未启用。

```bash
python -B -m tools.download_policy results/vicad_refinement/evidence/rectification-local-plan.json
python -B tools/audit/manifest.py --cache /mnt/d/vicad-syn/task-002-stage2/cache
GIT_NO_LAZY_FETCH=1 python -B -m tools.vicad_syn.reaudit_refinement --run /mnt/d/vicad-syn/task-004/run-v2 --upstream /mnt/d/vicad-syn/task-002-stage2/inputs/upstream --formal results/vicad_refinement/metrics/formal.csv --output /mnt/d/vicad-syn/task-004/rectification/official-replay.csv
python -B -m tools.vicad_syn.summarize_refinement_audit --replay /mnt/d/vicad-syn/task-004/rectification/official-replay.csv --output /mnt/d/vicad-syn/task-004/rectification/audit-tables
```

另用原导出器只读重验 22 个训练检查点的实际 SHA-256（确认 18 个、探索 4 个）、前置回执、正常训练与测试特征、每组预测结构、掩膜及数组哈希。重新导出的 formal.csv、formal-summary.csv、开发矩阵及核验回执与原件 SHA-256 完全一致。输出在 `D:/vicad-syn/task-004/rectification/reverified`。[重新校验记录](../results/vicad_refinement/evidence/rectification-test-log.txt)

原有数据位于 `E:/date/task-002-stage2`、`E:/date/task-004-bird`、`E:/date/task-004-holdout`。冻结的运行输出仍位于 `D:/vicad-syn/task-004/run-v2`。本次复评的原始 stdout/stderr 位于 `D:/vicad-syn/task-004/rectification/replay.log` 和 `replay.stderr`。历史询问与复验仍保留在原 PR #5。
