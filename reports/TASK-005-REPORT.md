# TASK-005 · 完整基线对照、正式消融与论文证据补强

**状态　PARTIAL。** 任务已启动，十二组新增官方训练尚未完成。结果表保留全部计划项，缺失正式指标留空。当前材料仅支持已完成范围的结论。[结果索引](../results/comprehensive_evaluation/README.md) · [执行协议](../instructions/TASK-005.md)

## TASK-004 结项

PR #5 的 OPEN 状态、已批准 HEAD `6131d278bf104196613f6e00f3c400c01b2eb8f8`、Review `6038575784` 的 APPROVED 与提交绑定、无冲突及无阻断性检查均已核实。以预期 HEAD 参数执行普通合并，GitHub 返回 merge SHA `f67b451683982e3632d45d8511dee7caec2a6732`，当时的 main 同值。负责人接受历史下载的事后处置，保留原数据、缓存和成果。历史事前审批仍未能核实。结项身份为 **ACCEPTED_WITH_DOCUMENTED_HISTORICAL_EXCEPTION**。[验收日志](../docs/ACCEPTANCE_LOG.md) · [历史下载审计](TASK-004-DOWNLOAD-AUDIT.md)

## 本地资源与预算

十类解压数据的成员路径与固定发布清单逐项比对、文件长度非零。历史归档 SHA-256 和 CRC 使用已存在的完整性回执。提取条件不含原 ZIP 重验。原样保留 Dice 的 29 个正常训练物件和 Holder 的 69 个测试物件。十类特征缓存和 B0/B6 预测文件大小逐项匹配既有索引，22 个检查点按字节重新哈希并通过预检冻结绑定。两模型使用同一已验证的 DINOv2 权重 `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`，346,393,545 字节。旧缓存完整 SHA-256 在实际调用时复核。[本地输入审计](../results/comprehensive_evaluation/evidence/resource-audit.json)

硬件为 RTX 3090 24 GiB，WSL2 Ubuntu 22.04，锁定的 Python、PyTorch CUDA 及 ADEval 环境。启动时新产物所在 E 盘可用约 470 GiB。现有 TASK-002 记录显示一组严格基线约需数小时；本轮 GPU 和存储存在共享负载，最新日志优先于历史估算。新产物位于仓库外的 `E:/date/task-005`。更新的完整性检查和队列均保留已完成产物。

## 原版基线进度

Bird 的 Dinomaly、INP-Former 均通过本地 CUDA 预检，校验两套 loader 的 3,600／8,400 图像和 300／700 batch。正式运行采用原版 100 epoch、seed 42、batch 12、官方损失及优化器。Bird 两条正式训练及冻结评价闭环通过后队列才推进其余类别。若原版失败，保留失败记录和空指标。已接受的 Doll 原版失败、独立稳定变体及历史恢复留在 TASK-002。Dice 为 29,000 步，其他常规类为 30,000 步。Holder 采用 8,280 张测试图和 828 个融合视角。[正式矩阵](../results/comprehensive_evaluation/tables/official-progress.csv) · [十类主表](../results/comprehensive_evaluation/tables/local-results.csv)

## 消融与统计

先于新评价提交协议 `48aa0ce066deb050f3da6ee33775a1640a1e2fd8`，SHA-256 `66d864ae2e06b6ee0429253ebdb7ea06acc9918ec3827dc0a24a513e3522ff1e`。B0/B6 沿用已冻结产物。新 B1–B5、B7–B9 读取缓存和冻结检查点，不重新拟合。B1/B2 是同一正常原型法的单层／多层对照，B4/B2、B6/B4 分别比较预测残差和局部稳定性边际贡献。官方指标按固定 Synergy 评价，身份为类别、物件编号和正常／异常状态。I-AUROC 使用官方融合掩膜的非空状态作为视角标签。正常与异常同名时仍分组，并复核重算的 AUC 与官方点估计吻合。

类别内部整物件配对 bootstrap 采用固定 seed 42005 和 2,000 次 draws，正常和异常分层并保持各物件的全部视角。按事先指定的比较族保留普通 95% 百分位区间和 Bonferroni 同时区间。AUPRO 保留官方点估计。每次重采样的官方 ADEval AUPRO 区间尚未完成验证，区间留空。统计区间反映已见数据中的类别内样本不确定性，三种子样本标准差描述训练种子波动，两者分列。[消融矩阵](../results/comprehensive_evaluation/tables/ablation.csv) · [种子均值、SD 和逐种子差值](../results/comprehensive_evaluation/tables/paired-seed-summary.csv)

## 论文与部署证据边界

TASK-004 冻结组合相对配对 v1 的既有收益支持研究失效机制。当前正式新增基线与消融覆盖尚不完整。同环境强基线的领先范围、多层原型是否已足够、预测器是否必要、局部融合是否值得额外成本均待补齐比较后判断。各表分列本机与作者报告，宏平均标明实际类别覆盖。原研究结构图和机制图继续保留。单次 GPU 时延探针与共享整类别计时须分列，不能将共享特征提取时间归给单独设置。现有证据允许继续补强机制型论文，尚不足以确认达到 SCI 二区论文所需的同环境全面比较。[现有结构与机制图](../results/vicad_refinement/README.md)

以 seed 42 比较十类已保存配对结果，冻结改进版的 O-AUROC／I-AUROC／AUPRO 宏平均为 **82.43／69.25／82.20%**，配对 v1 为 **61.33／53.33／36.62%**。与已验收的本机原版 Dinomaly 比较，三个共同已完成类别的三项均值均落后。与本机 INP-Former 的四个共同类别比较，三项宏平均也均较低，逐项领先类别数为 2／1／1。Doll 的原版 Dinomaly 保持失败，稳定版单列。当前依据支持机制研究和组合相对 v1 的提升。强基线领先、论文泛化范围与部署价值仍受后续完整对照约束。[同覆盖范围对照](../results/comprehensive_evaluation/tables/same-coverage-baseline.csv) · [三种子图](../results/comprehensive_evaluation/figures/historical-seed-stability.pdf)

已覆盖六类的 3 个种子和四类的 seed 42，共 44 条 B6/B0 物件级区间，每条 2,000 次有效抽样。22 个配对运行中，O-AUROC 有 21 个正点估计，11 个校正区间全为正。I-AUROC 分别为 22 和 19。首次 Bird 区间从原 PNG 掩膜复算，后用冻结掩膜哈希及身份重绑重跑；两种入口的 Bird 结果完全一致。Bird 的历史 B6/B0 seed 42 配对结果中，O-AUROC 为 +9.80 个百分点，普通 95% 区间为 [−1.60, 20.80]。I-AUROC 为 +11.63 个百分点，普通区间为 [5.51, 17.45]。约定的 Bonferroni 区间也一并公开。2,000 抽样下校正区间靠近经验尾部极端秩，适合作保守核查。这是分层配对分析，评价数据来自 TASK-004 已完成的历史运行。[区间及覆盖快照](../results/comprehensive_evaluation/tables/historical-paired-ci.csv)

Bird 的单物件时延探针同时记录两个设置。冻结主干 86,583,552 参数，可训练预测头 478,656 参数，三层类别原型占 283,115,520 字节（270 MiB）。在缓存条件下，对同一个 120 图物件，B2 和 B6 的时延中位数约为 0.15 和 0.43 秒。端到端从本地 PNG 读取起计，两个 120 图中位数为 22.64 和 70.28 秒；GPU 和磁盘由多个进程共享，分段测量受负载变化影响，只作为一次实测探针。单视角十光照时延、增量显存、耗时原始重复值列入 JSON。[资源探针](../results/comprehensive_evaluation/evidence/bird-runtime.json) · [性能图](../results/comprehensive_evaluation/figures/bird-runtime.pdf)

## 可复现入口与零下载审计

命令在当前任务工作树的 WSL 目录中运行。以下均读取已有本地目录。输出路径已占用时检查现有状态后再继续。运行失败的产物留在原外部目录。

```bash
py="$HOME/.local/venvs/vicad-task-002-stage2/bin/python"
initial=/mnt/d/vicad-syn/task-002-stage2
bird=/mnt/d/vicad-syn/task-004/holdout
holdout=/mnt/c/Users/xinnan/vicad-data/task-004-holdout
frozen=/mnt/d/vicad-syn/task-004/run-v2
work=/mnt/e/date/task-005
export PATH=/usr/lib/wsl/lib:$PATH
export CUDA_VISIBLE_DEVICES=0
"$py" -B -u -m tools.comprehensive.run_official --initial "$initial" --bird "$bird" --holdout "$holdout" --audit "$work/evidence/resource-audit.json" --output "$work/official"
"$py" -B -u -m tools.comprehensive.ablation --initial "$initial" --bird "$bird" --holdout "$holdout" --frozen "$frozen" --audit "$work/evidence/resource-audit.json" --output "$work/ablation"
"$py" -B -m tools.comprehensive.collect --work "$work" --output "$work/export-next"
```

实验进程禁止 Python 的网络连接、DNS 和模型资源自动获取，Git 固定源码禁用懒加载。没有在线安装、镜像探测、模型或数据下载、真实签名测试。新增实验资源下载为 **0 字节**。GitHub 请求用于合并元数据和代码发布。既有 65 项及新增 9 项单元测试通过，下载相关测试仅使用模拟。[验证记录](../results/comprehensive_evaluation/evidence/verification.md)
