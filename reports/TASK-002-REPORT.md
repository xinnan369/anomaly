# TASK-002 · M²AD 官方基线复现与实验环境固化

- 日期　2026-11-19（北京时间）
- 范围　第一阶段环境固化和单卡 Car smoke test
- 状态　TASK-002 第一阶段：独立验收通过（APPROVED）。TASK-002 整体尚未完成，仍为当前任务，PR #2 保持开放。
- 验收记录　ChatGPT Review ID `5970938786`，验收提交 `90fe550c28d00191ac67c0739d6097b6472fdc4b`。[审查记录](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-5970938786)
- 验收范围　仅包括 WSL2 环境固化、RTX 3090 CUDA 环境、Car/Dinomaly 单步训练、留出正常实例推理及 ADEval CUDA 合成测试。

## 1. 前置条件与上游源码

GitHub PR #1 经项目负责人明确授权，以合并提交 `d2da56c11fef084f47b90b201b0f905ec8e29c3e` 合入 `main`。检查确认整改提交 `cf0df0be764c895c15d9d68c2390f1c3bded16f7` 是其祖先，报告、审计工具与四份机器可读证据及测试日志齐全。合并后的 9 项离线审计测试通过。TASK-002 从这个 `main` 提交创建独立分支。 [PR #1](https://github.com/xinnan369/anomaly/pull/1) · [TASK-001 报告](TASK-001-REPORT.md)

检查了官方 `hustCYQ/M2AD` 的固定提交 `a606aa0e1614f23dad70da26430dd66095192141`，使用 Git 的浅层 sparse checkout，只展开配置、数据、模型、trainer、损失、评估工具及优化器代码。脚本在运行前校验 HEAD 和跟踪文件的 Git 差异。环境适配使用本项目的外部启动脚本和运行时路径，官方基线源码保留原样。 [上游代码](https://github.com/hustCYQ/M2AD/tree/a606aa0e1614f23dad70da26430dd66095192141)

## 2. 环境、依赖和隔离

实际设备为本机 RTX 3090（CC 8.6），Windows 驱动 610.88，显存 24,576 MiB。运行使用 WSL2 Ubuntu 22.04 x86_64（Linux 5.15.167.4-microsoft-standard-WSL2）中的 Python 3.11.11 独立 venv。实际运行库为 `torch 2.2.2+cu121`、`torchvision 0.17.2+cu121`、CUDA runtime 12.1、cuDNN 8.9.2，`torch.cuda.is_available()` 为真，CUDA 张量求和为 28。上游 README 的 PyTorch 2.1.2/cu118 和 Python 3.9 组合作为参考保留。RTX 4090 **未验证**。

为避免重新传输同样的 GPU 运行库，使用已有 WSL 环境中 PyTorch 2.2.2、torchvision 0.17.2、Triton 2.2.0 和对应 NVIDIA wheels 的安装文件，复制进独立 venv。其余依赖在 venv 内通过 PyPI 安装。已有环境未作为 site-packages 搜索路径共享，项目仓库不包含这些二进制文件。初次 pip 从默认镜像取 PyTorch wheel 过慢，终止该下载后采用复制既有二进制与显式 PyPI 索引的方式。实际 `python -m pip check` 返回 `No broken requirements found.`。以 `python -m pip freeze --all` 记录全部 87 个发行包，包含官方建议的 `timm==0.8.15.dev0`、`numpy==1.26.4`、`opencv-python==4.9.0.80`、`ADEval==1.1.0`。锁文件记录实际版本，从空环境完整重装尚未复测。 [环境命令](../tools/environment/README.md) · [依赖锁](../tools/environment/requirements-wsl2.lock)

第一次试运行停在官方 `util/net.py` 的 `from collections import Iterable`。Python 3.11 已移除该别名。外部启动脚本在进程内将它指向标准库的 `collections.abc.Iterable`，第二次运行通过。上游跟踪文件无改动。

## 3. 数据和权重

从 Hugging Face 发布 revision `05abc5d03167354a7ef98541236b5c0bd00abadf` 的 `Car.zip` 读取 ZIP 尾部、中央目录及三个成员，共收到 5,537,749 字节的成功 Range 响应。按固定清单选用最前两个正常训练物件 ID，`000` 用于 fit、`003` 用于 probe。ZIP CRC 在成员完整读取时验证。脚本输出每张图像的 SHA-256。清单中的原始 Synergy test 数据未进入任何试运行步骤。 [固定发布](https://huggingface.co/datasets/ChengYuQi99/M2AD/tree/05abc5d03167354a7ef98541236b5c0bd00abadf) · [数据准备脚本](../tools/environment/prepare_car.py)

| 用途 | 图像路径 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| Fit | `Car/Good/000/A000_I01.png` | 1,362,909 | `e0224ce1e0f3755eb5e8bbba49dbf971284718b7408c2eb7b24d4f598ba16226` |
| Fit | `Car/Good/000/A000_I05.png` | 1,314,648 | `a112ffff2795a65c069e04390ec0e8ac41d0689c97bd6898e8b569b5e7ced544` |
| Probe | `Car/Good/003/A000_I01.png` | 1,356,952 | `3d3a91190b8b216a898632597104a03ad9cd9ed8e373674b60d85ecbe1fb5e87` |

Dinomaly 配置指定 `dinov2reg_vit_base_14`。从 Meta 官方下载 DINOv2 ViT-B/14 的 reg4 骨干文件 `dinov2_vitb14_reg4_pretrain.pth`，共 346,393,545 字节。下载先尝试单流，随后使用 8 MiB 并行分段，检查每段 HTTP 206、`Content-Range` 与长度，按偏移顺序组装。Python 流式 SHA-256 与 PowerShell `Get-FileHash` 一致，哈希为 `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`。这是实测文件哈希。该文件是预训练骨干，Car 专用训练权重未下载、未生成保存。 [Meta 权重列表](https://github.com/facebookresearch/dinov2#pretrained-models) · [实际权重 URL](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth)

## 4. Dinomaly 端到端试运行

运行入口在官方固定版本之外，使用官方 M2AD Dataset 和 Dinomaly 配置预处理，再调用官方模型初始化、trainer 的输入/前向/损失/反向更新/异常打分函数。外部入口将类别限定为 Car，读取两个小型、分物件的训练清单视图，执行一次更新和单图 probe。官方测试入口、融合和正式分数均未调用。运行参数使用 seed 42、224×224、fit batch 2、probe batch 1、无 dataloader worker，保留官方配置中由默认批量推导的 Adam 学习率与 betas。 [Dinomaly 配置](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/configs/benchmark/dinomaly/dinomaly_100e.py) · [试运行脚本](../tools/environment/smoke_dinomaly.py)

**结果　PASS。** 加载张量分别为 `[2,3,224,224]` 和 `[1,3,224,224]`。模型确认 patch size 14、4 个 register token，配置使用 `DinomalyLoss(p=0.9, factor=0.0)`、Adam 学习率 0.00375、betas 0.5/0.999、无 AMP scaler。单步 loss 为 1.02214956，抽查的 bottleneck 参数发生更新，编码器参数无梯度。probe 输出有限的热图 `[1,224,224]` 和单图分数 `[1]`。loss 只作执行证据。

| 实测环节 | 时间（秒） |
| --- | ---: |
| 读取和预处理 fit batch | 0.2175 |
| 读取和预处理 probe batch | 0.0937 |
| 初始化模型、载入权重并搬至 CUDA | 2.7569 |
| fit 数据搬至 CUDA | 0.0003 |
| 官方前向及损失 | 0.3574 |
| 官方反向及优化器更新 | 0.2007 |
| probe 推理及异常打分 | 0.0223 |
| 驱动脚本整个运行函数（含哈希、依赖导入和 CUDA 初始化） | 12.8561 |

每个分段计时前后同步 CUDA。PyTorch allocator 峰值 allocated 为 **1,621,356,032 B（1.51 GiB）**，reserved 为 **1,864,368,128 B（1.74 GiB）**。统计在 CUDA 初始化之后重置，范围覆盖两批数据、模型初始化、训练步骤和推理。这是单次小批量的进程内 PyTorch 计量，整卡占用与正式批量显存另测。

实际以 `CUDA_VISIBLE_DEVICES=0` 运行，以下用变量代替仓库外的临时路径。

```bash
python -B tools/environment/prepare_car.py --cache "$release_cache" --output "$car_fixture"
CUDA_VISIBLE_DEVICES=0 python -B tools/environment/smoke_dinomaly.py \
  --upstream "$upstream" --data "$car_fixture" --weight "$weight" \
  --weight-sha256 73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71
CUDA_VISIBLE_DEVICES=0 python -B tools/audit/metric_smoke.py --cache "$audit_cache" --device cuda
```

## 5. INP-Former 与 ADEval

INP-Former 配置同样选择 DINOv2 ViT-B/14 reg4，`INP_num=6`，官方 256/512 输入配置映射为 224/448，默认批量为 12。其实现使用 Transformer 原型模块并复用 reg4 权重源。 [上游配置](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/configs/benchmark/inpformer/inpformer_100e.py) · [上游模型入口](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/model/inpformer.py)

在同一 venv 中实际导入了 `INPFormerTrainer` 和模型注册入口，实例化了官方配置并读回 `model=inpformer`、`size=224`、`encoder_arch=dinov2reg_vit_base_14`、`INP_num=6`。依赖检查覆盖该入口的 PyTorch、torchvision、timm、NumPy、SciPy、scikit-learn 以及本地 vendored DINOv2；Python 和共享依赖入口通过。在本阶段未初始化 INP-Former 模型本体。

Dinomaly 运行中通过 ADEval CUDA 类导入；同一环境另用 TASK-001 已校验的 ADEval 1.1.0 wheel（SHA-256 `125caf991037a668a1f072f7e99c81b6bdf86dc8c30cf1699c731d4b79e0bd6e`）执行两组合成数据。官方融合的分组均值及掩膜并集断言通过，`nstrips=50` 下合成 AUPRO 为 1.0 和 0.5。ADEval 输出已知的 precision 除法警告，三项检查指标均有限。此处结果只反映合成测试。

## 6. 边界与待验证项

- 第一阶段的单次试运行验证了从数据读取到优化和推理的执行路径。方法排名、准确率、收敛性和吞吐量需要正式实验。
- 待完成：从空环境按锁文件完整重装复验、INP-Former 端到端训练、RTX 4090 验证、正式基线训练与测试评估（含 Car Synergy 全配置评估）。上游 README 的原始整套环境也尚未实测。软件锁记录本次实际 Python 环境，系统驱动和 Linux 运行时另列。
- 保留 TASK-001 确认的发布数量差异。论文 v1 图 3(a) 比固定发布多 1 个 Holder 异常实例，不能据此补造数据。 [TASK-001 报告](TASK-001-REPORT.md)

## 第二阶段执行记录

**状态 PARTIAL（2026-11-19，北京时间）**。全新环境重装、两种模型的 Car 单步复验、官方训练入口预检和发布清单复核均已完成。完整归档仍待校验，正式训练与测试、结果分析及图表待完成。第一阶段的验收范围和历史记录见上文。

### 1. 研究目标

在 RTX 3090 上按固定的 M²AD-Synergy 协议复现 Dinomaly 和 INP-Former。每模型依次训练 Car、Cube、Doll、Ring，单类使用 100 epoch，最终一次测试。用实际预测及运行日志比较四类的准确率、视角与光照适应性和资源成本。

### 2. 实验环境

实测 WSL2 Ubuntu 22.04.5 LTS、Linux 5.15.167.4-microsoft-standard-WSL2，RTX 3090（CC 8.6，24,576 MiB），Windows 驱动 610.88。RTX 4090 未实测。用 CPython 3.11.11 创建独立的 `venv --copies`，新环境初始包集合只有 `pip` 和 `setuptools`。通过 PyPI 和 pip 下载缓存的发行包按现有锁重新安装，初始 Python 可执行文件来自已存在的 Conda 环境，安装未共享或复制原有 site-packages。第一阶段环境保留。实装 87 项与现有锁一致，`pip check` 返回 `No broken requirements found.`。

实测 PyTorch 2.2.2+cu121、torchvision 0.17.2+cu121、CUDA 12.1、cuDNN 8.9.2、timm 0.8.15dev0、ADEval 1.1.0、NumPy 1.26.4、OpenCV 4.9.0。CUDA 张量运算通过。完整冻结列表和记录见[环境证据](../results/baselines/README.md#环境和入口预检)。

### 3. 数据集版本

上游固定在 `a606aa0e1614f23dad70da26430dd66095192141`，代码差异检查通过。数据使用发布 `05abc5d03167354a7ef98541236b5c0bd00abadf`，已对本地 `jsons.zip` 校验 SHA-256 `bbeaa75c231c981e443c818859a701fcc0daf1769065abd91631f7e676725980`。四类远端 ZIP 中央目录与清单的图像、掩膜和分割路径一致。以下 ZIP SHA-256 来自发布 LFS 元数据，完整本地归档的 SHA-256 和全部成员的 CRC 仍待验证。[上游代码](https://github.com/hustCYQ/M2AD/tree/a606aa0e1614f23dad70da26430dd66095192141) · [发布目录](https://huggingface.co/datasets/ChengYuQi99/M2AD/tree/05abc5d03167354a7ef98541236b5c0bd00abadf) · [LFS 元数据](https://huggingface.co/api/datasets/ChengYuQi99/M2AD/tree/05abc5d03167354a7ef98541236b5c0bd00abadf?expand=true)

| 类别 | 发布 ZIP 字节 | 成员数 | 发布 ZIP SHA-256 |
| --- | ---: | ---: | --- |
| Car | 16,318,099,753 | 12,970 | `27e77cfe0faad99065489ce5c5a785a2450da771372997dc9cfca160e778af36` |
| Cube | 17,628,365,529 | 13,106 | `72b5d96a1281c61d247771387b0a7bc66307c7d6e9c14112ae0553b5b89cafe7` |
| Doll | 16,835,932,455 | 13,002 | `1fd27c8aaf89d6bdde1d5061edcc5e45a5c8d7241c9d287402807a1985f8bcb3` |
| Ring | 16,564,602,406 | 13,096 | `20b2b0f12c338747208a914aa98e4f175eef7fa0e26de8eae0bc1f8ac7a27d7c` |

四类压缩 62.72 GiB、解包约 62.79 GiB（依据目录记录的字节数）。D 盘启动前可用约 371.5 GiB。

### 4. 数据划分

TASK-001 审计脚本复核官方清单，9 项离线测试通过。每物理实例覆盖 12 个视角（`000` 至 `330`，间隔 30°）与 10 个光照（`01` 至 `10`）的全部组合。训练只含正常实例，官方清单无独立验证集。训练与测试的物理实例分组未重叠，远端中央目录中的训练/测试图像 CRC32 与长度二元组也无交集；磁盘全量检查仍依赖完整档案。[清单结果](../results/baselines/data_audit/manifest-summary.json) · [远端目录检查](../results/baselines/data_audit/remote-index.json)

| 类别 | 训练正常实例 / 图像 | 测试正常实例 / 图像 | 测试异常实例 / 图像 | 异常实例中标注为异常的图像 |
| --- | --- | --- | --- | ---: |
| Car | 30 / 3,600 | 20 / 2,400 | 50 / 6,000 | 4,850 |
| Cube | 30 / 3,600 | 20 / 2,400 | 50 / 6,000 | 5,530 |
| Doll | 30 / 3,600 | 20 / 2,400 | 50 / 6,000 | 5,010 |
| Ring | 30 / 3,600 | 20 / 2,400 | 50 / 6,000 | 5,480 |

### 5. 模型结构

均使用官方 DINOv2 ViT-B/14 reg4 骨干与各自重建结构，实测两种模型都加载了 patch size 14、4 个 register token 的编码器。INP-Former 实测 6×768 原型参数及其单步更新。入口预检的完整模型参数量分别为 148,008,960（Dinomaly）和 155,101,536（INP-Former），含冻结编码器。[Dinomaly 配置](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/configs/benchmark/dinomaly/dinomaly_100e.py) · [INP-Former 配置](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/configs/benchmark/inpformer/inpformer_100e.py)

### 6. 正式超参数

预检了固定版本的两份 `100e` 配置。两者共用 seed 42、224×224 输入、训练和测试 batch 12、Adam（学习率 0.00375，betas 0.5/0.999）、step 调度、0 warmup、80 epoch 衰减配置和梯度裁剪 5。损失分别为官方 `DinomalyLoss(p=0.9, factor=0.0)` 与 `INPFormerLoss(y=3)`。正式入口沿用官方数据变换和打分。外部启动器接管运行路径、兼容性别名、日志和最终评价时点。第 50 和第 100 epoch 的内联测试均采用上游空评估，八组训练完成后从第 100 epoch 的冻结检查点执行独立评价。批量逐图渲染关闭，保留原始预测数组。[配置和预检证据](../results/baselines/environment/preflight.json)

### 7. 训练过程

八个组合都通过官方入口预检。使用固定清单及官方 `run_dataset.py` 初始化数据加载器、模型、优化器与评估器，逐组合读回 3,600/8,400 行与 300/700 批；此项不迭代加载器。两种 Car 单步实测各用 2 张 fit 图像做一次优化、1 张另一官方训练正常实例图像做推理。两种方法均有可测的参数更新，编码器无梯度，推理输出有限，试运行未使用官方测试图像。

快照时四类档案尚无完整审计标记，八组正式训练各 0/100 epoch。正式入口的缺档保护实际拦截了 Car/Dinomaly，记录异常后退出，没有开始优化。[预检和拦截证据](../results/baselines/environment/preflight.json)

### 8. 正式指标

空值表示尚无测量。最终预定报告官方 `AUROC_obj_sp`、`AUROC_view_sp`、`AUPRO_view`（分别对应 O-AUROC、I-AUROC、AUPRO）及其他上游返回指标。[机器可读总表](../results/baselines/metrics/baseline-summary.csv)

| 模型 | 类别 | 正式状态 | Epoch | O-AUROC | I-AUROC | AUPRO |
| --- | --- | --- | ---: | --- | --- | --- |
| Dinomaly | Car | 未开始 | 0/100 | — | — | — |
| Dinomaly | Cube | 未开始 | 0/100 | — | — | — |
| Dinomaly | Doll | 未开始 | 0/100 | — | — | — |
| Dinomaly | Ring | 未开始 | 0/100 | — | — | — |
| INP-Former | Car | 未开始 | 0/100 | — | — | — |
| INP-Former | Cube | 未开始 | 0/100 | — | — | — |
| INP-Former | Doll | 未开始 | 0/100 | — | — | — |
| INP-Former | Ring | 未开始 | 0/100 | — | — | — |

固定评估先使用官方融合对同实例、同视角的不同光照求预测均值，掩膜取并集；物件分数跨 120 张图像取均值。启动器将输入分组和标签与清单逐行比对，并按 TASK-001 的分组逻辑复核官方融合数组。ADEval CUDA 与官方融合的合成回归通过。独立合成样例产生 142 条诊断输出（120 条视角×光照，10 条逐光照，12 条融合视角），并能拒绝被篡改的融合分数。合成结果只用于核查代码。真实子组诊断、指标有限性和掩膜检查待正式评价。[TASK-001 评估审计](TASK-001-REPORT.md) · [回归证据](../results/baselines/environment/smoke-and-evaluator.json)

### 9. 性能与资源分析

以下是同一 RTX 3090 上的小批量单次试运行，可用于定位各环节是否执行。训练、推理和显存的正式统计由启动器逐组记录，当前无数据可支持吞吐对比、显存上限判断或方法排名。

| 模型 | Fit loss | 前向与损失（s） | 反向及更新（s） | 留出推理（s） | CUDA 峰值 allocated（B） | 峰值 reserved（B） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dinomaly | 1.022150 | 0.1524 | 0.1382 | 0.0172 | 1,621,356,032 | 1,864,368,128 |
| INP-Former | 1.159236 | 0.1498 | 0.1214 | 0.0200 | 1,750,724,608 | 1,958,739,968 |

### 10. 可视化

论文级图表待正式数据产生后绘制，包括模型与类别指标、视角/光照分布、训练时间、显存和匿名实验样本热力图。当前没有可用于上述图表的正式实测输出。

### 11. 实验复现命令

重装、阶段二 smoke、完整档案校验、按序训练、冻结检查点复评和汇总的命令见[环境说明](../tools/environment/README.md#第二阶段完整基线)。审计实际执行 `python -B -m unittest discover -s tools/audit -p 'test_*.py'` 与 `python -B -X utf8 tools/audit/archives.py --cache <发布缓存> --classes Car Cube Doll Ring`。CUDA 上另执行 `python -B tools/environment/check_baseline_metrics.py --upstream <固定源码>`。八个正式组合沿用上游 seed 42。

### 12. 失败与阻塞记录

- 从 WSL 直接执行 Hugging Face Range 请求失败，`curl` exit 7（连接拒绝）。Windows 单路 8 MiB 试探返回 HTTP 206，35 秒收到 5,924,549 B 后超时（exit 28）。改用 Windows 上 10 路可续传分块下载。
- 16:34:34 快照记录 Car 的 244 个分块中有 152 个完整块（10,200,547,328 B）。这些分块已验证 HTTP 边界和长度，正式 ZIP SHA-256 尚待组装后核验。快照中四类全档校验为 0/4。[下载快照](../results/baselines/data_audit/download-snapshot.json)
- 正式启动器已实测拒绝缺档，错误为 `Car: no complete archive audit; run prepare_release.py first`。该次只检查数据前置条件，未触发训练，因此正式 OOM、NaN 或 Inf 均待实测。
- 在合成回归中发现官方融合后的掩膜去掉了单通道维度。校验器已按官方输出形状比较，重跑通过。上游 ADEval 在合成样例上仍可发出除法警告，诊断及断言结果有限。

### 13. 研究局限

四类的完整像素文件、掩膜与整批数据在本轮时点尚未全部校验。单步试运行与入口预检无法判断正式 batch 12 的显存峰值、收敛或模型效果。视角/光照稳健性、不同类别难度和 VICAD-Syn 方法假设需要预定的正式结果。TASK-001 已记录的 Holder 发布版差异保持开放，RTX 4090 未实测。[TASK-001 未决事项](TASK-001-REPORT.md)

### 14. 剩余工作与验收

先完成四类 ZIP 的发布哈希与本地 CRC 校验，再按顺序训练八组，执行冻结后的唯一官方测试，核查所有输出，汇总资源并制作图表。A（空环境安装）和 J（文档同步）已有证据。B 至 E 及 I 等待正式实验。F、G、H 与 M 已有清单、配置、溯源及隔离证据，仍须在正式运行中核验。K 和 L 自检通过，执行了 `git diff --check` 并核对仓库内 98 个 Markdown 相对链接及锚点。阶段二完整验收条件尚未达到。

### 15. 原始证据位置与哈希

原始环境、下载、预检及未来正式输出保存在仓库外的 `D:/vicad-syn/task-002-stage2`（WSL `/mnt/d/vicad-syn/task-002-stage2`）。固定源码、权重与 Car 单步样本留在 `inputs/`。训练与评价结果分别在 `runs/<模型>/<类别>/` 和 `evaluations/<模型>/<类别>/`，四类就绪信息在 `data_audit/`，传输日志在 `download.log`。元数据 SHA-256 与发布四类哈希见第 3 节。两次实测骨干字节数为 346,393,545，SHA-256 均为 `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`。完整实验检查点尚未生成。仓库提交环境、清单、预检与未完成状态的小型快照。[结果目录](../results/baselines/README.md) · [Meta 预训练权重](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth)
