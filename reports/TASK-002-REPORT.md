# TASK-002 · M²AD 官方基线复现与实验环境固化

- 当前状态　**ACCEPTED_WITH_EXCEPTION / 有限例外验收通过**。2026-11-23，PR #2 普通合并。Merge SHA 与合并后的 main 均为 d1ddbb9d698114ccb616ad96c23565f7f2b1e275。[正式验收](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-6018691109) · [合并提交](https://github.com/xinnan369/anomaly/commit/d1ddbb9d698114ccb616ad96c23565f7f2b1e275)
- 日期　2026-11-19（北京时间）
- 范围　第一阶段环境固化和单卡 Car smoke test
- 第一阶段验收状态　独立验收通过（APPROVED）。
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

**2026-11-19 首次交付快照（PARTIAL）。** 当时已完成全新环境重装、两种模型的 Car 单步复验、官方训练入口预检和发布清单复核。后续进度见[审查整改与正式实验](#第二阶段审查整改与正式实验)。

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

## 第二阶段审查整改与正式实验

**2026-11-19 整改交付快照（PARTIAL）。** 当时实例身份回归通过，RTX 3090 已开始正式 Car/Dinomaly 基线。下文保留当时的证据，[第二阶段最终结果](#第二阶段最终结果)记录后续完成的八组。第一阶段验收记录继续有效。[第二阶段审查](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-5974527521)

### 实例身份、标签和回归

- 重新校验固定发布的 `jsons.zip`。10 类测试集 Good 与 NG 的裸实例名交集均为空。Car、Cube、Doll、Ring 每类有 20 个正常和 50 个异常物件，完整预期为 70 个物件组、840 个视角组、8,400 张图像。Holder 的发布版差异仍按 TASK-001 记录。
- 官方物件键为 `(object_name, cls_name, object_anomaly)`，视角键为 `(object_name, cls_name, view, object_anomaly)`。校验器按完整键生成期望归约，在同类同显示名称内对分数、标签、掩膜与热图作无序一对一匹配。这样既保留重复输出的数量，也要求每行内容符合对应组。官方融合代码保留原样。
- O-AUROC 使用物件标签。I-AUROC 使用评价尺寸下的联合掩膜是否非空，AUPRO 使用联合掩膜区域。上游融合的视角元数据标签取同视角各光照图像标签的最大值，可能与掩膜存在性不同。诊断保留两者计数和分歧，单一掩膜标签的子组输出空指标。
- RTX 3090 CUDA 合成回归使用两个同名物件，各有 120 张配置。在同一异常物件含部分可见光照、存在标签与掩膜分歧时验证乱序组、故意篡改物件及视角分数和三项官方指标入口。142 个子组中 119 个因只有单一掩膜标签而留空，5 个子组存在标签分歧。合成值只用于校验程序。[固定上游融合](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/util/m2ad_utils.py) · [固定上游评价](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/util/metric.py) · [发布清单及合成回归证据](../results/baselines/environment/identity-regression.json)

### 数据恢复

前一次下载在 Ring 的字节范围 `5637144576–5704253439` 触发 `ConnectionError`，保留了 84 个已完成的分块。续传使用 `python -B -X utf8 -u tools/environment/prepare_release.py --root <外部目录> --classes Ring --workers 10`，全文日志分别保存在外部数据根的 `download.log` 和 `download-retry-ring.log`。Car、Cube、Doll 的原始完整 ZIP SHA-256、成员 CRC 及清单覆盖均通过。整改时复算了各 ZIP SHA-256、中央目录与磁盘路径集合。官方训练/测试实例身份无交叉，完整归档证据见[恢复与复验证据](../results/baselines/data_audit/restored-archives.json)。

### 正式闭环和日志

在 WSL2 Ubuntu 22.04、Python 3.11.11、RTX 3090 24 GiB 上启动 `CUDA_VISIBLE_DEVICES=0 <锁定的 Python> -B -u tools/environment/run_stage2.py --root /mnt/d/vicad-syn/task-002-stage2`，按 Car/Dinomaly 闭环优先排定八组。Car 的正式配置保持 seed 42、batch 12、100 epoch、官方优化器、调度、损失及权重。第 50 和第 100 epoch 的训练内测试均采用上游空评估。Car 完成训练并通过冻结后的正式评价，才会接续其余七组。

正式运行写出了 `checkpoints/DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_100e_exp1/['Car']_ckpt.pth`，使用 `torch.load(..., weights_only=True)` 只读核验了实际中途检查点的字段与步数。冻结入口要求最终检查点由成功训练的 `status.json` 绑定，路径、字节数、SHA-256、配置、模型、类别和固定发布版本均匹配，内部须含 epoch 100 与迭代 30,000。7 项新单元测试验证了实际路径契约、错误的 epoch/步数/哈希、以及结果表与同一冻结权重的对应关系。9 项 TASK-001 审计测试仍通过。[检查点实测及单元测试](../results/baselines/environment/checkpoint-validation.json)

交付时的训练步数、整卡显存、运行命令和下载进度见[机器可读快照](../results/baselines/metrics/formal-execution.json)。完整训练日志与每两秒 GPU 采样保存在外部数据根，仓库只提交实际日志节选与索引。PyTorch 峰值、完成耗时、推理和最终权重哈希须等相应进程结束。[实际日志节选](../results/baselines/metrics/car-dinomaly-train-excerpt.log) · [八组状态与指标表](../results/baselines/metrics/baseline-summary.csv)

### 剩余结果与研究判断

记录时第一组仍在训练，冻结评价、完整指标和论文图表均待产生。正式评价会收集每个预测数组的形状、类型、字节数及 SHA-256，并保留相同冻结检查点的复评路径。单卡串行的耗时由实际日志持续记录。汇总和绘图已在外部数据根排在运行计划之后，日志写入 `logs/finalize-stage2.log`。绘图入口在八组完成前拒绝运行。

现有记录尚不足以给出类别难度、误报光照、视角稳定性、平均融合的局限或 VICAD-Syn 方法效果。正式输出后，以正常物件的逐光照分数分布观察高分倾向，结合逐视角、光照及 120 种配置的诊断比较。高分倾向目前不等同于选定阈值后的误报率。RTX 4090 和 TASK-001 的 Holder 差异仍待后续对应核查。[恢复与结果生成](../tools/environment/README.md#第二阶段完整基线) · [结果证据索引](../results/baselines/README.md)

### 交付自检

7 项本阶段单元测试、9 项原审计测试、CUDA 身份回归和 `git diff --check` 通过。114 个 Markdown 相对链接与锚点、Shell 命令语法、试验配置与实际文件 SHA-256 的对应、阶段二 CSV 空值以及 Git 暂存文件大小和凭据模式均已核查。TASK-001 历史文件与固定上游跟踪文件均无差异。正式评价日志与正式指标的全量对账须等待训练和冻结评价完成。

## 2026-11-22 首次正式交付快照

以下为审查前记录。Dinomaly/Doll 在该快照中为 RECOVERED_DEVIATION。当前正式结果见下方最终整改。

八组训练及官方 Synergy 评价均通过。最后一组 INP-Former / Ring 于北京时间 2026-11-21 23:17:56 完成训练，23:28:31 完成评价。所有正式训练均含 100 epoch、30,000 步。独立复查仍待完成。[证据索引](../results/baselines/README.md) · [PR #2](https://github.com/xinnan369/anomaly/pull/2)

### 四类别官方指标和资源

| 类别 | 模型 | O-AUROC (%) | I-AUROC (%) | AUPRO (%) | 训练（分钟） | CUDA 峰值已分配（GiB） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Car | Dinomaly | 90.30 | 85.52 | 92.53 | 110.24 | 3.08 |
| Cube | Dinomaly | 90.10 | 80.45 | 93.27 | 120.86 | 3.08 |
| Doll | Dinomaly | 81.70 | 79.74 | 92.11 | 116.52 | 3.08 |
| Ring | Dinomaly | 95.80 | 86.34 | 87.72 | 103.37 | 3.08 |
| Car | INP-Former | 82.20 | 72.11 | 86.03 | 101.21 | 2.80 |
| Cube | INP-Former | 87.50 | 77.23 | 91.85 | 123.50 | 2.80 |
| Doll | INP-Former | 73.20 | 75.77 | 86.26 | 111.83 | 2.80 |
| Ring | INP-Former | 70.20 | 70.75 | 94.41 | 112.75 | 2.80 |

四类等权宏平均的 O-AUROC、I-AUROC、AUPRO 分别为 Dinomaly **89.48、83.01、91.41** 和 INP-Former **78.28、73.97、89.64**。Dinomaly 在四类的两项 AUROC 均较高。Ring 的定位 AUPRO 由 INP-Former 领先 **6.69 个百分点**。这些结果来自单个 seed（42）。训练分钟数来自记录的训练时间，分配显存来自 PyTorch，机器整卡峰值另列于总表。Dinomaly/Doll 训练时间仅合计保存检查点之前与成功续跑区段。[指标与资源总表](../results/baselines/metrics/baseline-summary.csv) · [指标图](../results/baselines/figures/official-metrics.png) · [资源图](../results/baselines/figures/training-resources.png)

### 视角、光照和改进机会

- **类别难度随模型和指标变化。** Dinomaly 在 Doll 的两项 AUROC 最低。INP-Former 在 Ring 的两项 AUROC 最低，但同一类别的 AUPRO 为该模型四类最高。物件判别与像素定位应分别分析。[官方指标](../results/baselines/metrics/model-comparison.csv)
- **视角和光照之间有明显差异。** 例如 INP-Former/Ring 的融合视角 AUROC 在 180° 为 62.67%，在 120° 为 80.30%。它的单光照诊断在 03 号为 59.38%，在 07 号为 77.13%。Dinomaly/Doll 的 120° / 01 号光照为 59.92%，330° / 09 号为 90.75%。这说明视角和光照对应的诊断难度不同，支持把跨光照预测作为待测方向。以上诊断基于不同分组，直接比较时应注意每组的掩膜阳性比例。[视角图](../results/baselines/figures/view-diagnostics.png) · [光照图](../results/baselines/figures/illumination-diagnostics.png) · [1136 条诊断](../results/baselines/metrics/diagnostics.csv)
- **正常件高分倾向与融合限制。** 八个模型与类别组合中，07 号有 4 组、10 号有 3 组取得组内最高的正常件分数 P95。P95 描述排序中的高分尾部，本次未选择分类阈值。固定平均融合对各视角和光照等权。诊断差异提供了测试加权和鲁棒聚合的理由，其收益仍需在固定划分上以同一模型做对照验证。热力图按预定规则选取有掩膜样本，展示局部差异。[逐光照正常分数](../results/baselines/metrics/normal-scores-by-light.csv) · [热力图](../results/baselines/figures/reference-heatmaps.png)

### 数据、恢复和核验

四类完整 ZIP 通过固定发布 SHA-256、成员 CRC 和清单覆盖验证，训练与测试产品身份无交叉。正式组每类为 3,600 张训练图和 8,400 张测试图，评价得到 70 个物件和 840 个视角组。发布时重新核对四个归档验证凭据和磁盘文件长度，复算固定模型配置、八个训练检查点、八组原始及融合数组的 SHA-256，并对照官方评价 JSON、8 份最终状态和日志。原始大文件留在仓库外，运行索引记录路径、长度、哈希和日志节选。[四类归档凭据](../results/baselines/data_audit/verified-archives.json) · [运行与哈希索引](../results/baselines/metrics/run-index.json)

Dinomaly/Doll 的原任务在第 11,241 步因非有限损失中断。原失败日志与检查点保留。恢复使用第 37 epoch、第 11,100 步的官方检查点，复用模型、Adam 与调度状态。第 11,101–11,400 步通过数值诊断。检查点没有保存随机数与数据加载器状态，恢复运行可能采用不同的样本顺序，短跑未复现异常，首次异常根因未知。最终检查点依然为第 100 epoch、第 30,000 步。[恢复凭据](../results/baselines/evidence/doll-recovery.json) · [原样保存的恢复脚本](../results/baselines/evidence/doll-recovery-script.py.txt) · [失败和续跑日志节选](../results/baselines/evidence/dinomaly-Doll.log)

复现沿用已核验的固定版本和环境。通常执行 `CUDA_VISIBLE_DEVICES=0 <锁定的 Python> -B -u tools/environment/run_stage2.py --root /mnt/d/vicad-syn/task-002-stage2`。Doll 按恢复脚本在单独的外部尝试目录保留失败记录后继续，评价仍由原启动器执行。全部 1136 条诊断均有限且保留掩膜与清单标签分歧计数。RTX 4090 未验证，Holder 差异仍按 TASK-001 记录。[环境与命令](../tools/environment/README.md#第二阶段完整基线)

本次发布前在 Ubuntu 22.04 / WSL2 的锁定环境中复跑 `python -B -m unittest discover -s tools/environment -p "test_*.py"` 和 `python -B -m unittest discover -s tools/audit -p "test_*.py"`，分别通过 7 项和 9 项测试。此轮核验未重新训练模型。

## 原版 Doll 连续复验与前次整改

以下保留前次整改记录。当前结论见[第二阶段最终结果](#第二阶段最终结果)。

**TASK-002 最终整改 PARTIAL。** 7/8 组连续训练和冻结评价有通过记录，最终整改存在阻塞。本轮对应 [Review 6009448364](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-6009448364)，审查基准提交为 `f224a27e8020c0d98100acd082196846f1cc761b`。第一阶段独立验收记录继续有效。[PR #2](https://github.com/xinnan369/anomaly/pull/2)

### Dinomaly/Doll 干净训练、数值诊断与复评

旧运行在第 11,241 步报告非有限损失。历史续跑从第 37 epoch、第 11,100 步的官方检查点恢复模型、Adam 和调度器。该检查点缺少 RNG 和采样顺序，旧结果保留为 `RECOVERED_DEVIATION`。[历史指标与来源](../results/baselines/metrics/doll-recovered-history.csv) · [原始中断和恢复证据](../results/baselines/evidence/doll-recovery.json)

本轮在 WSL2、RTX 3090、PyTorch 2.2.2+cu121 上，使用固定发布 05abc5d 和上游 a606aa0、原始 DINOv2 ViT-B/14 reg4 骨干、官方随机初始化的可训练模块、seed 42、batch 12、3,600 张正常图和官方损失、优化器、调度及变换。启动脚本首次因 Windows 工作树的 Git 路径和 WSL 不兼容在训练前退出。保留该启动日志，之后用新目录启动唯一一轮干净训练。启动器为正式评价和只读复评预留了独立目录，实际因训练失败未运行。官方训练期间采用原仓库既有的空评估路径，正式指标只从最终冻结权重计算。[运行命令](../tools/environment/README.md#dinomalydoll-连续复验和正式出图) · [实际运行记录](../results/baselines/evidence/doll-clean.json)

训练状态为 FAILED，北京时间 2026-11-22 10:33:14 停止。从训练启动到退出 45.90 分钟，PyTorch 峰值已分配 3.08 GiB。最后记录的迭代为 11240，完成 37 个 epoch。核验的唯一实验阻断为 Dinomaly/Doll 连续训练失败，影响最终检查点、正式评价、复评及 Doll 跨模型核对。

新训练于第 11,241 步检测到首个非有限阶段 `decoder_features`，该步输入和编码器特征均有限，解码后的 `de.0` 首先出现非有限值。观察器在计算损失、反向和更新前停止，保存问题张量名、批次路径和 RNG。第 11,240 步是最后一次成功更新。一次限定范围的诊断重放逐批匹配 141 个批次，并触发同一迭代和阶段的异常。

第 7 个解码块 `model.decoder.6.attn` 的点积归一化分母在零基坐标（批次索引 10、注意力头 4、token 112）精确为 0，对应 `Doll/Good/025/A300_I08.png`。取倒数后产生了 1 个非有限值，下一解码块同一样本的分母已有 3,132 个非有限值。这定位了故障的算术起点。该位置两个操作数为何形成零点积仍待查。稳定化分母会改变官方前向计算，本次保持原配置并记录阻断。诊断重放在目标步停止。[固定上游 LinearAttention2](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/model/dinomaly_utils/vision_transformer.py#L261-L276) · [诊断重放证据](../results/baselines/evidence/doll-forensic-replay.json)

批次日志记录官方 DataLoader 的实际送入顺序。RNG 保存主进程 Python、NumPy、Torch CPU/CUDA 的开跑与 epoch 边界状态。上游启用了 cuDNN benchmark，未开启确定性算法。该材料用于诊断，未验证跨 GPU 或任意中断点的逐位复现。[量化日志节选](../results/baselines/evidence/doll-clean.log)

未取得第 100 epoch 的干净检查点，正式冻结评价和复评未启动。干净 Doll 指标留空。[运行和哈希索引](../results/baselines/metrics/run-index.json) · [核验 JSON](../results/baselines/metrics/verification.json)

### 正式基线结果

| 类别 | 模型 | 训练来源 | O-AUROC (%) | I-AUROC (%) | AUPRO (%) | 训练（分钟） | 峰值已分配（GiB） |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Car | Dinomaly | `CLEAN_CONTINUOUS` | 90.30 | 85.52 | 92.53 | 110.24 | 3.08 |
| Cube | Dinomaly | `CLEAN_CONTINUOUS` | 90.10 | 80.45 | 93.27 | 120.86 | 3.08 |
| Doll | Dinomaly | `FAILED` | — | — | — | — | 3.08 |
| Ring | Dinomaly | `CLEAN_CONTINUOUS` | 95.80 | 86.34 | 87.72 | 103.37 | 3.08 |
| Car | INP-Former | `CLEAN_CONTINUOUS` | 82.20 | 72.11 | 86.03 | 101.21 | 2.80 |
| Cube | INP-Former | `CLEAN_CONTINUOUS` | 87.50 | 77.23 | 91.85 | 123.50 | 2.80 |
| Doll | INP-Former | `CLEAN_CONTINUOUS` | 73.20 | 75.77 | 86.26 | 111.83 | 2.80 |
| Ring | INP-Former | `CLEAN_CONTINUOUS` | 70.20 | 70.75 | 94.41 | 112.75 | 2.80 |

旧续跑三项分数依次为 81.70 / 79.74 / 92.11，仅保留作 RECOVERED_DEVIATION 历史记录。

- 在 Car、Cube、Ring 三类已通过核验的配对中，Dinomaly 的物件 O-AUROC 与融合视角 I-AUROC 均较高。
- 像素定位 AUPRO 在 Car、Cube 的 Dinomaly 较高，在 Ring 的 INP-Former 较高。Doll 的严格配对待补。

跨光照预测和可靠性加权仍是待验证假设。Doll 的连续复现结果待补时，四类严格宏平均和正式论文图表均缺少完整输入。前次图表仍作为历史快照保留。最终论文图表待八组正式核验通过。

### 交叉核验和留存

依据固定发布清单检查训练与测试产品身份及图像路径隔离。官方源码、配置、骨干、最终检查点、训练状态、评价命令、官方 JSON、CSV、预测数组与诊断按组交叉核验。现存评价按组核对原始样本身份、视角、光照、标签和掩膜。缺少新结果的组合在核验 JSON 中列为未通过。已验收的四类 ZIP 用发布 SHA-256、CRC 凭据和当前文件长度复核，完整 ZIP 未在本轮重复哈希。其余七组无需再次训练，表中原有数值逐字段核对。[机器可读核验](../results/baselines/metrics/verification.json)

完整文件保存在仓库外 `D:/vicad-syn/task-002-stage2`。新 Doll 文件位于 `doll-clean-20261122-run`，历史模型和评价仍在既有路径。哈希索引给出七组正式检查点、模型导出、原始与融合数组及运行日志的相对路径与 SHA-256。Doll 专项索引记录第 37 epoch 检查点、批次日志、随机状态和定向重放产物。[Doll 文件索引](../results/baselines/evidence/doll-clean.json)数据集、完整权重和大体积预测只保留在外部。RTX 4090 仍未实测。Holder 发布差异维持 TASK-001 的已验收调查记录。正式结果只有 seed 42。

本轮在锁定 WSL2 环境中通过 13 项环境与新旧回归测试、9 项审计测试和 CUDA 官方融合身份回归。全量磁盘复核使用 Windows 原生 CPU Python，以避免同一批大数组在 WSL 挂载盘上的页读取开销。核验 JSON 记录运行版本。检查点绑定、哈希、指标、文件结构与失败事项由分组核验 JSON 记录。提交前检查 Markdown 相对链接、Git 差异、提交范围、文件大小和凭据模式。[单元测试及核验节选](../results/baselines/evidence/finalization.log) · [CUDA 合成回归](../results/baselines/environment/final-identity-regression.json)

## 第二阶段最终结果

**TASK-002 最终研究闭环 · ACCEPTED_WITH_EXCEPTION / 有限例外验收通过。** 七组原版已复核，原版 Dinomaly/Doll 保留故障结论，稳定版完成冻结评价及复评。[审查记录 6011031984](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-6011031984) · [PR #2](https://github.com/xinnan369/anomaly/pull/2)

### 原版 Doll 的数值故障

固定 seed 42、RTX 3090 和 WSL2 条件下，两次原版从头训练在第 11,241 步失败。有界回放重新匹配第 37 epoch 后的 141 个批次。故障批次的预处理图像和编码器特征有限。故障前模型参数、100 个梯度张量和 100 份 Adam 状态也均有限。[完整张量与回放证据](../results/baselines/evidence/doll-zero-origin.json)

第 7 个解码块的 `LinearAttention2` 在 FP32 分母张量 `[12, 12, 261]` 的零基坐标 `[10, 4, 112]` 上得到精确零，对应 `Doll/Good/025/A300_I08.png`。原始 q 的 64 个值在 −53.49781 至 −4.75195 范围内。计算 `ELU(x)+1` 时有 62 维舍入为零。k 的 261 个 token 求和后有 6 维为零。两者的正值维度无重叠，取分母倒数产生 1 个非有限值。用同一批 q/k 的原始数值在 FP64 中直接计算负分支 `exp(x)`，得到参考分母 3.008708381e-08。本次故障来自有限负输入在 ELU 计算和随后的加一中丢失小量。该结论对应实测配置与轨迹。[固定上游表达式](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/model/dinomaly_utils/vision_transformer.py#L261-L276)

### Dinomaly-stabilized 与冻结评价

在正式训练前固定 `epsilon = 1e-6`。运行时仅将上游倒数改为 `1.0 / denom.clamp_min(1e-6)`。数据发布、初始化、优化器、损失、学习率调度、变换、种子、batch size 和 epoch 沿用官方设置。使用独立目录，从原始骨干和官方随机初始化开始。GPU 预检使用实际上游模块，安全分母的输出与梯度逐位相同，零分母输入的输出和梯度有限。补丁模块 SHA-256 为 `b7b1d74efd5f0e7e839ad0b8da5dbf5910f504d135cd04842adbfa0be211b4fa`，实际 forward 为 `dd646ff916104659c90f79f54f12824d5616018b2a0947e3ebe67b87b57d1cb9`，上游模块为 `f972c48485e930438ae38962c0fa6289170dfd6088b8b9e5704f33544ca6cd13`。检测到原始分母为负或非有限时，启动器中止更新并保留计数和批次日志。[补丁实现](../tools/environment/stabilized_attention.py) · [实际 forward](../results/baselines/evidence/doll-stabilized-forward.py.txt)

北京时间 2026-11-22 17:48:49 完成 100 epoch 和 30,000 次更新。训练用时 109.41 分钟，PyTorch 峰值已分配 3.09 GiB，峰值预留 3.54 GiB，整卡采样峰值 6,240 MiB。主评价推理至拼接耗时 241.15 秒。第 100 epoch 检查点 SHA-256 为 `9250e4b8162d5eec1bc972925dc815ab04b744729c7b3af8a5876ea074dd5b61`。

训练阶段保护逻辑触发 80 个位置，其中精确零 1 个，全部发生于第 11,241 步的第 7 个解码块。前 11,241 个实际批次路径及其顺序与原版逐批相同。

正式评价和只读复评各覆盖 8,400 张图、70 个物件组、840 个融合视角、142 条诊断。十项指标逐项精确一致。9 个原始数组逐字节一致。8 个融合文件因官方组顺序变化而具有不同哈希，按物件和视角身份对齐后 10 个融合数组逐项完全一致。两次评价使用同一冻结检查点，复评后重新计算了哈希。Doll 稳定版与 INP-Former 的测试身份、标签和掩膜一致。[稳定版诊断](../results/baselines/metrics/stabilized-doll-diagnostics.csv) · [复评数组对齐](../results/baselines/evidence/doll-stability-repeat-order.json)

稳定版 Doll 的 O-AUROC / I-AUROC / AUPRO 分别为 **82.50 / 79.79 / 92.18**（%），原版 INP-Former 为 **73.20 / 75.77 / 86.26**（%），差值分别是 **+9.30 / +4.02 / +5.91** 个百分点。该比较分别描述物件判别、视角判别和像素定位，且两组使用不同算法版本。[Doll 比较图](../results/baselines/figures/doll-variant-metrics.pdf) · [稳定版 CSV](../results/baselines/metrics/stabilized-doll.csv) · [保护计数与证据](../results/baselines/evidence/doll-stability-run.json)

### 七组原版结果与论文口径

每格按 O-AUROC / I-AUROC / AUPRO（%）排列。带分数的七格均为 `OFFICIAL_CLEAN`。

| 类别 | 原版 Dinomaly | 原版 INP-Former |
| --- | --- | --- |
| Car | 90.30 / 85.52 / 92.53 | 82.20 / 72.11 / 86.03 |
| Cube | 90.10 / 80.45 / 93.27 | 87.50 / 77.23 / 91.85 |
| Doll | OFFICIAL_FAILED，指标留空 | 73.20 / 75.77 / 86.26 |
| Ring | 95.80 / 86.34 / 87.72 | 70.20 / 70.75 / 94.41 |

Doll 原版故障行保持 `OFFICIAL_FAILED`。旧续跑的三项分数为 81.70 / 79.74 / 92.11，身份为 `RECOVERED_DEVIATION`，用于追溯。稳定版单列为 `NUMERICALLY_STABILIZED`。Car、Cube、Ring 的原版物件和视角判别由 Dinomaly 得分较高。定位 AUPRO 在 Car、Cube 由 Dinomaly 得分较高，在 Ring 由 INP-Former 得分较高。四类完整原版宏平均因缺少一组而无定义。固定种子和设备下的结果不建立跨设备或多种子稳定性。[原版论文图](../results/baselines/figures/official-only-metrics.pdf) · [原版正式表](../results/baselines/metrics/baseline-summary.csv) · [旧续跑结果](../results/baselines/metrics/doll-recovered-history.csv)

### 核验、留存和结项建议

重新从磁盘核验七组原版 checkpoint、133 个预测数组和官方指标。版本、清单、训练与测试产品隔离、评价 checkpoint 绑定及各组状态见机器核验。WSL2 的 19 项环境和回归测试、9 项审计测试及 CUDA 官方融合身份回归通过。RTX 4090 尚未验证。完整文件保存在 `D:/vicad-syn/task-002-stage2`，稳定版目录为 `doll-stabilized-20261122`。日志、RNG、批序、检查点及预测文件的路径和 SHA-256 可查。原始训练数据、模型权重和预测数组留在外部存储。[机器核验](../results/baselines/metrics/verification.json) · [运行索引](../results/baselines/metrics/run-index.json) · [结果及图表](../results/baselines/README.md)

建议对 TASK-002 作有限例外结项。七组原版和 Doll 的可复现原版故障给出清楚的证据边界，稳定版另外验证了在相同固定配置下，预先固定的分母下界可以支撑完整训练与评价。正式验收同意该有限例外范围。[验收记录](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-6018691109)
