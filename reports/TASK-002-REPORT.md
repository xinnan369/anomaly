# TASK-002 · M²AD 官方基线复现与实验环境固化

- 日期　2026-11-19（北京时间）
- 范围　第一阶段环境固化和单卡 Car smoke test
- 状态　第一阶段通过试运行，等待 ChatGPT 验收

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

- 单次机械试运行只用于检查数据到优化和推理路径的连通性，不能用于方法排名、准确率、收敛性或吞吐测试。正式 Car Synergy 的全配置评估、其它类别及其它基线需待阶段一验收。
- 本机未验证 RTX 4090、上游 README 的原始整套环境和 INP-Former 端到端执行。软件锁记录本次实际 Python 环境，系统驱动和 Linux 运行时另列。
- 保留 TASK-001 确认的发布数量差异。论文 v1 图 3(a) 比固定发布多 1 个 Holder 异常实例，不能据此补造数据。 [TASK-001 报告](TASK-001-REPORT.md)
