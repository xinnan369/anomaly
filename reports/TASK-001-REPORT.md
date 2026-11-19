# TASK-001 · M²AD-Synergy 数据集与实验协议审计

- **状态**　COMPLETED - AWAITING CHATGPT REVIEW
- **审计日期**　2026-11-19（北京时间）
- **审计范围**　官方论文、固定版本代码及发布清单、本地样本抽查和环境验证。

## 1. 任务执行摘要

M²AD-Synergy 的主协议同时评估**物件级异常检测**和**单视角异常检测与定位**。官方基线把同一物件的全部图像标量取均值用于 O-AUROC，把同一物件、同一角度下的光照分数及热图分别取均值用于 I-AUROC 和 AUPRO。定位热图始终保留各自视角的坐标系。

固定发布清单包含 **998 个实例、119,760 张图**，每实例均有 12 个视角 × 10 种光照。论文和项目主页称有 999 个实例、119,880 张图，两者相差一个完整实例的数量。当前复现实验应以有版本的发布清单为准，同时保留这项差异待核实。Car、Cube、Doll、Ring 均适合做首轮协议验证。当前 Windows 主机有 24 GiB 的 RTX 3090，WSL 中成功执行了 CUDA 张量和合成 AUPRO 检查。

| # | 核查项 | 审计结论 |
| --- | --- | --- |
| 1 | 官方论文及仓库 | 已核实引用、代码提交及数据发布版本 |
| 2 | 真实目录 | 已解析官方清单并对照 ZIP 中央目录，抽查覆盖见第 3 节 |
| 3 | 视角、光照及索引 | 已核实全部清单实例均具有完整 12×10 配置 |
| 4 | 同视角光照对齐 | 论文与配对文件支持同一视角坐标系，已抽样测全局平移，逐图及局部误差待测 |
| 5 | 数据划分 | 已核实仅 train/test、按正常实例拆分、异常实例全在 test |
| 6 | 实例 ID 与泄漏 | 已核实清单完整键、路径交集及部分归档的 CRC 检查，物理来源待核实 |
| 7 | O-AUROC、I-AUROC、AUPRO | 已核实代码路径、聚合方式及 FPR 上限，运行了融合和 ADEval 合成测试 |
| 8 | 四个候选类别 | 已核实类别、样本数及标签覆盖，适合做第一阶段复现子集 |
| 9 | 官方基线及依赖 | 已核实五种方法、配置和权重入口，端到端运行未验证 |
| 10 | 分辨率、空间与 GPU | 已核实官方规格、样本和发布大小，峰值显存待实测 |
| 11 | 3090/4090 兼容性 | 3090 本地 CUDA 和 ADEval 已实测，4090 为官方运行证据与规格推断 |
| 12 | 可复现环境建议 | 已给出可隔离的版本组合、顺序和最小检查，完整锁文件待环境验证 |

## 2. 官方数据来源和论文引用

- 期刊版：Yunkang Cao, Yuqi Cheng, Yuxin Zhang, Xiaohao Xu, Yiheng Zhang, Yihan Sun, Yuxiang Tan, Xiaonan Huang, Chao Huang, Weiming Shen. *Visual anomaly detection under complex view-illumination interplay: A large-scale benchmark*. **Pattern Recognition 179**, 113666 (2026). DOI 10.1016/j.patcog.2026.113666。期刊元数据由 Crossref 核对，期刊全文入口返回 HTTP 403。协议文字具体核对自下面的预印本。 [期刊 DOI](https://doi.org/10.1016/j.patcog.2026.113666) · [Crossref 记录](https://api.crossref.org/works/10.1016/j.patcog.2026.113666)
- 预印本：Yunkang Cao, Yuqi Cheng, Xiaohao Xu, Yiheng Zhang, Yihan Sun, Yuxiang Tan, Yuxin Zhang, Xiaonan Huang, Weiming Shen. 同题，arXiv:2505.10996v1（2025-05-16）。§3、§4.1 和附录 A.2 分别用于核对采集、评估与光照设置。 [arXiv 元数据](https://arxiv.org/abs/2505.10996v1) · [预印本全文](https://arxiv.org/html/2505.10996v1)
- 作者主页与官方代码：`hustCYQ/M2AD` 提交 `a606aa0e1614f23dad70da26430dd66095192141`（2026-04-05 UTC）。融合、指标、数据读取和基线结论均指这个版本。 [项目主页](https://hustcyq.github.io/M2AD/) · [固定代码](https://github.com/hustCYQ/M2AD/tree/a606aa0e1614f23dad70da26430dd66095192141)
- 官方 README 直接链接的发布源为 `ChengYuQi99/M2AD`，固定 revision `05abc5d03167354a7ef98541236b5c0bd00abadf`。`jsons.zip` 为 1,100,022 字节，SHA-256 为 `bbeaa75c231c981e443c818859a701fcc0daf1769065abd91631f7e676725980`。 [官方 README](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/README.md) · [数据发布文件树](https://huggingface.co/datasets/ChengYuQi99/M2AD/tree/05abc5d03167354a7ef98541236b5c0bd00abadf)

## 3. 数据目录结构

**公开发布物的本地检查。** Hugging Face 文件树中有 10 个按类命名的 ZIP 和一个 `jsons.zip`。元数据归档内只有 `meta_unsupervised.json`（53,127,977 字节展开）。官方 loader 要求按下例布局，压缩包中清单文件本身没有 `jsons/` 前缀。下例正常 PNG 已按范围解码，异常与掩膜路径来自固定清单。

```text
<data_root>/
  jsons/meta_unsupervised.json
  Car/
    Good/000/A000_I01.png
    NG/hole_1_000/A150_I01.png
    GT/hole_1_000/A150_mask.png
    GT/hole_1_000/A150_seg.png
  Cube/
  Doll/
  Ring/
  ...（其余 6 类）
```

视角字段为字符串 `000, 030, …, 330`，对应 12 个 30° 间隔。光照字段为 `01, …, 10`，是离散工况 ID，对应预印本图 10 的四条线光源与一个同轴环光源的组合。目录名 `Good`/`NG` 标记物件状态。文件名模板为 `A{view}_I{illumination}.png`，`GT` 的掩膜与分割文件按物件和角度命名，无光照后缀。官方 loader 对 `image_anomaly=0` 生成零掩膜，对正例读取 `mask_path` 并二值化。物件为 NG 时，部分视角仍可没有可标注缺陷。 [数据读取](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/data/ad_dataset.py#L28-L108) · [清单生成](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/data/gen_metadata/m2ad.py#L33-L87) · [预印本附录](https://arxiv.org/html/2505.10996v1#A1)

**空间对齐。** 预印本 §3.1 描述固定夹具、单相机和旋转台，§4.1 明确按同一角度融合 10 张空间对齐图像。官方 GT 文件名和融合代码也采用同视角坐标。抽样固定各类第一个训练正常实例 `Good/000`、`view=000`，比较 `I01→I05` 与 `I01→I10`。对 1024² 灰度图计算 Sobel 梯度幅值，使用 Hann 窗相位相关。见第 8 节日志。位移是基于内容的全局估计，光照变化会改变响应，抽样不足以确认局部像素精度、所有视角的对齐质量或跨视角对应关系。 [采集与协议](https://arxiv.org/html/2505.10996v1#S4.SS1)

## 4. 数据划分与评估协议

**固定发布清单实测。** `train` 有 299 个正常实例和 35,880 张图。`test` 有 200 个正常实例、499 个异常实例和 83,880 张图。清单没有 `val`。生成代码将每类排序后的正常实例以 Python `random.seed(42)` 打乱，取前 `int(N*0.6)` 作为训练，剩余正常加全部异常进入测试。实际应消费已发布清单，避免不同释放版本或重跑结果改变划分。Dice 的训练 Good 为 29，Holder 的测试 NG 为 49，其余对应为 30 和 50。采集论文 §3.1 用于筛选可见性的三个内部子集与这里的无监督 train/test 分工不同。Synergy 读取全部测试记录，只有 Invariant 过滤 `detectable=False`。 [划分生成代码](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/data/gen_metadata/m2ad.py#L91-L145) · [协议选择](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/data/ad_dataset.py#L43-L52)

| 类别 | 训练 Good 实例 / 图像 | 测试 Good / NG 实例 | 测试图像 | NG 中 `image_anomaly=1` | 其中 `detectable=False` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Car | 30 / 3,600 | 20 / 50 | 8,400 | 4,850 | 309 |
| Cube | 30 / 3,600 | 20 / 50 | 8,400 | 5,530 | 222 |
| Doll | 30 / 3,600 | 20 / 50 | 8,400 | 5,010 | 280 |
| Ring | 30 / 3,600 | 20 / 50 | 8,400 | 5,480 | 105 |

四类均有完整组合、正常与异常测试物件、定位标注路径以及预印本两档分辨率的五种基线结果。首轮可报告四类各自结果与四类宏平均，完整榜单比较需使用十类相同协议。样本数量按独立物件理解，重复配置之间高度相关。 [预印本表 2–3](https://arxiv.org/html/2505.10996v1#S4)

**由官方实现确定的指标。** 令 `s(o,v,l)` 为每张图的标量预测，`A(o,v,l)` 为同视角坐标内的热图，`M(o,v,l)` 为二值真值。trainer 固定调用 `m2ad_fuse(..., fuse='add')`，其中 `add` 的实际操作是算术平均。官方 evaluator 按类计算，trainer 将结果乘 100，多类时取宏平均。

| 指标 | 样本单位与默认规则 |
| --- | --- |
| O-AUROC | 一物件一分数，`mean(v,l) s(o,v,l)`，完整配置为 120 张图。真值为 `object_anomaly`，用 sklearn `roc_auc_score`。实现以类别、物件 ID 和物件状态分组。 |
| I-AUROC | 一物件一视角一分数，`mean(l) s(o,v,l)`。先对同视角的逐光照真值取像素最大值形成联合掩膜，视角级真值为该联合掩膜是否非空，再用 sklearn `roc_auc_score`。evaluator 从掩膜重算视角标签。 |
| AUPRO | 对同一视角的 10 张热图逐像素求均值，对照同视角联合掩膜的连通区域。默认 ADEval CUDA 路径指定 `nstrips=50`，用背景像素 FPR 和按区域等权的 PRO 曲线，插值积分到 FPR 0.30 后除以 0.30。ADEval 1.1.0 已审阅并验证合成例，官方仓库未固定 ADEval 版本。 |

三项的预测输入由各基线 trainer 提供单图分数和热图，视角级标量不是融合热图的重新池化。代码另外记录 `AUROC_view_px`，它是像素 AUROC。关闭 ADEval 时，仓库回退方法取 `FPR < 0.3`，再对保留 FPR 做 min-max 重标并积分，和默认后端数值不能直接视作相同。官方代码按物件状态分组只用于评估对齐，VICAD-Syn 的推理和融合应由不含标签的实例键驱动。 [官方融合](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/util/m2ad_utils.py) · [指标代码](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/util/metric.py#L146-L339) · [汇总入口](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/trainer/_base_trainer.py#L362-L399) · [ADEval 1.1.0](https://pypi.org/project/ADEval/1.1.0/)

## 5. 数据泄漏风险分析

- **清单实测。** 以 `(cls_name, Good/NG 路径段, object_name)` 为实例键，train/test 交集为 0，图像相对路径交集为 0。当前发布清单中 `(cls_name, object_name)` 的跨集交集也为 0，仅按裸 `object_name` 有 8 个交集。每个实例的 120 个角度与光照记录必须一起进入同一划分。四类的 ZIP 中央目录检查范围及 CRC32+成员长度结果见第 8 节。
- **证据边界。** 路径、作者实例 ID 和 ZIP CRC32+长度可筛查命名泄漏和完全相同的归档成员。它们无法证明 Good 与 NG 没有同一实体改制、近重复或同一批次相关性。正式前需以可用的物理溯源或受控的近重复审计为证据。
- **校准与调参。** 从官方 train 按实例建立正常验证集，预先锁定划分、 seed、预处理、截断阈值、光照权重、epoch 与统计量。异常率、`image_anomaly`、`object_anomaly`、`detectable`、GT 和 `Good`/`NG` 路径状态都不能进入训练、推理、权重计算或模型选择。盲推理清单应通过脱敏映射提供 `(class, opaque_id, view, light)`，同时避免由 `hole_...` 等缺陷命名泄露标签。正常校准和训练阶段生成的代理扰动是一种拟议实验设计。
- **官方运行器防误用。** trainer 会展示每个指标历次测试的 Max 及对应 epoch。预先固定次数和检查点，只在冻结后运行官方测试。若将测试 Max 用来选 epoch、方法、权重或分辨率，会产生测试集选择偏差。 [trainer 日志与汇总](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/trainer/_base_trainer.py#L372-L398)

## 6. 基线方法分析

官方基线为 CDO、RD++、MSFlow、Dinomaly、INP-Former。固定代码的五个配置均设 seed 42 与 100 epochs。下列批量为配置中 train/test 的默认值，实际显存需按卡与后端测量。

| 方法 | 默认 train/test batch | 依赖及权重入口 |
| --- | --- | --- |
| CDO | 12 / 12 | 知识蒸馏，配置活跃分支用 `mshrnet32` teacher，`yacs`，HRNet32 ImageNet 权重由 `model/cdo.py` 的 URL 加载。注释中的 WideResNet checkpoint 不属于活跃配置。 |
| RD++ | 8 / 8 | 反向蒸馏，`timm_wide_resnet50_2` ImageNet teacher 和 `de_wide_resnet50_2` decoder。 |
| MSFlow | 24 / 24 | 多尺度流，`FrEIA` 和代码内 WideResNet50-2 的 ImageNet 预训练入口。 |
| Dinomaly | 12 / 12 | 重建方法，`dinov2reg_vit_base_14`，注册 token 版 Meta DINOv2，编码器下载或读取缓存的 `dinov2_vitb14_reg4_pretrain.pth`。 |
| INP-Former | 12 / 12 | 原型重建，同为 DINOv2 ViT-B/14 reg4，配置 `INP_num=6`。 |

README 列出 Python 3.9、`timm==0.8.15dev0`、`opencv-python==4.9.0.80`、NumPy 1.26 和 PyTorch 2.1.2/cu118，并列出其它通用与可选依赖。需按实际入口检查 `scikit-learn`、`scipy`、`scikit-image`、`numba`、ADEval、`yacs`、`FrEIA` 等导入。配置引用权重来源不表示已在本地下载或校验；固定跑分前应记录每个实际 checkpoint 的 SHA-256。DINOv2 reg4 与非 reg4 权重应区别记录并在骨干匹配对照中统一。 [基线配置目录](https://github.com/hustCYQ/M2AD/tree/a606aa0e1614f23dad70da26430dd66095192141/configs/benchmark) · [模型代码](https://github.com/hustCYQ/M2AD/tree/a606aa0e1614f23dad70da26430dd66095192141/model)

## 7. 环境配置与兼容性

- **本地实测。** Windows 11 Pro 10.0.26200，PowerShell 7.6.9，Git 2.53.0.windows.2，Python 3.13.2。RTX 3090 有 24,576 MiB，驱动 610.88，Windows `nvcc` 为 12.6.85。Windows Python 的 torch 2.11.0、torchvision 0.26.0 环境是 CPU build（`torch.version.cuda=None`），缺 sklearn 和 timm。WSL2 Ubuntu 22.04 的已有 Conda 环境使用 Python 3.11.11、torch 2.2.2+cu121，CUDA 可用，设备 capability 8.6，张量求和结果 28，ADEval 合成测试通过。
- **硬件与数据规格。** NVIDIA 列出 RTX 3090 为 CC 8.6，RTX 4090 为 CC 8.9。预印本报告单 RTX 4090、torch 2.1.2，但本机没有 4090。作者使用相机 3648×5472，README 列出 1024²、256² 发布版本，论文实验用 256/512，Dinomaly 与 INP-Former 用 224/448。公开四类 ZIP 共 67,347,000,143 B（62.72 GiB），十类 ZIP 共 164,429,881,188 B（153.14 GiB），不含元数据 ZIP。C、D、E 盘检查时分别空闲 358.33、371.53、410.62 GiB。 [NVIDIA capability](https://developer.nvidia.com/cuda/gpus) · [官方分辨率和资源](https://arxiv.org/html/2505.10996v1#S4.SS1)
- **资源估算。** 四类 ZIP 中央目录的解包成员总大小为 67,422,456,045 B。保留四类的 ZIP 和解包文件共约 125.51 GiB，再计文件系统开销、缓存和日志。可预留四类约 150 GiB、十类约 350 GiB，十类值为按压缩量粗估。未压缩单物件 120×3×H×W 的 FP32 输入约为 68.9 MiB（224²）或 275.6 MiB（448²），不含激活、梯度和权重。官方 tester 会先累积单图热图和掩膜，再融合。按一类 8,400 张和 256²、FP32 热图及 8 字节整数掩膜估算，仅原始两组数组约 6.15 GiB RAM，拼接和融合另计。GPU 峰值还受 forward batch、融合实现及逐类评估缓存影响。建议先单类、按视角分组流式处理，读取 `max_memory_allocated`、`max_memory_reserved` 和主机 RSS，再确定安全批量。

**建议的隔离顺序（尚未安装完整环境）。** 在 Linux 或 WSL2 新建 Python 3.9 环境。先装官方匹配的 CUDA wheels，再独立装数值/指标依赖，最后对照所选方法配置核实其余导入。仓库 README 的 `--index-url` 指向 PyTorch wheel 仓库，该索引仅用来装 torch/torchvision。采用 NumPy 1.26.4 让 ADEval 1.1.0 的 `np.trapz` 路径和旧依赖可用。下列命令只供后续获准的环境复现，无数据与权重下载。 [PyTorch 2.1.2 安装版本](https://pytorch.org/get-started/previous-versions/#v212) · [官方依赖说明](https://github.com/hustCYQ/M2AD/blob/a606aa0e1614f23dad70da26430dd66095192141/README.md#environment-setup)

```bash
conda create -n m2ad-audit python=3.9.21 pip=25.0.1 -y
conda activate m2ad-audit
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 opencv-python==4.9.0.80 Pillow==10.4.0 \
  scipy==1.13.1 scikit-learn==1.5.2 scikit-image==0.24.0 numba==0.60.0 adeval==1.1.0
python -m pip install timm==0.8.15.dev0 yacs==0.1.8 FrEIA==0.2 fvcore==0.1.5.post20221221
python -m pip check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, torchvision, cv2, sklearn; from adeval import EvalAccumulatorCuda; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_capability(0)); assert torch.arange(8, device='cuda').sum().item() == 28"
python -m pip freeze --all
```

开始基线复现前，额外固定官方代码、发布 revision、每个方法的依赖及权重哈希。将清单放在 `<data_root>/jsons/`，对照仓库的读写路径配置。全库入口会遍历十类，四类子集需要显式限制类列表。PyPI 已确认上述 timm、yacs、FrEIA 和 fvcore 版本存在。2.1.2/cu118 组合和完整基线导入在本机尚未验证。

## 8. 已完成的验证及其日志证据

以下记录本次实际执行。数据抽样与合成数组为固定选择，没有随机种子。官方划分与基线配置中的种子 42 来自源码，本次未运行拆分生成器或训练。

| 实际命令或调用 | 结果 |
| --- | --- |
| `git fetch origin main`、`git switch -c codex/task-001-m2ad-audit origin/main` | 从 `797639dc9c2bba3b8cad86a4feb1df457a364a3f` 创建独立分支 |
| `git ls-remote https://github.com/hustCYQ/M2AD.git HEAD`、`gh api repos/hustCYQ/M2AD/git/trees/<SHA>?recursive=1`、对固定 SHA 的 Raw 文件用 `curl.exe -fsSL` 获取源码 | 固定代码 SHA 如第 2 节。初始 shallow clone 因包含大体积官网 GIF 取消，静态审计改为逐个获取小源码文件 |
| `Invoke-RestMethod` 查询 Crossref 标题及 HF 数据集 API，`curl.exe -fsSL --max-time 45 --max-filesize 1500000 <jsons.zip URL>`，`Get-FileHash`，`tar -tf` | DOI、发行 revision、清单文件和哈希见第 2 节 |
| `python -X utf8 -u audit_release.py <TEMP_DIR> --archives`，同参数运行 `audit_release_retry.py` 和 `audit_release_http.py`（临时脚本，zipfile + OpenCV 4.13.0.92） | 对 TLS 或传输截断错误重试，并改用 Python requests 流式验证 206、Content-Range 和长度。全量核对清单配置、物件键、路径和标签 |
| `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`、`nvcc --version`、`wsl --list --verbose`，Windows Python 查询 `torch.version.cuda`，WSL Python 执行 `torch.arange(8, device=torch.device(0)).sum().item()` | GPU、驱动、CUDA 和结果见第 7 节 |
| `python -X utf8 metric_smoke.py`，WSL 的 `~/anaconda3/envs/yolov12/bin/python metric_smoke.py --cuda` | 官方融合函数和 ADEval CUDA 合成测试通过。权重和模型未参与 |
| `Invoke-RestMethod https://pypi.org/pypi/adeval/1.1.0/json`、用 `curl.exe` 获取 26,044 B 的发布 wheel、`Get-FileHash`、`python -m zipfile -e` | SHA-256 `125caf991037a668a1f072f7e99c81b6bdf86dc8c30cf1699c731d4b79e0bd6e` 与 PyPI 一致，合成测试直接读取解包源码 |
| `rg --files --hidden --no-ignore --maxdepth 4` 按 M2AD/VICAD 和清单文件名筛查，查看候选数据目录，WSL `find ~ /mnt/d/datasets -maxdepth 4` 按相关名筛查 | 工作区、常用下载/文档目录、D/E 盘有限深度及 WSL 家目录未发现可用的完整已有 M²AD。结论限于查找范围 |

```text
release manifest: train=299 objects/35880 images; test=699 objects/83880 images
configurations: 998/998 objects have exactly 12 views x 10 illuminations
train ∩ test: full keys=0; paths=0; class + ID=0; bare ID=8
synthetic fusion: object a=4, object b=5/3, view a_000=2, union mask OK
synthetic ADEval CUDA: p_aupro=1.0, i_auroc=1.0, p_auroc=1.0
```

合成融合输入的标量是 `[1,3,8,2,1,2]`，前后三张分别属于物件 a、b。各物件前两张为 `view=000`，第三张为 `030`。前两张掩膜各有一个不同位置的像素，逐像素并集断言通过。将前两张的 image label 人为设为 0、1 时，官方函数产生不一致警告但继续返回。ADEval 输入是两组 2×2 热图及一个正像素，预测最高值 0.9 落在该像素，`nstrips=50`。只填 image/pixel 累加器，未填的 sample 累加器返回 NaN。以上均为合成测试结果。

**复核方法。** 用 `zipfile.ZipFile(...).open("meta_unsupervised.json")` 读取清单，将各 `img_path` 拆成类别、状态、实例和文件名，用正则 `A(\d{3})_I(\d{2})\.png` 提取索引并与字段比较。按完整实例键断言配置集合等于 `{000,030,…,330} × {01,…,10}`，分别构建 train/test 的实例键、简化键及路径集合，取交集。ZIP 审计只请求存档尾部 65,557 B、ZIP64 中央目录所需区段和选定 PNG 成员的压缩区段。请求按字节数设上限，核对区间或返回长度，解码时由 zipfile 核对成员 CRC，比较 train/test 归档成员的 `(CRC32, file_size)`。

**归档实测日志。** 6 个 ZIP 的文件成员均直接以类名为首层。每个 ZIP 的清单图像、非空 mask/seg 路径缺失数和未列成员数均为 0。以下仅列计量成功的 Range 响应，重试及失败流量另计。Dice 和 Holder 用于交叉检查清单例外，二者的归档目录也未发现补齐这 120 张差异的额外成员。

| ZIP | 文件成员数 | 解包成员字节数 | 跨集 CRC32+长度交集 | 成功读取范围字节数 |
| --- | ---: | ---: | ---: | ---: |
| Car | 12,970 | 16,347,837,495 | 0 | 5,606,488 |
| Cube | 13,106 | 17,642,663,742 | 0 | 6,063,770 |
| Doll | 13,002 | 16,853,767,408 | 0 | 5,784,283 |
| Ring | 13,096 | 16,578,187,400 | 0 | 5,630,765 |
| Dice | 12,778 | 16,707,805,277 | 0 | 1,490,511 |
| Holder | 12,834 | 17,495,460,545 | 0 | 1,522,539 |

**对齐抽样日志。** 四类共 12 张样图均为 1024×1024 PNG。每类仅取 `Good/000/A000_I{01,05,10}.png`，以 I01 为参照。8 组平移估计的两个分量绝对值均小于 0.5 像素。最低响应为 0.1565，显示部分光照组合的估计置信度有限。

| 类别 | I01→I05 `(dx, dy; response)` | I01→I10 `(dx, dy; response)` |
| --- | --- | --- |
| Car | `(0.431, 0.006; 0.3465)` | `(0.262, 0.071; 0.5824)` |
| Cube | `(0.096, -0.105; 0.4629)` | `(0.023, 0.017; 0.6504)` |
| Doll | `(-0.001, 0.166; 0.1565)` | `(-0.026, 0.060; 0.6551)` |
| Ring | `(0.087, 0.051; 0.2696)` | `(0.130, -0.038; 0.6969)` |

## 9. 未完成事项和阻塞问题

1. 需要作者确认论文 999/119,880 与当前固定发布 998/119,760 差异的具体来源。清单本身一致，缺口的物理身份和原论文跑分所用版本未知。
2. 完整数据集未在所查本地路径找到。在未获取该版本全量像素、采集标定与实物溯源记录的情况下，逐图可解码性、亚像素局部对齐、跨视角配准和同一实物的近重复泄漏均未验证。
3. README 提到低分辨率发行版，当前未验证其下载资产、清单一致性和实际空间占用。4090、官方 2.1.2/cu118 组合、五个端到端基线以及正式峰值显存未在本地验证。源码说明缺少完整依赖锁及原论文评估依赖版本。
4. 已检查当前分支、仓库历史、项目章程和可访问的项目聊天记录。找到四项设计假设和对旧数学方案的提及，但没有可逐式核对的数学原稿或公式版本。第 10 节对可核实的设计假设给出协议审查，原稿级验收须取得原稿后开展。

## 10. 对 VICAD-Syn 研究方案的修正建议

以下是基于官方协议的检查和待验证修正，未实现或训练任何模型。

| 既有方向 | 与官方协议的匹配条件和修正 |
| --- | --- |
| 光照条件化特征预测 | 用 `x(o,v,l)` 明确区分物件、角度、类别和光照 ID。光照 `01–10` 表示离散工况；不能从编号大小推断光强或用差值解释相似性。用同物件同视角下的其他光照预测目标特征时，阻断目标图和同一目标 token 经任一上下文支路回流。训练只使用官方正常 train 的物件。 |
| 双上下文预测稳定性 | 同视角跨光照可在相同空间网格比较，两个上下文均要排除目标。跨视角只有同物件关系及角度顺序可直接使用，局部像素无天然一一对应，应比较视角级或池化表征。在使用局部跨视角一致性前，需要独立几何映射与遮挡验证。 |
| 预测不确定性 | 分开记录预测残差与模型估计的不确定性。可靠程度的训练、尺度与边界应在训练或按实例划出的正常校准集上确定。真实缺陷可能只在少数照明出现，分析高不确定度是否会抑制这些响应。 |
| 可靠性加权异常融合 | 以等权均值复刻官方基线。如要比较 `sum_l w_l A(o,v,l)`，保持权重非负、同视角聚合且归一，规定低可靠度回退。权重只能使用图像、配置或冻结预测统计。物件级另用跨视角标量聚合，同一视角的定位图独立留在其坐标系。加权融合是候选方法，应与公开均值对照并沿用同一真值和评价单位。 |

第一轮优先固定四类、完整 12×10、一个明确的 DINOv2 版本和输入规格。224² 与 448² 对应 ViT-B/14 的 16×16 与 32×32 patch 网格。若需要其他分辨率，应明确对齐 resize/crop、图像与掩膜插值规则及局部坐标转换。同方案输出每样本的评分、融合版本、实例划分及配置哈希，便于后来对照。等权与加权、同视角和跨视角上下文应做可区分的控制；四类结果明确标为子集。
