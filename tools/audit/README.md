# 复查 TASK-001

从仓库根目录运行以下命令。缓存放在仓库外，可在 Windows 与 WSL 间共享。脚本输出只包含元数据统计、抽样测量和合成测试，范围读取的图像不落盘。

> 在线依赖安装与远程 Range 用法属于 TASK-001 历史记录，不可直接执行。现用入口只读取经过验证的本地缓存和 ZIP。[下载审批](../DOWNLOADS.md)

## 依赖

- 清单、归档目录和离线测试需要 Python 3.9+ 与 `requests`。
- 配准抽样另需 NumPy 和 OpenCV。本次目录与配准检查使用 Windows Python 3.13.2、requests 2.33.1、NumPy 2.3.5、OpenCV 4.13.0.92，可用下列命令准备同版本依赖。

```bash
python -m pip install requests==2.33.1 numpy==2.3.5 opencv-python==4.13.0.92
```

- 指标 smoke test 另需 PyTorch。CUDA 实测使用 WSL2 Ubuntu 22.04 中已有的 Python 3.11、PyTorch 2.2.2+cu121、NumPy 1.26.4 和 OpenCV 4.9.0。可使用已有环境，或按下列命令准备独立 CUDA 环境。ADEval 由脚本直接从校验通过的固定 wheel 加载。 [PyTorch wheel 索引](https://pytorch.org/get-started/previous-versions/#v222)

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install requests==2.33.1 numpy==1.26.4 opencv-python==4.9.0.80
```

## 固定输入

| 输入 | 版本及完整性检查 |
| --- | --- |
| 数据清单 | Hugging Face `ChengYuQi99/M2AD` revision `05abc5d03167354a7ef98541236b5c0bd00abadf`，`jsons.zip` SHA-256 `bbeaa75c231c981e443c818859a701fcc0daf1769065abd91631f7e676725980`。 |
| 10 个分类 ZIP | 使用相同 revision，HTTP 206 和 `Content-Range` 总长度须匹配固定文件大小。审计仅读取尾部、中央目录及可选的 PNG 成员，归档 SHA-256 没有整包实算。 |
| 官方融合 | `hustCYQ/M2AD` commit `a606aa0e1614f23dad70da26430dd66095192141` 的 `util/m2ad_utils.py`，SHA-256 `e22c05165b65542d41f7f45828b0eb641614a0f84a7347ec9c1d17e1b855f009`。 |
| ADEval | PyPI 通用 wheel 1.1.0，26,044 B，SHA-256 `125caf991037a668a1f072f7e99c81b6bdf86dc8c30cf1699c731d4b79e0bd6e`。 |
| 论文计数 | arXiv:2505.10996v1 图 3(a) 的 `#Object` 面板。类别计数人工转录，脚本计算与发布清单的差值。人工核对的 PNG SHA-256 `fd44ac54cdd15ea5ea26501ec3f23cfe67592ff8fcc0d8dcb50ff8b0a94bc0a6`。 [图 3 原图](https://arxiv.org/html/2505.10996v1/dataset_statistics.png) |

## 命令与断言

```bash
cache="/path/to/existing-verified-cache"
archives="/path/to/existing-verified-zips"
python -B -m unittest discover -s tools/audit -p 'test_*.py' -v
python -B tools/audit/manifest.py --cache "$cache"
python -B tools/audit/archives.py --cache "$cache" --archives "$archives" --alignment
python -B tools/audit/metric_smoke.py --cache "$cache" --device cpu
python -B tools/audit/metric_smoke.py --cache "$cache" --device cuda
```

PowerShell 中用 `$cache = Join-Path $env:TEMP 'm2ad-task-001'` 代替首行。若 WSL 网络不可用，先在 Windows 成功运行 CPU 命令，CUDA 命令用同一目录的 WSL 挂载路径，已有资源每次仍会校验 SHA-256。

1. **离线测试**预期 9 项通过。覆盖完整网格、缺组、重复配置、跨划分实例、官方划分不一致、错误缓存和 HTTP 范围/大小约束。
2. **清单**预期 `train=299/35880`、`test=699/83880`（实例/图像），共 `998/119760`。实例完整键、图像路径及类别+实例 ID 的跨划分交集为 0，裸 ID 交集为 8。每个实例恰有 12×10。对排序的每类正常 ID 以 seed 42 和比例 0.6 复算官方划分，训练 ID 对称差须为 0。与预印本图 3(a) 的物件数相比，唯一差项为 Holder NG（50→49）。查看[机器可读清单摘要](../../reports/evidence/TASK-001/manifest-summary.json)。
3. **归档**默认检查十类，可用 `--classes Car Cube Doll Ring` 只查指定类。当前仅接受本地归档，不发送 Range 请求。缺失清单路径、额外文件或跨集 CRC32+成员大小交集非零会报错。加 `--alignment` 后，仅对所选候选类各解码 3 张 1024² PNG，检查 ZIP 成员 CRC 和 PNG 解码，输出每类正常实例 `Good/000` 单一视角 `000` 的 I01→I05 和 I01→I10 相位相关位移及响应。响应是估计质量线索，数值可因 OpenCV 版本有小幅变化。
4. **指标**断言官方 `fuse='add'` 的物件分数为 4 和 5/3，四个视角分数为 2、8、1.5、2，热图均值和掩膜并集通过。ADEval 以 `nstrips=50` 对三个 2×2 样本运行，完美案例的 I-AUROC、像素 AUROC、AUPRO 均为 1.0，只有一区域正确定位的案例依次为 1.0、0.5、0.5。CPU 作为辅助检查，CUDA 是官方配置使用的后端。

只有划分复算使用 seed 42。图像抽样和指标输入由固定选择或合成数组给出，无随机种子。输出按[执行日志](../../reports/evidence/TASK-001/test-log.txt)标注实际环境和证据。失败时根据非零退出码和错误消息排查 `--cache`、网络 Range 支持或 CUDA。
