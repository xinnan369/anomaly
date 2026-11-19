# TASK-002 第一阶段环境与 Car 试运行

目标环境为 WSL2 Ubuntu 22.04 x86_64、Python 3.11.11、RTX 3090。命令在仓库根目录执行，所有源码检出、图像、权重和 Python 环境都放在仓库之外。复现只使用官方 Car 训练划分的三个 PNG 和一个预训练骨干权重。

## 建立环境

```bash
work="$HOME/vicad-task-002"
mkdir -p "$work"
python3.11 -m venv "$work/venv"
py="$work/venv/bin/python"
"$py" -m pip --isolated install --index-url https://pypi.org/simple pip==25.0.1
"$py" -m pip --isolated install --index-url https://pypi.org/simple -r tools/environment/requirements-wsl2.lock
"$py" -m pip check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

锁文件固定了实际环境的全部 Python 发行包版本。PyTorch 从 PyPI 提供 Linux CUDA 依赖，实际 build 和 GPU 信息见报告。系统驱动由 Windows/WSL 提供，无需向仓库添加 CUDA 工具包。当前仅在 RTX 3090 验证。

## 固定源码、数据和权重

```bash
rev=a606aa0e1614f23dad70da26430dd66095192141
git clone --depth 1 --filter=blob:none --sparse https://github.com/hustCYQ/M2AD.git "$work/upstream"
git -C "$work/upstream" fetch --depth 1 origin "$rev"
git -C "$work/upstream" checkout --detach "$rev"
git -C "$work/upstream" sparse-checkout set configs data model util trainer loss optim
git -C "$work/upstream" diff --exit-code HEAD --

"$py" -B tools/environment/prepare_car.py --cache "$work/cache" --output "$work/car"

mkdir -p "$work/weights"
weight="$work/weights/dinov2_vitb14_reg4_pretrain.pth"
url=https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
sha=73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71
curl --fail --location --retry 3 --max-filesize 346393545 "$url" --output "$weight"
printf '%s  %s\n' "$sha" "$weight" | sha256sum --check
```

源端下载不稳定时，可在 Windows 下载并验证文件后从 WSL 读取挂载路径。图像提取也可先在 Windows 运行，之后将 `--data` 指向 WSL 挂载路径。Windows 只需 requests，即可运行数据准备脚本。

## CUDA 试运行

```bash
CUDA_VISIBLE_DEVICES=0 "$py" -B tools/environment/smoke_dinomaly.py \
  --upstream "$work/upstream" --data "$work/car" --weight "$weight" --weight-sha256 "$sha"

CUDA_VISIBLE_DEVICES=0 "$py" -B tools/audit/metric_smoke.py --cache "$work/cache" --device cuda
```

Car 试运行使用官方 loader、Dinomaly 配置、模型、损失、优化器和推理打分函数，执行一个更新和一次留出正常实例推理。脚本断言权重哈希、两个实例分离、模型参数发生更新、编码器无梯度、推理输出有限。启动脚本为 Python 3.11 在进程内补齐上游旧的 `collections.Iterable` 别名。不会保存训练权重或调用官方测试集及评估入口。独立 ADEval 命令只使用合成数组核查 CUDA 兼容性。

## 第二阶段完整基线

在仓库根目录执行。第二阶段重新安装全部 87 项锁定发行包，复验两种模型，并使用发布版四类完整数据。下方路径是本次 D 盘的挂载方式，可替换为有足够空间的仓库外目录。第一阶段 venv 独立保留。

### 新建环境并复验

```bash
set -euo pipefail
stage=/mnt/d/vicad-syn/task-002-stage2
venv="$HOME/.local/venvs/vicad-task-002-stage2"
python3.11 -c 'import sys; assert sys.version_info[:3] == (3, 11, 11)'
test ! -e "$venv"
python3.11 -m venv --copies "$venv"
py="$venv/bin/python"
"$py" -c 'from importlib import metadata; assert {p.metadata["Name"].lower() for p in metadata.distributions()} == {"pip", "setuptools"}'
mkdir -p "$stage/environment"
"$py" -m pip --isolated --disable-pip-version-check install --index-url https://pypi.org/simple \
  --timeout 90 --retries 5 -r tools/environment/requirements-wsl2.lock | tee "$stage/environment/install.log"
"$py" -m pip check | tee "$stage/environment/pip-check.txt"
"$py" -m pip freeze --all > "$stage/environment/freeze.txt"
diff -u <(grep -v '^#' tools/environment/requirements-wsl2.lock | tr -d '\r') "$stage/environment/freeze.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

实测安装以现有 Conda 环境中的 Python 3.11.11 可执行文件替代 `python3.11`，在新 venv 中安装，并通过初始包集合断言检查隔离。用上文的固定版本源码、Car 正常训练小样本和 Meta 权重设置 `work`、`weight`、`sha`。本机把已核验的三个输入保留在 `$stage/inputs`，可把 `work` 设为该目录。依赖重装独立进行。

```bash
for model in dinomaly inpformer; do
  CUDA_VISIBLE_DEVICES=0 "$py" -B tools/environment/smoke_dinomaly.py --model "$model" \
    --upstream "$work/upstream" --data "$work/car" --weight "$weight" --weight-sha256 "$sha" \
    | tee "$stage/environment/$model-smoke.json"
done
CUDA_VISIBLE_DEVICES=0 "$py" -B tools/audit/metric_smoke.py \
  --cache "$work/cache" --device cuda | tee "$stage/environment/adeval-cuda.json"
```

### 下载与复核

从 Windows PowerShell 执行。此脚本在 Windows 侧用 Python 3.11 或更高版本和 `requests`，下载与数据校验无需 GPU。一次只运行一个下载进程。重跑可复用已完整下载的分块；每类在校验发布 SHA-256、CRC 和清单覆盖后才写入该类的就绪记录。

```powershell
$stage = 'D:/vicad-syn/task-002-stage2'
New-Item -ItemType Directory -Force $stage | Out-Null
python -B -X utf8 -u tools/environment/prepare_release.py --root $stage --workers 10 |
    Tee-Object -FilePath "$stage/download.log" -Append
if ($LASTEXITCODE) { exit $LASTEXITCODE }
```

若 SHA-256 或 ZIP 与清单的比对失败，先保留失败日志并核对发布版本。重新下载损坏的文件前将它移到隔离位置。四份 ZIP 合计 62.72 GiB，解包后约 62.79 GiB。另需为组装、冻结检查点、预测数组和日志预留空间。归档和输出保存在仓库外。

### 按官方配置正式运行

在 WSL 的仓库根目录执行。当前项目使用 Python 3.11.11 的锁定 venv，数据和输入位于仓库外。上次传输中断后只续传 Ring，保留其已有完整分块。

```powershell
$stage = 'D:/vicad-syn/task-002-stage2'
python -B -X utf8 -u tools/environment/prepare_release.py --root $stage --classes Ring --workers 10 2>&1 |
    Tee-Object -FilePath "$stage/download-retry-ring.log" -Append
if ($LASTEXITCODE) { exit $LASTEXITCODE }
```

以下先复验评价及冻结检查点约束。正式计划先运行 Car/Dinomaly 的完整训练与评价，再运行其余七组。每组开始前要求该类存在就绪记录；训练使用官方 100 epoch、batch 12 和 seed 42。检查点经过配置、权重、路径和结果记录校验，再执行 Synergy 评价。

```bash
set -euo pipefail
stage=/mnt/d/vicad-syn/task-002-stage2
py="$HOME/.local/venvs/vicad-task-002-stage2/bin/python"
CUDA_VISIBLE_DEVICES=0 "$py" -B tools/environment/check_baseline_metrics.py \
  --upstream "$stage/inputs/upstream" --cache "$stage/cache"
"$py" -B -m unittest discover -s tools/environment -p 'test_*.py'
"$py" -B -m unittest discover -s tools/audit -p 'test_*.py'
CUDA_VISIBLE_DEVICES=0 "$py" -B -u tools/environment/run_stage2.py --root "$stage"
```

可用 `--only-car` 单独执行首组。启动脚本会为同一数据根加锁，按顺序执行训练、冻结评价和核查。第 50 与第 100 epoch 训练内的测试位置均保留上游空评估路径。各组正式测试从 `runs/<模型>/<类别>/checkpoints/<上游实验目录>/['类别']_ckpt.pth` 读取，只接受 `status.json` 绑定的最终检查点及 `epoch=100`、`iter=30000`。备用的 `*_net_100.pth` 记录 SHA-256，用于溯源。

完整日志在 `logs/`。日常文件包括训练目录的 `config.json`、`status.json`、`epochs.json`、`gpu.csv`，以及独立评价目录的 `synergy.json`、`predictions.json`、`raw/`、`fused/` 和 `diagnostics.csv`。训练中的整卡采样和最终 PyTorch 峰值分别记录。

若评价失败，先将已停止的失败评价目录移动到仓库外的另一个保留目录，再运行相同计划。已通过的训练会复用，评价仍绑定同一冻结检查点。若训练中断或失败，保留日志和整个失败运行目录。当前启动器从固定种子重新执行该组，因为上游检查点没有保存完整随机数和数据加载器状态。通过的前序组合会复用。重试时原日志保存在外部目录，新的执行时间和命令追加到日志。

```bash
"$py" -B tools/environment/summarize_baselines.py --runs "$stage/runs" \
  --evaluations "$stage/evaluations" --output "$stage/metrics/baseline-summary.csv"
"$py" -B tools/environment/plot_baselines.py --root "$stage" --output "$stage/figures"
```

汇总可在正式训练过程中执行，未达到完整评价的指标留空。绘图命令要求八组均通过。视角和光照表里的 `positive` 表示掩膜非空的样本，`metadata_positive` 和 `mask_label_disagreements` 分别记录原元数据计数与分歧。给定单一掩膜标签的子组保留空指标。正常件分数按光照汇总，用于比较高分倾向；没有在测试集上设定误报阈值。热力图在每类两模型共同的正式输出中按身份顺序选首个有官方标签与非空掩膜的样本，每类两图同尺度。

当前证据见[审查整改与正式实验](../../reports/TASK-002-REPORT.md#第二阶段审查整改与正式实验)和[结果目录](../../results/baselines/README.md)。
