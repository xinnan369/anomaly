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

在 WSL 执行以下命令。可使用同一训练命令加 `--preflight` 及指向独立目录的 `--output`，仅用已经生成的官方清单构建数据加载器、模型和评估器，不读取图像。

```bash
set -euo pipefail
stage=/mnt/d/vicad-syn/task-002-stage2
py="$HOME/.local/venvs/vicad-task-002-stage2/bin/python"
upstream="$work/upstream"
mkdir -p "$stage/logs"
for category in Car Cube Doll Ring; do
  test -f "$stage/data_audit/$category.json"
done
for model in dinomaly inpformer; do
  for category in Car Cube Doll Ring; do
    CUDA_VISIBLE_DEVICES=0 "$py" -u -B tools/environment/run_baseline.py \
      --model "$model" --category "$category" --upstream "$upstream" \
      --release-root "$stage" --weight "$weight" --output "$stage/runs/$model/$category" \
      2>&1 | tee "$stage/logs/$model-$category.log"
  done
done

# All eight training commands must have succeeded before evaluating.
shopt -s nullglob
for model in dinomaly inpformer; do
  for category in Car Cube Doll Ring; do
    checkpoints=("$stage/runs/$model/$category"/checkpoints/*/*_ckpt.pth)
    test "${#checkpoints[@]}" -eq 1
    CUDA_VISIBLE_DEVICES=0 "$py" -u -B tools/environment/run_baseline.py \
      --model "$model" --category "$category" --upstream "$upstream" \
      --release-root "$stage" --weight "$weight" --checkpoint "${checkpoints[0]}" \
      --output "$stage/evaluations/$model/$category" \
      2>&1 | tee "$stage/logs/$model-$category-evaluate.log"
  done
done
"$py" -B tools/environment/summarize_baselines.py --runs "$stage/runs" --evaluations "$stage/evaluations" \
  --output "$stage/metrics/baseline-summary.csv"
```

已有输出目录时启动器拒绝覆盖。续跑时从首个未完成组合开始，保留旧目录及日志。若评价失败，继续使用同模型、同类别的原冻结检查点，用新 `--output` 目录执行评价命令。脚本核对最终 epoch、迭代数、实验配置并记录检查点哈希；汇总时保留原训练与复评两份证据。若训练尚未完成 100 epoch，保留失败证据，用新输出目录从相同种子和参数重跑。记录显存、报错或中断原因之后再重试。

每组训练目录记录 `config.json`、`status.json`、`epochs.json`、`gpu.csv`、官方日志及权重。独立评价目录含 `synergy.json`、`raw/`、`fused/` 和 `diagnostics.csv`。第 50 和第 100 epoch 的训练内评估沿用上游空评估路径，官方测试集只用于冻结检查点的独立评价。关闭逐图批量渲染，保存可审计的原始数组。融合、评估输入和指标实现见 [TASK-001 审计](../../reports/TASK-001-REPORT.md)。

当前证据见 [第二阶段执行记录](../../reports/TASK-002-REPORT.md#第二阶段执行记录) 和 [结果目录](../../results/baselines/README.md)。
