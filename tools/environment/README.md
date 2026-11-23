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

汇总可在正式训练过程中执行，未达到完整评价的指标留空。正式绘图要求八组均通过连续训练及冻结评价。视角和光照表里的 `positive` 表示掩膜非空的样本，`metadata_positive` 和 `mask_label_disagreements` 分别记录原元数据计数与分歧。给定单一掩膜标签的子组保留空指标。正常件分数按光照汇总，用于比较高分倾向；没有在测试集上设定误报阈值。热力图在每类两模型共同的正式输出中按身份顺序选首个有官方标签与非空掩膜的样本，每类两图同尺度。

当前证据见[审查整改与正式实验](../../reports/TASK-002-REPORT.md#第二阶段审查整改与正式实验)和[结果目录](../../results/baselines/README.md)。

### Dinomaly/Doll 连续复验和正式出图

Review [6009448364](https://github.com/xinnan369/anomaly/pull/2#pullrequestreview-6009448364) 要求从原始骨干开始补做一轮，不加载旧 Doll 检查点。其余七组复用原记录。旧结果保存在外部原位，在交付索引中标记为 RECOVERED_DEVIATION。[原失败和恢复证据](../../results/baselines/evidence/doll-recovery.json)

在 WSL2 中，从仓库根目录执行以下命令。为 attempt 指定一个仓库外的全新路径。脚本依次训练、冻结评价及对同一检查点只读复评。任一步失败即停，保留日志、退出码与现有产物。[本轮实际调用脚本](../../results/baselines/evidence/doll-clean-launch.sh)

```bash
set -euo pipefail
stage=/mnt/d/vicad-syn/task-002-stage2
py="$HOME/.local/venvs/vicad-task-002-stage2/bin/python"
attempt="$stage/doll-clean-new"
launch_status=0
bash results/baselines/evidence/doll-clean-launch.sh "$PWD" "$attempt" "$(git rev-parse HEAD)" || launch_status=$?

"$py" -B tools/environment/summarize_baselines.py --runs "$stage/runs" \
    --evaluations "$stage/evaluations" --doll-attempt "$attempt" --output "$attempt/baseline-summary.csv"
verification_status=0
"$py" -B tools/environment/verify_baselines.py --root "$stage" \
    --doll-attempt "$attempt" --summary "$attempt/baseline-summary.csv" --output "$attempt/verification" || verification_status=$?
if (( launch_status == 0 && verification_status == 0 )); then
    "$py" -B tools/environment/plot_baselines.py --root "$stage" \
        --doll-attempt "$attempt" --output "$attempt/figures"
fi
```

观察器当步记录官方 DataLoader 实际输出路径，监测预处理张量、编码器和解码器输出、损失、梯度裁剪前后、Adam 动量和更新后的参数。遇到非有限值写出阶段、张量名、批次、迭代及随机状态后中断。RNG 快照包括主进程 Python、NumPy、PyTorch CPU 和各 CUDA 设备，在首次采样前及每个 epoch 边界保存。固定配置的图像变换为确定性变换，DataLoader 使用默认生成器和非持久 worker。样本顺序单独记录。RNG 快照仅用于诊断。基线遵循上游的 cuDNN 配置，GPU 算子的逐位确定性仍未保证。

只读复评在独立目录写出相同格式的官方指标、预测和诊断，验证脚本重算文件 SHA-256、验证评估组身份与跨模型掩膜/标签，逐项比较两次指标，并复核检查点 SHA-256。旧归档通过既有的 SHA-256 与 CRC 凭据检查，本轮验证器复核凭据及当前 ZIP 大小。[结果和外部文件索引](../../results/baselines/README.md)

### Doll 数值故障诊断

本轮第 11,241 步在解码器特征出现异常，训练中断。用本轮自己的第 37 epoch 检查点和对应 RNG 状态进行一次有界重放，逐步核对实际批次路径，重放到发生异常的步为止。这是诊断运行，产物不进入正式指标。第 7 个解码块中实际观测到线性注意力归一化分母为零。任何修改分母的保护方案须作为独立的实验偏差评审。[诊断结果](../../results/baselines/evidence/doll-forensic-replay.json)

```bash
attempt="$stage/doll-clean-20261122-run"
CUDA_VISIBLE_DEVICES=0 "$py" -B -u tools/environment/diagnose_doll_failure.py \
    --root "$stage" --attempt "$attempt" --output "$attempt/forensic-new"
```

最终核验另在 Windows 原生 CPU Python 中执行。传入同一数据根、尝试目录和 CSV 的 Windows 路径即可。JSON 记录了实际 Python、PyTorch 和 NumPy 版本。训练和官方评价环境仍为 WSL2。[最终核验](../../results/baselines/metrics/verification.json)

### Dinomaly-stabilized 数值稳定性实验

原版 Dinomaly 继续使用默认 `--attention-mode official`。稳定版只接受 Doll，显式指定 `--attention-mode stabilized`，训练时另加 `--diagnose-doll`。正式开始前先确认原版有界回放的 q/k 舍入证据，再在真实固定上游模块上运行数值预检。分母下界固定为 1e-6，启动器记录实际注入 forward、补丁代码、上游源码和运行设置的哈希。两种模式的检查点不能交叉评价。完整批次计数保存在各阶段的 `attention-guards.json`，训练事件按 epoch 和解码层归档，评价事件以批次序号为准。[数值证据与外部文件索引](../../results/baselines/evidence/doll-stability-run.json)

从 WSL2 的仓库根目录执行，使用已锁定的 Python 和固定数据根。源文件按 LF 保存以核对本轮执行的字节级 SHA-256。故障证明若已生成，可直接复用。实际启动脚本调用两次冻结评价，每次使用独立目录，每次均验证最终检查点和补丁绑定。

```
stage=/mnt/d/vicad-syn/task-002-stage2
py="$HOME/.local/venvs/vicad-task-002-stage2/bin/python"
official_attempt="$stage/doll-clean-20261122-run"
attempt="$stage/doll-stabilized-new"
if [ ! -f "$official_attempt/forensic-qk/forensic-replay.json" ]; then
    CUDA_VISIBLE_DEVICES=0 "$py" -B -u tools/environment/diagnose_doll_failure.py \
        --root "$stage" --attempt "$official_attempt" --output "$official_attempt/forensic-qk"
fi
bash tools/environment/run_doll_stabilized.sh "$stage" "$attempt" "$(git rev-parse HEAD)" "$py"
```

验证器可以从 Windows 原生 CPU Python 使用同一数据路径，以降低 WSL 挂载盘的数组扫描开销。先用 `verify_baselines.py` 核验原版总表，再将其结果交给变体验证器。已知原版 Doll 失败使第一条核验返回 2，仍会输出七组成功和已知阻断。变体验证器审查原版已知阻断清单、变体的完全训练证据及两次评价。它输出单独的变体表和机器核验 JSON，稳定版不具备完整证据时保留空指标。以下是使用 WSL 路径的等价核验命令。

```
"$py" -B tools/environment/verify_baselines.py --root "$stage" \
    --doll-attempt "$official_attempt" --summary results/baselines/metrics/baseline-summary.csv \
    --output "$attempt/official-audit"
"$py" -B tools/environment/verify_doll_variant.py --root "$stage" --attempt "$attempt" \
    --official-summary results/baselines/metrics/baseline-summary.csv \
    --official-audit "$attempt/official-audit" --output "$attempt/variant-audit"
"$py" -B tools/environment/plot_stability_results.py \
    --official results/baselines/metrics/baseline-summary.csv \
    --variant "$attempt/variant-audit/stabilized-doll.csv" --output "$attempt/paper-figures"
```
