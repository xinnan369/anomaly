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
