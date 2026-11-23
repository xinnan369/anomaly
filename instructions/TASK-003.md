# TASK-003 · VICAD-Syn 第一版

基线来自 TASK-002 的有限例外验收。原版 Dinomaly/Doll 记录失败，稳定性变体和历史续跑各有独立身份。[TASK-002 报告](../reports/TASK-002-REPORT.md)

## 预先固定的执行协议

- 固定发布版、上游源码和 DINOv2 原始权重见[配置](../vicad_syn/configs/experiment.json)。公开权重保留在仓库外。
- 每个类别将 30 个正常训练物件的目录身份匿名化，按 seed 42 与身份的 SHA-256 排序。前 24 个用于正常开发训练，后 6 个仅用于正常预测损失检查。开发 3 epoch。按照相同配置从头用全部 30 个正常物件训练 12 epoch。固定最后一个 epoch。
- 先运行单元测试、真实正常 Car 的 CPU/CUDA 梯度烟测和 CUDA 官方融合身份回归。再跑 Car 的完整训练和评价。Car 通过后顺序跑 Cube、Doll、Ring。
- Car 预定 A0 至 A4。Cube、Doll、Ring 预定 A1 和 A4。A1–A4 共用各类别条件预测器和冻结检查点，A0 单独训练去除源、目标光照嵌入的头。A0/A1 都采用等权。A2、A3、A4 分别只使用稳定性、只使用认知不确定性、使用两者加权。A0 仅 Car。首个正式指标前另外封存扩展优先级 Cube、Doll、Ring 的 A2/A3，复用条件预测器输出，按类别再按 A2、A3 执行。
- 对目标光照显式选择其他九种源光照。两类不确定性使用预先确定的采样来源。对象分组和前向均不读取测试异常标签或掩膜。匹配、掩膜和 Synergy 指标在冻结后的评价阶段使用。
- 日志、特征缓存、模型与预测数组写入外部根目录。中断时仅按匹配配置和 SHA 恢复。所有类别和已有结果均报告，包括失败或未完成项。
- 在同一 TASK-003 PR 提交代码、测试、指标、诊断、报告和全部状态文档。提交后等待独立验收，当前任务为 TASK-003。

[执行报告](../reports/TASK-003-REPORT.md) · [研究决策](../docs/RESEARCH_DECISIONS.md) · [结果索引](../results/vicad_syn/README.md)

## 运行与恢复

在 WSL2 仓库根目录运行，使用 TASK-002 固定环境和已完成校验的数据根。[环境锁文件](../tools/environment/requirements-wsl2.lock)

```bash
py="$HOME/.local/venvs/vicad-task-002-stage2/bin/python"
inputs=/mnt/d/vicad-syn/task-002-stage2
run=/mnt/d/vicad-syn/task-003-v1-new
mkdir -p "$run"
bash tools/vicad_syn/run.sh "$PWD" "$py" "$inputs" "$run"
```

恢复时保留原日志与退出码，使用匹配的源码、配置和实验根。缓存记录已写入的图像数，训练检查点记录 epoch、优化器、PyTorch 随机状态及训练物件绑定。可直接调用同一运行入口，将恢复日志输出至另一文件。

```bash
CUDA_VISIBLE_DEVICES=0 "$py" -B -u -m vicad_syn.cli run --inputs "$inputs" --output "$run" > "$run/resume.log" 2>&1
"$py" -B -m tools.vicad_syn.verify --inputs "$inputs" --run "$run" --output "$run/verified"
"$py" -B -m tools.vicad_syn.plot --inputs "$inputs" --run "$run" \
    --verification "$run/verified" --output "$run/figures"
CUDA_VISIBLE_DEVICES=0 "$py" -B -m tools.vicad_syn.benchmark --run "$run" --output "$run/uncertainty-cost.json"
```

实际磁盘核验使用 Windows 原生 CPU Python，以 Windows 路径传参。扩展 A2/A3 命令使用冻结产物、预先封存的优先级文件及官方 CUDA 评价。正式结果的全部路径与哈希见结果索引。


对于已完成核心核验的运行，可复用本次[预先封存的扩展计划](../results/vicad_syn/evidence/ablation-extension-plan.json)。扩展只读取冻结的残差及不确定性数组。

```bash
cp results/vicad_syn/evidence/ablation-extension-plan.json "$run/ablation-extension-plan.json"
CUDA_VISIBLE_DEVICES=0 "$py" -B -m tools.vicad_syn.extend --inputs "$inputs" --run "$run" \
    --verified "$run/verified/verification.json" --output "$run/extension"
"$py" -B -m tools.vicad_syn.diagnose --run "$run" \
    --verified "$run/verified/verification.json" --output "$run/localization.csv"
```
