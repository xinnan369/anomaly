#!/usr/bin/env bash
set -euo pipefail
repo=$1
attempt=$2
root=/mnt/d/vicad-syn/task-002-stage2
python=/home/xinnan/.local/venvs/vicad-task-002-stage2/bin/python
cd "$repo"
mkdir "$attempt"
exec > >(tee -a "$attempt/pipeline.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$attempt/exit-code"; date -Is' EXIT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
printf "%s\n" "$3" > "$attempt/parent-commit"
sha256sum tools/environment/{run_baseline,training_diagnostics,baseline_metrics}.py > "$attempt/launcher.sha256"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$attempt/gpu-environment.csv"
args=(--upstream "$root/inputs/upstream" --release-root "$root"
      --weight "$root/inputs/weights/dinov2_vitb14_reg4_pretrain.pth" --model dinomaly --category Doll)
set -x
"$python" -B -u tools/environment/run_baseline.py "${args[@]}" \
    --output "$attempt/train" --diagnose-doll > "$attempt/train.log" 2>&1
checkpoint=$("$python" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); print(root / json.loads((root/"status.json").read_text())["final_checkpoint"]["path"])' "$attempt/train")
for name in evaluation re-evaluation; do
    "$python" -B -u tools/environment/run_baseline.py "${args[@]}" \
        --checkpoint "$checkpoint" --output "$attempt/$name" > "$attempt/$name.log" 2>&1
done
