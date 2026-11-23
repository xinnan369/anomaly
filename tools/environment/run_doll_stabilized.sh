#!/usr/bin/env bash
set -euo pipefail
root=$1
attempt=$2
parent=$3
py=$4
mkdir "$attempt"
cd "$(dirname "$0")/../.."
repo=$PWD
exec > >(tee -a "$attempt/pipeline.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$attempt/exit-code"; date -Is' EXIT
exec 9>"$root/logs/stage2.lock"
flock -n 9
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
printf "%s\n" "$parent" > "$attempt/parent-commit"
mkdir "$attempt/source"
cp tools/environment/{run_baseline,stabilized_attention,training_diagnostics,baseline_metrics,diagnose_doll_failure,check_doll_stability}.py "$attempt/source"
sha256sum "$attempt"/source/*.py > "$attempt/launcher.sha256"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$attempt/gpu-environment.csv"
"$py" -B tools/environment/check_doll_stability.py --upstream "$root/inputs/upstream" \
    --proof "$root/doll-clean-20261122-run/forensic-qk/forensic-replay.json" > "$attempt/preflight.json"
args=(--upstream "$root/inputs/upstream" --release-root "$root"
      --weight "$root/inputs/weights/dinov2_vitb14_reg4_pretrain.pth"
      --model dinomaly --category Doll --attention-mode stabilized)
set -x
"$py" -B -u tools/environment/run_baseline.py "${args[@]}" \
    --output "$attempt/train" --diagnose-doll > "$attempt/train.log" 2>&1
checkpoint=$("$py" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); print(root / json.loads((root/"status.json").read_text())["final_checkpoint"]["path"])' "$attempt/train")
for name in evaluation re-evaluation; do
    "$py" -B -u tools/environment/run_baseline.py "${args[@]}" \
        --checkpoint "$checkpoint" --output "$attempt/$name" > "$attempt/$name.log" 2>&1
done
