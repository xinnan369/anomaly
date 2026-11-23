#!/usr/bin/env bash
set -euo pipefail
repo=$1
py=$2
inputs=$3
out=$4
cd "$repo"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec >"$out/pipeline.log" 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit-code"; date -Is' EXIT
date -Is
"$py" -B -m unittest discover -s tests/vicad_syn -p 'test_*.py'
"$py" -B -m unittest discover -s tools/environment -p 'test_*.py'
"$py" -B -m unittest discover -s tools/audit -p 'test_*.py'
"$py" -B -m vicad_syn.cli smoke --inputs "$inputs" --output "$out"
"$py" -B tools/environment/check_baseline_metrics.py --upstream "$inputs/inputs/upstream" \
    --cache "$inputs/cache" > "$out/identity-synthetic.json"
"$py" -B -m vicad_syn.cli run --inputs "$inputs" --output "$out"
