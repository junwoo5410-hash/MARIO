#!/usr/bin/env bash
# Gated-fusion experiment: control and gated, four seeds each, then the comparison.
#
# Both arms run through train_gated.py, which takes its data, its checkpoint selection
# and its evaluation from train_blackbird_v2.py, so the fusion layer is the only
# difference. No extra tuning for the gated arm.
#
# Usage: GPUS="9 2 5 8" bash mario_sitl/scripts/run_gated.sh
set -uo pipefail
REPO=/src/gs25122/MARIO
cd "$REPO"
PY=/src/gs25122/miniconda3/envs/mamba/bin/python
export TMPDIR=${TMPDIR:-/src/gs25122/tmp}
read -r -a GPUS <<< "${GPUS:-9 2 5 8}"      # GPU 4 refuses CUDA contexts on this host
SEEDS=${SEEDS:-"42 1 2 3"}
OUT=$REPO/runs/gated
LOGS=$REPO/mario_sitl/results/gated/logs
mkdir -p "$OUT" "$LOGS"

i=0; pids=()
for arch in control gated; do
    for seed in $SEEDS; do
        gpu=${GPUS[$((i % ${#GPUS[@]}))]}
        tag="${arch}_s${seed}"
        CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u mario_sitl/scripts/train_gated.py \
            --arch "$arch" --seed "$seed" --out "$OUT/$tag" > "$LOGS/$tag.log" 2>&1 &
        pids+=($!)
        echo "$(date +%H:%M) launched $tag on gpu $gpu (pid $!)"
        i=$((i + 1))
        sleep 10
        if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait "${pids[@]}"; pids=(); fi
    done
done
wait

echo "=== runs ==="
for d in "$OUT"/*/; do
    [ -f "$d/results.json" ] && echo "ok   $(basename "$d")" || echo "FAIL $(basename "$d")"
done

echo "=== comparison ==="
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY -u mario_sitl/scripts/eval_gated.py --runs "$OUT"
echo "GATED DONE"
