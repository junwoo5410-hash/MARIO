#!/usr/bin/env bash
# Rerun transfer_nm jobs that died without results.json (GPU 4 on the lab server refuses
# every CUDA context: "CUDA-capable device(s) is/are busy or unavailable").
#
# Retries the ones already failed now, waits for run_transfer_nm.sh to finish, retries
# whatever else failed, then redoes the summary and the full-trajectory rescore so they
# cover all 52 runs.
#
# Usage: GPUS="9 2" bash mario_sitl/scripts/retry_transfer_nm.sh
set -uo pipefail
REPO=/src/gs25122/MARIO
cd "$REPO"
PY=/src/gs25122/miniconda3/envs/mamba/bin/python
export TMPDIR=${TMPDIR:-/src/gs25122/tmp}
read -r -a GPUS <<< "${GPUS:-9 2}"
RUNROOT=$REPO/runs/transfer_nm
LOGDIR=$REPO/mario_sitl/results/transfer_nm/logs
DRIVER=$REPO/mario_sitl/results/transfer_nm/driver.log
INIT=$REPO/runs/nm_s42/best.pt

failed() {  # tags whose log has a traceback and no results.json
    for log in "$LOGDIR"/*.log; do
        tag=$(basename "$log" .log)
        [ -f "$RUNROOT/$tag/results.json" ] && continue
        grep -q Traceback "$log" && echo "$tag"
    done
}

retry() {  # retry <tag>...
    local i=0 pids=()
    for tag in "$@"; do
        [[ $tag =~ ^euroc_([a-z]+)(_n([0-9]+))?_s([0-9]+)$ ]] || { echo "skip $tag"; continue; }
        mode=${BASH_REMATCH[1]}; n=${BASH_REMATCH[3]}; seed=${BASH_REMATCH[4]}
        gpu=${GPUS[$((i % ${#GPUS[@]}))]}
        mv "$LOGDIR/$tag.log" "$LOGDIR/$tag.failed.log"
        NARG=""; [ -n "$n" ] && NARG="--n-train $n"
        CUDA_VISIBLE_DEVICES=$gpu $PY -u mario_sitl/scripts/finetune_transfer.py \
            --mode "$mode" --dataset euroc --seed "$seed" --epochs 60 --select ate $NARG \
            --init "$INIT" --out "$RUNROOT/$tag" > "$LOGDIR/$tag.log" 2>&1 &
        pids+=($!)
        echo "$(date +%H:%M) retry $tag on gpu $gpu (pid $!)"
        i=$((i + 1)); sleep 20
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
}

mapfile -t now < <(failed)
echo "failed now: ${now[*]:-none}"
retry "${now[@]}"

echo "$(date +%H:%M) waiting for run_transfer_nm.sh"
until grep -q "ALL DONE" "$DRIVER" 2>/dev/null || ! pgrep -f "bash mario_sitl/scripts/run_transfer_nm.sh" >/dev/null; do
    sleep 60
done

mapfile -t later < <(failed)
echo "failed after matrix: ${later[*]:-none}"
retry "${later[@]}"

echo "runs with results: $(ls "$RUNROOT"/*/results.json | wc -l) / 52"
failed | sed 's/^/STILL FAILED: /'

echo "=== windowed summary ==="
$PY mario_sitl/scripts/summarise_transfer.py euroc "$RUNROOT"
echo "=== full-trajectory rescore ==="
$PY mario_sitl/scripts/reeval_transfer_full.py --runs-dir "$RUNROOT" \
    --zeroshot runs/nm_s42/best.pt runs/nm_s1/best.pt runs/nm_s2/best.pt runs/nm_s3/best.pt \
    --out mario_sitl/results/transfer_nm/full_rollout_rescore.json
echo "RETRY DONE"
