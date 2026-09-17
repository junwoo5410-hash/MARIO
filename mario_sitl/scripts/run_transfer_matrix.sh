#!/usr/bin/env bash
# B-3: fine-tuning method comparison. Runs the (mode x seed) matrix for one dataset,
# packing jobs onto the free GPUs. Each job writes its own results.json, so a failed
# job costs only itself.
set -u
PY=/src/gs25122/miniconda3/envs/mamba/bin/python
REPO=/src/gs25122/MARIO
DS=${1:-euroc}
EPOCHS=${2:-60}
SEEDS=${3:-"42 1 2"}
MODES=${4:-"head trunk full scratch"}
LR=${5:-}          # empty = per-mode default (source lr for scratch, a tenth otherwise)
SUFFIX=${6:-}
SELECT=${7:-ate}
NTRAIN=${8:-}   # empty = all training sequences
read -r -a GPUS <<< "${GPUS:-3 4 5 6 7}"
# RUNROOT/LOGDIR keep a second matrix (e.g. the no-motor rerun) from overwriting the first
RUNROOT=${RUNROOT:-$REPO/runs/transfer}
LOGDIR=${LOGDIR:-$REPO/mario_sitl/results/transfer/logs}
INIT=${INIT:-}     # empty = finetune_transfer.py default
mkdir -p "$LOGDIR" "$RUNROOT"

i=0
pids=()
for seed in $SEEDS; do
  for mode in $MODES; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    tag="${DS}_${mode}${SUFFIX}_s${seed}"
    NARG=""; [ -n "$NTRAIN" ] && { NARG="--n-train $NTRAIN"; tag="${DS}_${mode}${SUFFIX}_n${NTRAIN}_s${seed}"; }
    LRARG=""; [ -n "$LR" ] && LRARG="--lr $LR"
    INITARG=""; [ -n "$INIT" ] && INITARG="--init $INIT"
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u "$REPO/mario_sitl/scripts/finetune_transfer.py" \
      --mode "$mode" --dataset "$DS" --seed "$seed" --epochs "$EPOCHS" $LRARG --select "$SELECT" $NARG $INITARG \
      --out "$RUNROOT/$tag" > "$LOGDIR/$tag.log" 2>&1 &
    pids+=($!)
    echo "launched $tag on gpu $gpu (pid $!)"
    i=$((i + 1))
    # stagger so the window cache is written once, not raced by every job at t=0
    if [ $i -le ${#GPUS[@]} ]; then sleep 20; fi
    if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait "${pids[@]}"; pids=(); fi
  done
done
wait
echo "matrix done: $DS"
