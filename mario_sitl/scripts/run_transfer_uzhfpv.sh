#!/usr/bin/env bash
# UZH-FPV transfer matrix: does a few minutes of real fast-flight data make this work?
#
# EuRoC could only rank arms, because every arm there sat below the stationary baseline
# (4.54 m) -- its 0.4-0.5 m/s flight carries almost no velocity signal. UZH-FPV is real
# racing flight at 2.8-8.0 m/s mean, and nm_s42 zero-shot already clears the baseline on
# the test split (13.08 m vs 15.07 m). So this is the dataset where "fine-tune on a
# couple of minutes and it works" can actually be tested.
#
# Same protocol as the EuRoC matrix, three things different:
#   * dataset uzhfpv, our own 12/4 split (train_list.txt, test_list.txt);
#   * alignment is a plain yaw of 310 deg, picked by uzhfpv_yaw_sweep.py on the TRAIN
#     split. EuRoC's gravity flip made things worse here (19.05 m vs 15.39 unaligned);
#   * n_train counts sequences and they differ in length, so train_list.txt is ordered
#     longest first: 1 = 1.43 min of ground-truth span (1.04 min of it trainable after the
#     validation tail), 2 = 2.54 / 1.83, 4 = 4.19 / 2.95, 12 = 8.43 / 5.55.
#
# Usage: GPUS="9 2 5 8" bash mario_sitl/scripts/run_transfer_uzhfpv.sh
set -euo pipefail
REPO=/src/gs25122/MARIO
cd "$REPO"
PY=/src/gs25122/miniconda3/envs/mamba/bin/python
export TMPDIR=${TMPDIR:-/src/gs25122/tmp}
export GPUS=${GPUS:-"9 2 5 8"}          # GPU 4 refuses CUDA contexts on this host
export RUNROOT=$REPO/runs/transfer_uzhfpv
export LOGDIR=$REPO/mario_sitl/results/uzhfpv/logs
export INIT=$REPO/runs/nm_s42/best.pt
export ALIGN=yaw
export VALGAP=600                       # 1200 leaves the shortest sequences with no windows
export YAW=310
SEEDS="42 1 2 3"
M=mario_sitl/scripts/run_transfer_matrix.sh

echo "=== arm comparison, all 12 training sequences ==="
bash $M uzhfpv 60 "$SEEDS" "head trunk full scratch" "" "" ate

for n in 1 2 4; do
    echo "=== data efficiency, n_train=$n ==="
    bash $M uzhfpv 60 "$SEEDS" "head full scratch" "" "" ate "$n"
done

echo "runs with results: $(ls "$RUNROOT"/*/results.json 2>/dev/null | wc -l) / 52"
for d in "$RUNROOT"/*/; do [ -f "$d/results.json" ] || echo "MISSING: $d"; done

echo "=== windowed summary ==="
$PY mario_sitl/scripts/summarise_transfer.py uzhfpv "$RUNROOT"

echo "=== full-trajectory rescore ==="
$PY mario_sitl/scripts/reeval_transfer_full.py --runs-dir "$RUNROOT" \
    --zeroshot runs/nm_s42/best.pt runs/nm_s1/best.pt runs/nm_s2/best.pt runs/nm_s3/best.pt \
    --zeroshot-dataset uzhfpv --zeroshot-align yaw --zeroshot-yaw 310 \
    --out mario_sitl/results/uzhfpv/full_rollout_rescore.json
echo "ALL DONE"
