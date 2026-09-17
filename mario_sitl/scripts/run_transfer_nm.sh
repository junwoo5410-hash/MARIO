#!/usr/bin/env bash
# EuRoC transfer matrix rerun on the no-motor branch (PAPER_NOTES.md §13 item 3).
#
# Every EuRoC number so far came from the motor-era m_s42. This repeats the two tables the
# write-ups cite -- the arm comparison and the data-efficiency curve -- from nm_s42 with the
# protocol unchanged (gravity + yaw 185 alignment, tail-20 % validation, rollout-ATE
# selection, seeds 42/1/2/3), into runs/transfer_nm so the motor-era runs stay intact.
# Then rescores everything over the full trajectory, with all four NM seeds as zero-shot.
#
# Usage: GPUS="8 2 5 9 4" bash mario_sitl/scripts/run_transfer_nm.sh
set -euo pipefail
REPO=/src/gs25122/MARIO
cd "$REPO"
PY=/src/gs25122/miniconda3/envs/mamba/bin/python
export TMPDIR=${TMPDIR:-/src/gs25122/tmp}
export GPUS=${GPUS:-"3 4 5 6 7"}
export RUNROOT=$REPO/runs/transfer_nm
export LOGDIR=$REPO/mario_sitl/results/transfer_nm/logs
export INIT=$REPO/runs/nm_s42/best.pt
SEEDS="42 1 2 3"
M=mario_sitl/scripts/run_transfer_matrix.sh

echo "=== arm comparison, all 6 training sequences ==="
bash $M euroc 60 "$SEEDS" "head trunk full scratch" "" "" ate

for n in 1 2 4; do
    echo "=== data efficiency, n_train=$n ==="
    bash $M euroc 60 "$SEEDS" "head full scratch" "" "" ate "$n"
done

echo "=== failed runs (no results.json) ==="
for d in "$RUNROOT"/*/; do [ -f "$d/results.json" ] || echo "MISSING: $d"; done
echo "runs with results: $(ls "$RUNROOT"/*/results.json | wc -l) / 52"

echo "=== windowed summary ==="
$PY mario_sitl/scripts/summarise_transfer.py euroc "$RUNROOT"

echo "=== full-trajectory rescore ==="
$PY mario_sitl/scripts/reeval_transfer_full.py --runs-dir "$RUNROOT" \
    --zeroshot runs/nm_s42/best.pt runs/nm_s1/best.pt runs/nm_s2/best.pt runs/nm_s3/best.pt \
    --out mario_sitl/results/transfer_nm/full_rollout_rescore.json
echo "ALL DONE"
