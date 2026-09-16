#!/usr/bin/env bash
# Track A redone on the no-motor branch: NM Blackbird checkpoints -> PX4 x500.
#
# The adopted x500 model (runs/sitl_agg) started from trial8_100ep, which carries the motor
# encoder and no longer loads here, and it was a single unseeded run. This repeats the same
# recipe (FINETUNE_METHOD.md §1.4) from each of the four NM seeds, then scores every
# checkpoint -- and the four un-tuned NM checkpoints as the zero-shot reference -- on the
# held-out flight_6 with the same two scripts that produced the sitl_agg numbers.
#
# Usage: GPU=8 bash mario_sitl/scripts/run_sitl_nm.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-/src/gs25122/miniconda3/envs/mamba/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-8}
export TMPDIR=${TMPDIR:-/src/gs25122/tmp}
SEEDS=${SEEDS:-"42 1 2 3"}
DATA=mario_sitl/results/dataset_agg
OUT=mario_sitl/results/sitl_nm
mkdir -p "$OUT/figures"

score() {  # score <ckpt> <tag>
    $PY mario_sitl/scripts/eval_openloop.py --npz "$DATA/flight_6.npz" --ckpt "$1" \
        --out "$OUT/openloop_$2.json" --fig "$OUT/figures/openloop_$2.png"
    $PY mario_sitl/scripts/diagnose_observability.py --data "$DATA/flight_6.npz" --ckpt "$1" \
        --out "$OUT/direction_$2.json"
}

for s in $SEEDS; do
    echo "=== zero-shot nm_s$s ==="
    score "runs/nm_s$s/best.pt" "zeroshot_s$s"
done

for s in $SEEDS; do
    echo "=== fine-tune nm_s$s -> runs/sitl_agg_nm_s$s ==="
    $PY mario_sitl/scripts/finetune_sitl.py \
        --dataset "$DATA" --ckpt "runs/nm_s$s/best.pt" --out "runs/sitl_agg_nm_s$s" \
        --epochs 30 --lr 1e-4 --holdout-flights flight_6.npz --no-blackbird --attitude gt --seed "$s"
    # Scoring on flight_6 is only valid if training never saw it.
    $PY - "runs/sitl_agg_nm_s$s/finetune_meta.json" <<'PY' || { echo "ABORT: flight_6 not held out (seed $s)"; exit 1; }
import json, sys
m = json.load(open(sys.argv[1]))
sys.exit(0 if m["sitl_test_flights"] == ["flight_6.npz"]
         and "flight_6.npz" not in m["sitl_train_flights"] else 1)
PY
    score "runs/sitl_agg_nm_s$s/best.pt" "finetuned_s$s"
done

$PY mario_sitl/scripts/summarise_sitl_nm.py --dir "$OUT" --seeds $SEEDS
