#!/bin/sh
# Train every line of a job file with train.py (two jobs in parallel), skipping finished runs.
cd "$(dirname "$0")"
mkdir -p logs
cat "$1" | xargs -P 2 -I{} sh -c 'o=$(echo "{}" | sed "s/.*--out //"); if [ ! -f $o/pred_val.npy ]; then NT=1 python3 train.py {} >> logs/train.log 2>&1; fi'
