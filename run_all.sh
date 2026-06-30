#!/usr/bin/env bash
# Overnight pipeline. Each stage writes its own outputs, so an early crash
# still leaves you with usable submissions from the earlier stages.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p logs
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] STAGE 1/3: ensemble on existing cross-encoder scores"
$PY ensemble.py 2>&1 | tee "logs/1_ensemble.log"
echo "[$(ts)] STAGE 1 done."

echo "[$(ts)] STAGE 2/3: retrain cross-encoder with HARD negatives + rescore"
$PY train_hardneg.py 2>&1 | tee "logs/2_hardneg.log"
echo "[$(ts)] STAGE 2 done."

echo "[$(ts)] STAGE 3/3: ensemble on hard-negative scores"
CE_SCORES_PATH=ce_scores_hn.npy OUT_SUFFIX=_hn $PY ensemble.py 2>&1 | tee "logs/3_ensemble_hn.log"
echo "[$(ts)] STAGE 3 done. ALL COMPLETE."

echo "[$(ts)] Submissions written:"
ls -1 submission*.csv 2>/dev/null | sort
