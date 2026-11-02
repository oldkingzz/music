#!/bin/bash
# D4 (换个风格): regenerate everything. Run from the repo root (duet_v0).
set -e
PY=.venv/bin/python
# $PY futures/d4_styles/gru_train.py --minutes 6.2          # (optional) retrain the BiGRU on MPS
$PY futures/d4_styles/chord_board.py                         # val leaderboard (add --test once for the final test run)
$PY futures/d4_styles/styles.py                              # 3 styles x 4 melodies: no-clash, stats, demos, blind kit
$PY futures/d4_styles/timing.py m4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 taskpolicy -c background $PY futures/d4_styles/timing.py ecore
$PY futures/d4_styles/make_results.py
