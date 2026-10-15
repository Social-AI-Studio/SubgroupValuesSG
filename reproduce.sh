#!/usr/bin/env bash
# Recreates the dataset from the WVS survey file, scores all models from their released answers,
# writes every paper table, and checks that all outputs are byte-identical to ours (CPU, ~3 min).
set -euo pipefail
cd "$(dirname "$0")"

python 01_process_wvs.py   # stops with instructions if the WVS file is not in data/raw/
python 02_build_dataset.py
python 04_score.py
python 05_data_tables.py
python 06_result_tables.py
python 07_robustness.py
python 08_human_eval.py
python verify.py
