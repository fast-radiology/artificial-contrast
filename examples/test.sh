#!/bin/bash

current_script_path=$(dirname $0)

export REPO="$( cd "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P )"
export DATA="$current_script_path/../data/train/"
export TEST_DATA="$current_script_path/../data/test/"
export RESULTS="$current_script_path/../results/"
export FOLDS="$current_script_path/../results/config/folds.csv"
export MODEL_SAVE="$current_script_path/../results/"
export MODEL_NAME="simple_multiple_windows"
export DCM_CONF="{\"windows\": [[-40, 120], [-100, 300], [300, 2000]], \"norm_stats\": [[0.6532502770423889, 0.6329413652420044, 0.28225114941596985], [0.466626912355423, 0.4579363167285919, 0.32907354831695557]]}"
export BS=2
export NUM_EPOCHS=1
export EXPERIMENT_NAME="simple_multiple_windows"

python -O $current_script_path/../scripts/testset_evaluation.py
