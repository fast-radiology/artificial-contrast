#!/bin/bash

current_script_path=$(dirname $0)

export REPO="$( cd "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P )"
export DATA="$current_script_path/../data/train/"
export RESULTS="$current_script_path/../results/"
export FOLDS="$current_script_path/../results/config/folds.csv"
export MODEL_SAVE="$current_script_path/../results/"
export BS=2
export NUM_EPOCHS=1
export EXPERIMENT_NAME="simple_multiple_windows"

python -O $current_script_path/../scripts/cv.py
