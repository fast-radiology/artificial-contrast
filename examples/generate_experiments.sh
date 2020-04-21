#!/bin/bash

current_script_path=$(dirname $0)

export N_FOLDS="2"
export DATA="$current_script_path/../data/train/"
export RESULTS="$current_script_path/../results/config/"

python -O $current_script_path/../scripts/generate_experiments.py
