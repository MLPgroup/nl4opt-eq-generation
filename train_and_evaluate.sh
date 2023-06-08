#!/bin/bash

# Using conda within a shell script
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate conda environment
CONDA_ENV_NAME=tgen
conda env remove -n $CONDA_ENV_NAME
conda env create -f environment.yml -n $CONDA_ENV_NAME
conda activate $CONDA_ENV_NAME

# Upgrade PyTorch and Cuda
conda install --yes pytorch=1.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

# Run training
export NL4OPT_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --config configs/default.json --seed 42

# Evaluate trained model on test set and print results to "results.out"
python test.py --gpu 0 --checkpoint output/default/$NL4OPT_TIMESTAMP/best-checkpoint.mdl --test-file data/test.jsonl --batch-size 32 --beam-size 1
