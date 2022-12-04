#!/bin/bash

# Using conda within a shell script
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate conda environment
conda env create -f environment.yml -n ngtgen
conda activate ngtgen

# Ensure the correct interpreter is executed
echo $(which python)
echo $(which pip)


# Upgrade Pytorch for CUDA 11.6
pip install --upgrade --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Evaluate trained model on test set and print results to "results.out"
python test.py --gpu 0 --checkpoint best-checkpoint.mdl --test-file test.jsonl

