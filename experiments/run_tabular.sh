#!/bin/bash

# Example of usage:
# CUDA_VISIBLE_DEVICES=1 ./run_tabular.sh 10 xp1
# Then connect to wandb to see the results.

jupyter nbconvert --to=script --output-dir=experiments/ ./run_tabular.ipynb
for i in `seq 1 $1`; do
  echo "Run experiment n°$i from script $0"
  PLOTLY_RENDERER=png WANDB_GROUP=$2 python experiments/run_tabular.py
done
