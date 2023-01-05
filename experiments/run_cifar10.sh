#!/bin/bash

# Example of usage:
# CUDA_VISIBLE_DEVICES=1 ./run_cifar.sh 10 xp1
# Then connect to wandb to see the results.

jupyter nbconvert --to=script --output-dir=experiments/ ./run_cifar10.ipynb
for i in `seq 1 $1`; do
  for c in 0 1 2 3 4 5 6 7 8 9; do
    echo "Run experiment n°$i from script $0 with group cifar${c}_$2 in class $c"
    PLOTLY_RENDERER=png DATASET_NAME=cifar10 IN_CLASS=$c WANDB_GROUP=cifar${c}_$2 python experiments/run_cifar10.py
  done
done
