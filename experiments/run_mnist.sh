#!/bin/bash

# Example of usage:
# CUDA_VISIBLE_DEVICES=1 WANDB_GROUP=cvpr_fig2 ./run_mnist.sh 10
# Then connect to wandb to see the results.

#jupyter nbconvert --to=script --output-dir=experiments/ ./run_mnist.ipynb
for i in `seq 1 $1`; do
  echo "Run experiment n°$i from script $0"
  PLOTLY_RENDERER=png DATASET_NAME=mnist WANDB_GROUP=mnist4 python experiments/run_mnist.py
done
