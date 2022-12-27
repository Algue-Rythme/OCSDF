#!/bin/bash

# Example of usage:
# CUDA_VISIBLE_DEVICES=1 ./run_tabular.sh 10 xp1
# Then connect to wandb to see the results.

jupyter nbconvert --to=script --output-dir=experiments/ ./run_tabular.ipynb
for i in `seq 1 $1`; do
  for ds in optdigits; do
    for adoc in ad; do
      echo "Run experiment n°$i from script $0 with dataset $ds and protocol $adoc in group ${2}_${ds}_${adoc}"
      PLOTLY_RENDERER=png DATASET_NAME=$ds ADOC=$adoc WANDB_GROUP=${2}_${ds}_${adoc} python experiments/run_tabular.py
    done
  done  
done
