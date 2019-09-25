#!/bin/bash -l

datamode="noniid_max_single"
dataset="stanford"
mode="per_study"
desc="no_positive_weight"
runtime_dir="20190924_${datamode}_${dataset}_${mode}_${desc}"

rm -rf runtime/$runtime_dir

python train_noniid.py \
  --cuda 0 \
  --main-dataset $dataset \
  --runtime-dir $runtime_dir \
  --tensorboard \
  --ignore-repo-dirty \
