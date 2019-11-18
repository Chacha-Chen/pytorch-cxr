#!/bin/bash -l

datamode="noniid_max_single"
dataset="stanford"
mode="per_study"
desc="densenet169_dropout"
runtime_dir="20191009_${datamode}_${dataset}_${mode}_${desc}"
start_epoch=1

if [ $start_epoch -gt 1 ]; then
  rm -rf runtime/$runtime_dir
fi

python train_noniid.py \
  --cuda 0 \
  --main-dataset $dataset \
  --runtime-dir $runtime_dir \
  --start-epoch $start_epoch \
  --tensorboard \
  --ignore-repo-dirty \
