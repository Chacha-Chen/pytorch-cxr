#!/bin/bash -l

dataset="stanford"
runtime_dir="20190919_noniid_max_single_stanford_per_study_custom"

rm -rf runtime/$runtime_dir

python train_noniid.py \
  --cuda 0 \
  --main-dataset $dataset \
  --runtime-dir $runtime_dir \
  --tensorboard \
  --ignore-repo-dirty
