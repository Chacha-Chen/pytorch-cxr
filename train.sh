#!/bin/bash -l

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="23456"

python -m torch.distributed.launch \
  --nproc_per_node 3 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  train_noniid.py \
  --cuda 0,1,2 \
  --tensorboard \
  --epoch 500 \
  --start-epoch 14 \
  --runtime-dir 20190805_20k_noniid_dist_label_split
