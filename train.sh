#!/bin/bash -l

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="23456"

python -m torch.distributed.launch \
  --nnodes 2 \
  --node_rank 0 \
  --nproc_per_node 1 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  train_noniid.py \
  --cuda 0 \
  --runtime-dir 20190710_20k_noniid_dist_partial_share
