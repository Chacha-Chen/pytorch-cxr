#!/bin/bash -l

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

export MASTER_ADDR="172.19.2.106"
export MASTER_PORT="12345"

python -m torch.distributed.launch \
  --nnodes 2 \
  --node_rank 0 \
  --nproc_per_node 6 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  train_noniid.py \
  --cuda 0 \
  --runtime-dir 20190913_noniid_1k_dist_rank12 \
  --tensorboard \
  --ignore-repo-dirty
