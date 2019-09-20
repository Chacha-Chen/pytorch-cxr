#!/bin/bash -l

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="12345"

runtime_dir="20190919_iid_max_dist_rank3_per_study"

python -m torch.distributed.launch \
  --nnodes 1 \
  --node_rank 0 \
  --nproc_per_node 3 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  train.py \
  --cuda 0 \
  --runtime-dir $runtime_dir \
  --tensorboard \
  --ignore-repo-dirty
