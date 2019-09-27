#!/bin/bash -l

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="12345"

datamode="noniid_max_dist"
mode="per_study"
desc="no_positive_weight"
runtime_dir="20190924_${datamode}_${mode}_${desc}"
start_epoch=1

if [ $start_epoch -gt 1 ]; then
  rm -rf runtime/$runtime_dir
fi

python -m torch.distributed.launch \
  --nnodes 1 \
  --node_rank 0 \
  --nproc_per_node 3 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  train_noniid.py \
  --cuda 0,1,2 \
  --runtime-dir $runtime_dir \
  --start-epoch $start_epoch \
  --tensorboard \
  --ignore-repo-dirty
