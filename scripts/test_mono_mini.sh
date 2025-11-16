#!/bin/bash

# CUDA_VISIBLE_DEVICES is dependent on your testing device
export CUDA_VISIBLE_DEVICES=0  # single GPU
export MASTER_PORT=29504

# 'workdir' is your target location to place testing results
torchrun --nproc_per_node=1 --master-port=$MASTER_PORT test_mono.py \
         --work-dir workdir/vis/occscannet/mini \
         --py-config config/test_mono_mini_config.py \

# [example]
# torchrun --nproc_per_node=1 --master-port=$MASTER_PORT test_mono.py \
#          --work-dir /data/EmbodiedOcc/vis/occscannet/mini \
#          --py-config config/test_mono_mini_config.py \
