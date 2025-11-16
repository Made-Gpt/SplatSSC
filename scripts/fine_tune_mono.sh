#!/bin/bash

# CUDA_VISIBLE_DEVICES is dependent on your training device
export CUDA_VISIBLE_DEVICES=1,3
MASTER_PORT=29500

# 'workdir' is your target location to place training results
torchrun --nproc_per_node=2 --master-port=$MASTER_PORT fine_tune_mono.py \
         --work-dir workdir/result/occscannet/depth/main \
         --py-config config/fine_tune_mono_config.py

# [example]
# torchrun --nproc_per_node=2 --master-port=$MASTER_PORT fine_tune_mono.py \
#          --work-dir /data2/EmbodiedOcc/result/occscannet/depth/main \
#          --py-config config/fine_tune_mono_config.py
