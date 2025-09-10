#!/bin/bash

# CUDA_VISIBLE_DEVICES is dependent on your training device
export CUDA_VISIBLE_DEVICES=1,2
export MASTER_PORT=29500

# 'workdir' is your target location to place training results
torchrun --nproc_per_node=2 --master-port=$MASTER_PORT train_mono.py \
         --work-dir workdir/result/occscannet/base/main \
         --py-config config/train_mono_config.py \

# [example]
# torchrun --nproc_per_node=2 --master-port=$MASTER_PORT train_mono.py \
#          --work-dir /data2/EmbodiedOcc/result/occscannet/base/main \
#          --py-config config/train_mono_config.py \
