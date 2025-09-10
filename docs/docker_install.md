# Installation
Our code is based on the following environment.

## 1. Prepare docker 

### 1. build docker
```bash
cd SplatSSC
docker build -t splatssc:latest .
```

### 2. Build container
```bash
# build container
docker run -it --privileged \
--gpus all --shm-size=64g -p 8686:22 \
-v /path/to/data/out/container:/path/to/data/in/container \
--name splatssc splatssc 
```

### 3. Activate conda in container

```bash
conda init
source ~/.bashrc
conda activate splatssc
```

### 4. Remote configuration

```bash
passwd  # set passwd
vim /etc/ssh/sshd_config

# -------------------- 
PermitRootLogin yes # add anywhere in /etc/ssh/sshd_config 
# -------------------- 

/etc/init.d/ssh restart  # run this commend every time restarting the container (restart)
```

## 2. Install custom CUDA ops

```bash
cd SplatSSC

# deformable cross attention
cd model/encoder/gaussianformer/ops/pointops && pip install -e .
cd ../../../../..

# Gaussian-to-voxel splatting
cd model/head/gaussian_occ_head/ops/localagg_prob_new && pip install -e .
cd ../../../../..

# Prob Scale Loss 
cd loss/ops/occ_prob && pip install -e .
cd ../../..
```

## 3. Install the additional dependencies and third-party models
1. Install the additional dependencies (optional). 

   ```bash
   cd SplatSSC 
   pip install -r requirements.txt 
   ```

2. Install Depth-Anything-V2. 

   ```bash
   cd SplatSSC 
   git clone https://github.com/DepthAnything/Depth-Anything-V2.git Depth_Anything_V2
   
   # Folder structure
   SplatSSC 
   ├── ... 
   ├── Depth_Anything_V2 
   ```

3. Install EfficientNet-Pytorch. 

   ```bash
   cd SplatSSC
   git clone https://github.com/lukemelas/EfficientNet-PyTorch.git EfficientNet_PyTorch 
   
   # Folder structure
   EmbodiedOcc 
   ├── ...
   ├── Depth_Anything_V2  
   ├── EfficientNet_Pytorch 
   ```

4. replace `Depth_Anything_V2/metric_depth/depth_anything_v2/dpt.py` with our modified version at [`dpt.py`](dpt.py). 

## 4. Download [finetuned checkpoint](https://huggingface.co/YkiWu/EmbodiedOcc) of Depth-Anything-V2 on Occ-ScanNet and put it under the **checkpoints**

1. fine tuned checkpoint -- [Here](https://huggingface.co/YkiWu/EmbodiedOcc). 

2. Folder structure: 

   ```
   SplatSSC 
   ├── ...
   ├── checkpoints/
   │   ├── finetune_scannet_depthanythingv2.pth
   ```
