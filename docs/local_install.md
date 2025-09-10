# Installation
Our code is based on the following environment.

## 1. Create conda environment
```bash
conda create -n splatssc python=3.8.19
conda activate splatssc
```

## 2. Install PyTorch
```bash
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu113 
```

## 3. Install some packages following [GaussianFormer](https://github.com/huang-yh/GaussianFormer) and [EmbodiedOcc](https://github.com/ykiwu/embodiedocc)

### 1. Install packages from MMLab
```bash
pip install openmim==0.3.9
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmsegmentation==1.2.2
mim install mmdet3d==1.1.1
```

### 2. Install other packages
```bash
pip install spconv-cu114==2.3.6 
pip install timm 
pip install vtk==9.0.1 
pip install h5py 
pip install einops
```

### 3. Install custom CUDA ops
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

## 4. Install the additional dependencies and third-party models

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

## 5. Download [finetuned checkpoint](https://huggingface.co/YkiWu/EmbodiedOcc) of Depth-Anything-V2 on Occ-ScanNet and put it under the **checkpoints**

1. fine tuned checkpoint -- [Here](https://huggingface.co/YkiWu/EmbodiedOcc). 

2. Folder structure: 

   ```bash
   SplatSSC 
   ├── ...
   ├── checkpoints/
   │   ├── finetune_scannet_depthanythingv2.pth
   ```
