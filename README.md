# SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion
## Overview

Monocular 3D Semantic Scene Completion (SSC) is a challenging yet promising task that aims to infer dense geometric and semantic descriptions of a scene from a single image. While recent object-centric paradigms significantly improve efficiency by leveraging flexible 3D Gaussian primitives, they still rely heavily on a large number of randomly initialized primitives, which inevitably leads to 1) inefficient primitive initialization and 2) outlier primitives that introduce erroneous artifacts. In this paper, we propose SplatSSC, a novel framework that resolves these limitations with a depth-guided initialization strategy and a principled Gaussian aggregator. Instead of random initialization, SplatSSC utilizes a dedicated depth branch composed of a Group-wise Multi-scale Fusion (GMF) module, which integrates multi-scale image and depth features to generate a sparse yet representative set of initial Gaussian primitives. To mitigate noise from outlier primitives, we develop the Decoupled Gaussian Aggregator (DGA), which enhances robustness by decomposing geometric and semantic predictions during the Gaussian-to-voxel splatting process. Complemented with a specialized Probability Scale Loss, our method achieves state-of-the-art performance on the Occ-ScanNet dataset, outperforming prior approaches by over $6.3\%$ in IoU and $4.1\%$ in mIoU, while reducing both latency and memory consumption by more than $9.3\%$. The code will be released upon acceptance. 

![overview](./assets/pipline.png)

----

![overview](./assets/appendix_base.png)

## Getting Started

### Installation
1. Local Environment: Follow instructions [HERE](docs/local_install.md) to prepare the local environment.
2. Docker Environment: Follow instructions [HERE](docs/docker_install.md) to prepare the docker environment.

### Data Preparation
Prepare **posed_images** and **gathered_data** following the [Occ-ScanNet dataset](https://huggingface.co/datasets/hongxiaoy/OccScanNet) and move them to **data/occscannet**. 

1. **Folder structure **

```bash
SplatSSC
├── results/
│   ├── occscannet/
│   │   ├── ...
├── vis/
│   ├── occscannet/
│   │   ├── ...
├── data/
│   ├── occscannet/
│   │   ├── gathered_data/
│   │   ├── posed_images/
│   │   ├── train_final.txt
│   │   ├── train_mini_final.txt
│   │   ├── test_final.txt
│   │   ├── test_mini_final.txt
```

## Train

1. Pre-train our depth branch using 2 GPUs on Occ-ScanNet:
    ```bash
    cd SplatSSC 
    bash fine_tune_mono.sh 
    ```
2. Train SplatSSC using 4 GPUs on Occ-ScanNet and Occ-ScanNet-mini:
    ```bash
    cd SplatSSC
    # mini 
    bash train_mono_mini.sh  
    # base 
    bash train_mono.sh  
    ```

## Test

1. Test our depth branch using 1 GPU on Occ-ScanNet:

   ```bash
   cd SplatSSC
   bash test_fine_tuned_mono.sh   
   ```

2. Test SplatSSC using 1 GPU on Occ-ScanNet and Occ-ScanNet-mini:

   ```bash
   cd SplatSSC
   # mini 
   bash test_mono_mini.sh  
   # base 
   bash test_mono.sh  
   ```

   

## Related Projects

Our work is inspired by these excellent open-sourced repos:
[EmbodiedOcc](https://github.com/ykiwu/embodiedocc), [GaussianFormer](https://github.com/huang-yh/GaussianFormer), [ISO](https://github.com/hongxiaoy/ISO).

Our code is based on [EmbodiedOcc](https://github.com/ykiwu/embodiedocc) and [GaussianFormer](https://github.com/huang-yh/GaussianFormer).

## License 

The source code and trained models associated with this paper will be made publicly available upon acceptance. The repository will be hosted on GitHub, and the specific URL will be provided here.

All our original source code is licensed under the [CC-BY-NC-SA-4.](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. This permits any non-commercial use, distribution, and reproduction in any medium, provided the original work is properly cited and any derivative works are shared under the same license.
# SplatSSC
