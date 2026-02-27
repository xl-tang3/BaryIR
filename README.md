# (TPAMI 2026) Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2602.23169)

<hr />

> **Abstract:** *Despite substantial advances in all-in-one image restoration for addressing diverse degradations within a unified model, existing methods remain vulnerable to out-of-distribution degradations, thereby limiting their generalization in real-world scenarios. To tackle the challenge, this work is motivated by the intuition that multisource degraded feature distributions are induced by different degradation-specific shifts from an underlying degradation-agnostic distribution, and recovering such a shared distribution is thus crucial for achieving generalization across degradations. With this insight, we propose BaryIR, a representation learning framework that aligns multisource degraded features in the Wasserstein barycenter (WB) space, which models a degradation-agnostic distribution by minimizing the average of Wasserstein distances to multisource degraded distributions. We further introduce residual subspaces, whose embeddings are mutually contrasted while remaining orthogonal to the WB embeddings. Consequently, BaryIR explicitly decouples two orthogonal spaces: a WB space that encodes the degradation-agnostic invariant contents shared across degradations, and residual subspaces that adaptively preserve the degradation-specific knowledge. This disentanglement mitigates overfitting to in-distribution degradations and enables adaptive restoration grounded on the degradation-agnostic shared invariance. Extensive experiments demonstrate that BaryIR performs competitively against state-of-the-art all-in-one methods. Notably, BaryIR generalizes well to unseen degradations (e.g., types and levels) and shows remarkable robustness in learning generalized features, even when trained on limited degradation types and evaluated on real-world data with mixed degradations.* 
<hr />

##  Setup
This repository is built in PyTorch 2.1.1 and tested on Ubuntu 18.04 environment (Python3.8, CUDA11.8). For super-resolution, the LR images undergo bicubic rescaling to match the dimensions of their respective high-resolution counterparts.
Follow these instructions.
###  Dependencies Installation


1. Clone our repository
```
git clone https://github.com/xl-tang3/BaryIR.git
cd BaryIR
```

2. Create conda environment
The Conda environment used can be recreated using the env.yml file
```
conda env create -f env.yml
```

or


```
conda create -n BaryIR python=3.8
conda activate BaryIR
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
pip install scikit-image
pip install einops
pip install h5py
pip install opencv-python
pip install tqdm
pip install lpips
pip install matplotlib
```

###  Dataset Download and Preperation

All the datasets used in the paper can be downloaded from the following locations:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing), [Kodak24]([https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u](https://www.kaggle.com/datasets/drxinchengzhu/kodak24/data)), [BSD68](https://github.com/cszn/DnCNN/tree/master/testsets)  

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (SOTS)

Deblurring: [GoPro](https://seungjunnah.github.io/Datasets/gopro.html) 

Low-light enhancement: [LOLv1](https://github.com/caiyuanhao1998/Retinexformer) 

The training data should be placed in ``` data/Train/{task_name}``` directory where ```task_name``` can be Denoise, Derain, Dehaze or any single degradation.
After placing the training data the directory structure would be as follows:
```
└───Train
    ├───Dehaze
    │   ├───original
    │   └───synthetic
    ├───Denoise
    └───Derain
        ├───gt
        └───rainy
    └───Deblur
        ├───blur
        ├───sharp
    └───low_light
        ├───high
        ├───low
    └───single
    │   ├───degraded
    │   └───target
```

The testing data should be placed in the ```test``` directory wherein each task has a separate directory. The test directory after setup:

```
└───Test
    ├───dehaze
    │   ├───input
    │   └───target
    ├───denoise
    │   ├───bsd68
    │   └───kodak24
    ├───deblur
    │   ├───input
    │   └───target
    ├───lowlight
    │   ├───low
    │   └───high
    └───derain
    │   └───Rain100L
    │        ├───input
    │        └───target
```
## Citation
```
@article{tang2026learning,
  author={Tang, Xiaole and Gu, Xiang and He, Xiaoyi and Hu, Xin and Sun, Jian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration}, 
  year={2026}
  }

```
