# OctreeSplat 
*OctreeSplat: Hierarchical Octree-Guided Feed-Forward Gaussian Splatting*

<a href="https://arxiv.org/abs/2611.02658"><img src="https://img.shields.io/badge/arXiv-2611.02658-red?logo=arxiv" alt="arXiv"></a>

## Overview
3D Gaussian Splatting (3DGS) has shown considerable potential for novel view synthesis (NVS) and 3D reconstruction. Feed-forward methods avoid costly per-scene optimization, but fixed image or spatial resolutions limit their ability to allocate primitives according to local complexity. We present OctreeSplat, a hierarchical octree-guided feed-forward Gaussian splatting. OctreeSplat extracts multi-view features and geometric priors from a frozen visual foundation model (VFM), constructs a hierarchical octree, and predicts densification scores. The resulting representation allocates Gaussian primitives at spatially varying densities according to local complexity. Stochastic solids volume rendering improves depth rendering, strengthens geometric supervision, and supports surface reconstruction. When camera parameters are available, an optional test-time optimization (TTO) module improves agreement between predicted and ground-truth parameters. Training updates only 76 million additional parameters and requires 35 GPU-hours. Experiments on Replica, TUM, Tanks and Temples, and DTU demonstrate competitive RGB view synthesis, depth synthesis, and surface reconstruction. OctreeSplat reduces the number of Gaussian primitives while maintaining high reconstruction quality.

## TODO

- [ ] Plans after publication
  - [ ] Release pre-trained weights
  - [ ] Release a minimal demo

## Environment Setup
Linux x86_64, Python 3.10, NVIDIA GPU
```bash
cd OctreeSplat
conda create -n OctreeSplat python=3.10 -y
conda activate OctreeSplat
pip install -r requirements.txt
```

## Running the Code
### 1. Prepare the data

The scene directory (Colmap pose) is expected to contain:

```text
data/
├── images/                 # input RGB images
└── sparse/
    └── gt/                 # COLMAP-format camera priors used by TTO
        ├── cameras.txt
        ├── images.txt
        └── points3D.txt    # \
```
The default inference configuration runs test-time camera optimization (TTO),
so `sparse/gt/cameras.txt` and `sparse/gt/images.txt` are required.


### 2. Run inference
```bash
python infer.py --config configs/inference.yaml \
  --input ${dataset_dir} \
  --output ${outputs_dir} \
  --bf16
```