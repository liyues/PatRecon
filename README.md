# Patient-specific reconstruction of volumetric computed tomography images from few-view projections via deep learning


## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Instructions for Use](#instructions-for-use)
- [License](./LICENSE)
- [Citation](#citation)

# 1. Overview

This project provides a deep-learning framework for generating patient-specific volumetric tomographic X-ray images when a few 2D projections are given as input. The study demonstrates the feasibility of the approach using several patients of different disease sites.
Using the code requires users to have basic knowledge about python, PyTorch, and deep neural networks.


# 2. Repo Contents

- [test.py](./test.py): main code to run evaluation.
- [net.py](./net.py): network definition of proposed framework.
- [exp/model/model.pth.tar](./exp/model/model.pth.tar): trained model for running experiment.
- [exp/data/2D_projection.jpg](./exp/data/2D_projection.jpg): 2D projection of the data sample, which is the input of model.
- [exp/data/3D_CT.jpg](./exp/data/3D_CT.bin): 3D CT volume of the data sample, which will be used as groundtruth to compare with the output (prediction) results.
- [exp/result](./exp/result): output folder to save the model prediction as .png files.
- Please download the model and data sample and put them under `exp` folder as above. The downloading link of the model and data sample will come soon.

# 3. System Requirements

## Hardware Requirements

Loading and running deep network requires a standard computer with enough memory to support the model defined by a user. For optimal performance, a GPU card can largely accerlerate computation. In our experiment, we use a NVIDIA Tesla V100 GPU card with about 32 GB of memory. 
<!-- we recommend a computer with the following specs:
RAM: 16+ GB  
CPU: 4+ cores, 3.3+ GHz/core -->

The runtimes below are generated using a computer with a NVIDIA Tesla V100 GPU.


## OS Requirements

This package is supported for *Linux* operating systems. The package has been tested on the following systems:

Linux: Ubuntu 16.04  


# 4. Installation Guide

Before running this package, users should have `Python`, `PyTorch`, and several python packages (`numpy`, `sklearn`, `skimage`, `PIL`, and `matplotlib`) installed.

## Installing Python version 3.5.5 on Ubuntu 16.04

The Python can be installed in Linux by running following command from terminal:
```
sudo apt-get update
sudo apt-get install python3.5
```

which should install in about 30 seconds.

## Package Versions

This code functions with following dependency packages. The versions of software are, specifically:
```
pytorch: 0.4.1
numpy: 1.15.0
sklearn: 0.19.1
skimage: 0.14.0
PIL: 5.1.0
matplotlib: 2.2.2
```


## Package Installment

Users should install all the required packages shown above prior to running the algorithm. Most packages can be installed by running following command in terminal on Linux. To install of PyTorch, please refer to their official [website](https://pytorch.org). 
```
pip install package-name
```

which will install in about 30 mins on a recommended machine.


# 5. Instructions for Use

To running the trained model to evaluate reconstruction performance on data samples, please type in following comman in terminal. Parameter `exp` is used to run different experiments. Running a model inference on one data sample should take approximately 2 mins using a computer with a NVIDIA Tesla V100 GPU.


## Running Experiment
Please use the parameter `vis_plane` to get output image slices on different planes of 3D pancreas CT. The prediction results are saved under path `exp/test_results/sample_1`. The results contains `.png` files, which are named after `Plane_[0/1/2]_ImageSlice_[].png`. Each file shows the prediction, groundtruth, and difference image for one slice along the chosen plane. Specifically, the following commands could be run in terminal to get model results and visualize based on three different planes [0: axial, 1: sagittal, 2: coronal] of 3D pancreas CT.
```
python3 test.py --vis_plane 0
python3 test.py --vis_plane 1
python3 test.py --vis_plane 2
```

# 6. License
This study is protected by US patent. The code and other materials in this study can be used for research only. Correspondence and requests for materials should be addressed to the corresponding author in the paper.


# 7. Citation
If you find the code are useful, please cite the paper.
```
@article{shen2019PatRecon,
  title={Patient-specific reconstruction of volumetric computed tomography images from a single projection view via deep learning},
  author={Shen, Liyue and Zhao, Wei and Xing, Lei},
}
```
