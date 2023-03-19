# EDN

- [Overview](#overview)
- [Installation](#installation)
	- [Create a conda environment](#create-a-conda-environment) 
    - [Install torch](#install-torch)
    - [Install pytorch-lightning and other generic dependencies](#install-pytorch-lightning-and-other-generic-dependencies)
    - [Install e3nn](#install-e3nn)
    - [Install atom3d](#install-atom3d)
- [Usage](#usage)
	- [Creating an LMDB input dataset](#creating-an-lmdb-input-dataset)
	- [Training](#training)
    - [Inference](#inference) 
    - [Using a GPU](#using-a-gpu)
    - [Training and testing on CASP datasets](#training-and-testing-edn-on-casp-datasets)

----------------------------------

## Overview

This repository provides a PyTorch implementation of the EDN architecture from [Protein model quality assessment
using rotation-equivariant transformations on point clouds](https://arxiv.org/abs/2011.13557). EDN is an equivariant neural network designed to predict the accuracy of a protein model. We tested EDN as part of the blind prediction experiment on model quality assessment in CASP 14 (https://predictioncenter.org/casp14/index.cgi). 

This document contains instructions on how to use the EDN architecture for general training and inference. In addition, we provide specific instructions on how to train and evaluate a network using protein model datasets from CASP 5-14.

EDN builds on [tensor field networks](https://arxiv.org/abs/1802.08219) and the [PAUL network for protein complexes](https://onlinelibrary.wiley.com/doi/10.1002/prot.26033).

## Installation 

### Create a conda environment

```
conda create -n edn python=3.9 pip
conda activate edn
```
### Install torch

Install appropriate versions of torch and attendant libraries.  Please set the adequate version of CUDA for your system.  The instructions shown are for CUDA 11.7. If you want to install the CPU-only version, use CUDA="".

```
TORCH="1.13.0"
CUDA="cu117"
pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
```
### Install pytorch-lightning and other generic dependencies

`pip install pytorch-lightning python-dotenv wandb`

### Install e3nn

In addition, we need to install an EDN-compatible version of the e3nn library (https://github.com/e3nn/e3nn). Please note that this specific version is only provided for compatability, further development should be done using the main e3nn branch.

`pip install git+ssh://git@github.com/drorlab/e3nn_edn.git`

### Install atom3d

We use the LMDB data format from Atom3D (https://www.atom3d.ai/) for fast random access. To install the atom3d package:

`pip install atom3d`

## Usage

### Creating an LMDB input dataset

The code expects protein models in LMDB format as input. To convert a set of PDB files (here files in `data/pdbs`) to the LMDB format, run:

`python -m atom3d.datasets data/pdbs example_lmdb -f pdb`

### Training

To train a network on a CPU using an example LMDB file located at `data/example_data`, run the following command:

`python -m edn.train data/example_data data/example_data --batch_size=2 --accumulate_grad_batches=32 --learning_rate=0.001 --max_epochs=6 --output_dir out/model --num_workers=4`

Note that this will run quite slowly. To run faster, consider using a GPU (see below).

### Inference

To make predictions, the general format is as follows:

`python -m edn.predict input_dir checkpoint.ckpt output.csv [--nolabels]`

For example, to predict on the example LMDB file included in the repository, using dummy weights:

`python -m edn.predict data/example_data data/sample_weights.ckpt output.csv --nolabels`

The expected output in `output.csv` for the above command would be (with possible fluctuation in up to 7th decimal place):

```
id,target,pred
T0843-Alpha-Gelly-Server_TS3.pdb,0.0000000,0.4594232
T0843-BioSerf_TS1.pdb,0.0000000,0.5363037
```

### Using a GPU

You can enable a GPU with the `--gpus` flag.  It is also recommended to provision additional CPUs with the `--num_workers` flags (more is better). The GPU should have at least 12GB of memory.  A training example:

`python -m edn.train data/example_data data/example_data --batch_size=2 --accumulate_grad_batches=32 --learning_rate=0.001 --max_epochs=6 --output_dir out/model --gpus=1 --num_workers=4`

To run inference using a GPU:

`python -m edn.predict data/example_data data/sample_weights.ckpt output.csv --nolabels --gpus=1 --num_workers=4`

### Training and testing EDN on CASP datasets
In order to facilitate training and evaluation on the structural model sets from previous CASP experiments, we make pre-processed CASP 5-14 datasets available in LMDB format. These can be accessed at https://drive.google.com/drive/u/1/folders/1ssJpmdCKcPZo5iQfo2_f5iEwn1Z6zx32. The structural models were downloaded from the official CASP website at https://predictioncenter.org/.

Assuming that the CASP datasets are placed in the `data` folder, you can run the following command to train an EDN network with the CASP 5-10 datasets on a GPU as follows:

`python -m edn.train data/casp5_to_10/data/train data/casp5_to_10/data/val --batch_size=2 --accumulate_grad_batches=32 --learning_rate=0.001 --max_epochs=1 --output_dir out/model --gpus=1 --num_workers=4`

The output files are written to the folder `out/model`.

To test the trained network on a CASP dataset (here the stage 2 scoring set from CASP 11) using a GPU, run the following command:

`python -m edn.predict data/casp11_stage2/data out/model/checkpoints/last.ckpt output.csv --nolabels --gpus=1 --num_workers=4`
