
# Spline-Based Convolution Operator of SplineCNN


--------------------------------------------------------------------------------

This is a PyTorch implementation of the spline-based convolution operator of SplineCNN for handling ML augmented Liver Registration, as described in our paper:

Dingrong Wang, Soheil et al.: [LIBR+: Improving Intraoperative Liver Registration by Learning the Residual of Biomechanics-Based Deformable Registration](https://arxiv.org/abs/2403.06901)

The operator works on all floating point data types and is implemented both for CPU and GPU.

## Installation

Run

```
conda create -n spline -f spline.yml
```

### Anaconda

**Update:** You can now install `pytorch-spline-conv` via [Anaconda](https://anaconda.org/pyg/pytorch-spline-conv) for all major OS/PyTorch/CUDA combinations 🤗
Given that you have [`pytorch >= 1.8.0` installed](https://pytorch.org/get-started/locally/), simply run

```
conda install pytorch-spline-conv -c pyg
```

### Binaries

We alternatively provide pip wheels for all major OS/PyTorch/CUDA combinations, see [here](https://data.pyg.org/whl).

#### PyTorch 2.0

To install the binaries for PyTorch 2.0.0, simply run

```
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu117`, or `cu118` depending on your PyTorch installation.

|             | `cpu` | `cu117` | `cu118` |
|-------------|-------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      |
| **macOS**   | ✅    |         |         |

#### PyTorch 1.13

To install the binaries for PyTorch 1.13.0, simply run

```
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu116`, or `cu117` depending on your PyTorch installation.

|             | `cpu` | `cu116` | `cu117` |
|-------------|-------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      |
| **macOS**   | ✅    |         |         |

**Note:** Binaries of older versions are also provided for PyTorch 1.4.0, PyTorch 1.5.0, PyTorch 1.6.0, PyTorch 1.7.0/1.7.1, PyTorch 1.8.0/1.8.1, PyTorch 1.9.0, PyTorch 1.10.0/1.10.1/1.10.2, PyTorch 1.11.0 and PyTorch 1.12.0/1.12.1 (following the same procedure).
For older versions, you need to explicitly specify the latest supported version number or install via `pip install --no-index` in order to prevent a manual installation from source.
You can look up the latest supported version number [here](https://data.pyg.org/whl).


### Spline-CNN Parameters

* **x** *(Tensor)* - Input node features of shape `(number_of_nodes x in_channels)`.
* **edge_index** *(LongTensor)* - Graph edges, given by source and target indices, of shape `(2 x number_of_edges)`.
* **pseudo** *(Tensor)* - Edge attributes, ie. pseudo coordinates, of shape `(number_of_edges x number_of_edge_attributes)` in the fixed interval [0, 1].
* **weight** *(Tensor)* - Trainable weight parameters of shape `(kernel_size x in_channels x out_channels)`.
* **kernel_size** *(LongTensor)* - Number of trainable weight parameters in each edge dimension.
* **is_open_spline** *(ByteTensor)* - Whether to use open or closed B-spline bases for each dimension.
* **degree** *(int, optional)* - B-spline basis degree. (default: `1`)
* **norm** *(bool, optional)*: Whether to normalize output by node degree. (default: `True`)
* **root_weight** *(Tensor, optional)* - Additional shared trainable parameters for each feature of the root node of shape `(in_channels x out_channels)`. (default: `None`)
* **bias** *(Tensor, optional)* - Optional bias of shape `(out_channels)`. (default: `None`)

### Spline-CNN Returns

* **out** *(Tensor)* - Out node features of shape `(number_of_nodes x out_channels)`.

### Training and Test Example

Simply Run

1. train_test_sparse.py
2. train_test_dense.py

## Cite

Please cite our paper if you use this code in your own work:

```
@article{wang2024libr+,
  title={LIBR+: Improving Intraoperative Liver Registration by Learning the Residual of Biomechanics-Based Deformable Registration},
  author={Wang, Dingrong and Azadvar, Soheil and Heiselman, Jon and Jiang, Xiajun and Miga, Michael and Wang, Linwei},
  journal={arXiv preprint arXiv:2403.06901},
  year={2024}
}
```

