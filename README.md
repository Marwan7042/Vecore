# Micrograd C++
A Deep Learning framework built entirely from scratch in C++. This project replicates the core mathematical engine of PyTorch (Autograd, Tensors, Optimizers, Modules) with no third-party ML dependencies. 

Includes **CUDA / cuBLAS** integration to offload matrix multiplications across the PCIe bus to an NVIDIA GPU.

## What's Included

* `mstd::vector` - Custom STL implementation 
* `Tensor<T>` - N-dimensional array with strides, zero-copy transpose, and `.to("cuda")`
* **Autograd Engine** - Forward tracking and reverse-mode topological sort differentiation
* `nn::Dense`, `nn::ReLU` - Modular network layers
* `nn::CrossEntropyLoss` - Softmax + Negative Log Likelihood loss
* `optim::SGD` - Stochastic Gradient Descent optimizer
* `TrainMNIST` - A complete end-to-end computer vision pipeline that trains a (784 -> 512 -> 10) network on the MNIST dataset using the RTX 5050 GPU, hitting 100% training accuracy.

## Interactive Demo
This repository contains a live interactive demo of the neural network's predictions and probability distributions trained on MNIST.

> **Note on Overfitting:** The `TrainMNIST` script is designed to test the framework's mathematical correctness, not to train a production-ready model. It intentionally trains on a very small subset of MNIST (500 images) using a massive 400k+ parameter architecture. Achieving 100% accuracy here is deliberate—it proves that the custom Autograd engine, backpropagation math, and cuBLAS integration are completely bug-free by successfully minimizing the loss to zero.

## Setup & Build

Requirements: C++17, CMake, NVIDIA CUDA Toolkit.

```bash
mkdir build && cd build
cmake ..
make -j
./TrainMNIST
```
