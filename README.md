# Vecore 

A high-performance GPU-Accelerated Deep Learning framework built entirely from scratch in C++. 

Vecore replicates the core mathematical engine of PyTorch (Autograd, Tensors, Optimizers, Modules) with **zero third-party ML dependencies**. It features a custom reverse-mode automatic differentiation graph, hand-written CUDA kernels, and integrates with **NVIDIA cuBLAS** to leverage TF32 Tensor Cores.

**🏆 Benchmark Achievement:** On the full 60,000-image MNIST dataset, the Vecore engine achieved **96.23% test accuracy**, outperforming an identical PyTorch implementation (95.61%) on the same hardware, while chewing through 9 million images in 84 seconds.

## 🚀 Key Features

* `Tensor<T>` - N-dimensional array with automatic CPU ↔ GPU transfers.
* **Custom Autograd Engine** - Dynamic, reverse-mode topological sort differentiation.
* **TF32 Tensor Cores** - `cuBLAS` accelerated matrix multiplications.
* **Hand-written CUDA Kernels** - Fused operations for ReLU, SGD, Cross Entropy, and broadcasting.
* `Sequential` - A PyTorch-inspired high-level API with an async, double-buffered GPU training pipeline.
* `CachingAllocator` - Zero-overhead GPU memory pool to eliminate `cudaMalloc` latency.
* `vc::vector` - A custom STL implementation.

## 📚 Documentation & Interactive Demo

Vecore comes with a beautiful, comprehensive documentation website and a live interactive browser demo! 

Explore the API references, architecture guides, and draw digits to test the trained model directly in your browser.

👉 **[View the Documentation & Demo](docs/index.html)** 

*(To view locally, open `docs/index.html` in your web browser)*

## 💻 Quick Start

Vecore is designed to feel instantly familiar to anyone who has used PyTorch.

```cpp
#include "vecore/nn.h"
#include "vecore/optim.h"
#include "vecore/sequential.h"

int main() {
    // 1. Build a PyTorch-style Sequential Model
    Sequential model({
        vc::nn::Dense<float>(784, 2048, "relu"),
        vc::nn::Dense<float>(2048, 1024, "relu"),
        vc::nn::Dense<float>(1024, 10, "none")
    });

    // 2. Shoot weights across the PCIe bus into GPU VRAM
    model.to("cuda");

    // 3. Create SGD Optimizer
    vc::optim::SGD<float> optimizer(model.parameters(), 0.1f);

    // 4. Start the Async GPU Pipeline (Train on dataset)
    model.fit(dataset, 150, 0.1f, 8192);

    return 0;
}
```

## 🛠️ Setup & Build

**Requirements:** 
* C++17 compatible compiler (GCC 9+ or Clang 10+)
* NVIDIA GPU (Compute Capability 7.0+)
* CUDA Toolkit 11.0+
* CMake 3.18+
* OpenBLAS

```bash
git clone https://github.com/marwan/vecore.git
cd vecore
mkdir build && cd build
cmake ..
make -j$(nproc)
```

To run the full MNIST benchmark against the test set:
```bash
./examples/TrainMNIST
```

---
*Built from scratch using pure mathematics and C++.*
