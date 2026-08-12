#include "vecore/Tensor.h"
#include <iostream>
#include <stdexcept>
#include <cblas.h>
#include <type_traits>
#include <cmath>

namespace vc {
template <typename T>
__global__ void maxpool2d_forward_kernel(
    const int n, const T* bottom_data,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    T* top_data, T* argmax_data) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int b = index / pooled_width / pooled_height / channels;
        
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        
        T maxval = -1e20;
        int maxidx = -1;
        
        const T* bottom_slice = bottom_data + (b * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                if (bottom_slice[h * width + w] > maxval) {
                    maxidx = h * width + w;
                    maxval = bottom_slice[maxidx];
                }
            }
        }
        top_data[index] = maxval;
        argmax_data[index] = (T)maxidx;
    }
}

template <typename T>
__global__ void maxpool2d_backward_kernel(
    const int n, const T* top_diff, const T* argmax_data,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    T* bottom_diff) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int b = index / pooled_width / pooled_height / channels;
        
        int maxidx = (int)argmax_data[index];
        if (maxidx != -1) {
            T* bottom_slice = bottom_diff + (b * channels + c) * height * width;
            atomicAdd(&bottom_slice[maxidx], top_diff[index]);
        }
    }
}

    /// Computes the forward pass for 2D max pooling on the GPU.
    template <typename T>
    void maxpool2d_forward_gpu(const Tensor<T>& bottom, Tensor<T>& top, Tensor<T>& argmax, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
        int batch_size = bottom._shape[0];
        int channels = bottom._shape[1];
        int height = bottom._shape[2];
        int width = bottom._shape[3];
        
        int pooled_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        int pooled_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        
#ifdef __CUDACC__
        int num_kernels = batch_size * channels * pooled_height * pooled_width;
        int threads = 256;
        int blocks = (num_kernels + threads - 1) / threads;
        maxpool2d_forward_kernel<<<blocks, threads>>>(
            num_kernels, bottom.gpu_data->ptr,
            channels, height, width, pooled_height, pooled_width,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            top.gpu_data->ptr, argmax.gpu_data->ptr
        );
#endif
    }

    /// Backpropagates max pooling gradients using saved argmax indices.
    template <typename T>
    void maxpool2d_backward_gpu(const Tensor<T>& top_diff, const Tensor<T>& argmax, Tensor<T>& bottom_diff, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
        int batch_size = bottom_diff._shape[0];
        int channels = bottom_diff._shape[1];
        int height = bottom_diff._shape[2];
        int width = bottom_diff._shape[3];
        
        int pooled_height = top_diff._shape[2];
        int pooled_width = top_diff._shape[3];
        
#ifdef __CUDACC__
        int num_kernels = batch_size * channels * pooled_height * pooled_width;
        int threads = 256;
        int blocks = (num_kernels + threads - 1) / threads;
        maxpool2d_backward_kernel<<<blocks, threads>>>(
            num_kernels, top_diff.gpu_data->ptr, argmax.gpu_data->ptr,
            channels, height, width, pooled_height, pooled_width,
            bottom_diff.gpu_data->ptr
        );
#endif
    }
}

namespace vc {
    template void maxpool2d_forward_gpu<float>(const Tensor<float>&, Tensor<float>&, Tensor<float>&, int, int, int, int, int, int);
    template void maxpool2d_backward_gpu<float>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, int, int, int, int, int, int);
}

namespace vc {
    template class Tensor<float>;
}
