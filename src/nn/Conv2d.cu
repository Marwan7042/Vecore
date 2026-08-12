#include "vecore/Tensor.h"
#include <iostream>
#include <stdexcept>
#include <cblas.h>
#include <type_traits>
#include <cmath>

namespace vc {
template <typename T>
__global__ void im2col_kernel(
    const int n, const T* data_im,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    T* data_col) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        int w_out = index % width_col;
        int index_1 = index / width_col;
        int h_out = index_1 % height_col;
        int index_2 = index_1 / height_col;
        int channel_in = index_2 % channels;
        int batch = index_2 / channels;

        int channel_out = channel_in * kernel_h * kernel_w;
        int h_in = h_out * stride_h - pad_h;
        int w_in = w_out * stride_w - pad_w;

        T* col = data_col + (batch * channels * kernel_h * kernel_w * height_col * width_col) + 
                 (channel_out * height_col * width_col) + (h_out * width_col + w_out);
                 
        const T* im = data_im + (batch * channels * height * width) + (channel_in * height * width);

        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                *col = (h >= 0 && w >= 0 && h < height && w < width) ? im[h * width + w] : 0;
                col += height_col * width_col;
            }
        }
    }
}

template <typename T>
__global__ void col2im_kernel(
    const int n, const T* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    T* data_im) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        T val = data_col[index];
        int w_out = index % width_col;
        int index_1 = index / width_col;
        int h_out = index_1 % height_col;
        int index_2 = index_1 / height_col;
        int k_w = index_2 % kernel_w;
        int index_3 = index_2 / kernel_w;
        int k_h = index_3 % kernel_h;
        int index_4 = index_3 / kernel_h;
        int c = index_4 % channels;
        int b = index_4 / channels;
        
        int h_in = h_out * stride_h - pad_h + k_h;
        int w_in = w_out * stride_w - pad_w + k_w;
        
        if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
            T* im_ptr = data_im + (b * channels * height * width) + (c * height * width) + (h_in * width + w_in);
            atomicAdd(im_ptr, val);
        }
    }
}

template <typename T>
__global__ void cuda_add_conv_bias_kernel(T* out, const T* bias, int out_channels, int spatial_size, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = (idx / spatial_size) % out_channels;
        out[idx] += bias[c];
    }
}

template <typename T>
__global__ void cuda_grad_bias_kernel(const T* out_grad, T* grad_bias, int out_channels, int spatial_size, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = (idx / spatial_size) % out_channels;
        atomicAdd(&grad_bias[c], out_grad[idx]);
    }
}

    /// Extracts sliding window patches from an image batch into column form.
    template <typename T>
    Tensor<T> im2col_gpu(const Tensor<T>& im, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
        int batch_size = im._shape[0];
        int channels = im._shape[1];
        int height = im._shape[2];
        int width = im._shape[3];
        
        int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        
        vc::vector<size_t> out_shape(3);
        out_shape[0] = batch_size;
        out_shape[1] = channels * kernel_h * kernel_w;
        Tensor<T> col = Tensor<T>::empty_gpu(out_shape);
        
#ifdef __CUDACC__
        int num_kernels = batch_size * channels * height_col * width_col;
        int threads = 256;
        int blocks = (num_kernels + threads - 1) / threads;
        im2col_kernel<<<blocks, threads>>>(
            num_kernels, im.gpu_data->ptr,
            height, width, channels,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            height_col, width_col, col.gpu_data->ptr
        );
#endif
        return col;
    }

    /// Reassembles an image batch from column-form patches.
    template <typename T>
    void col2im_gpu(const Tensor<T>& col, Tensor<T>& im, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
        int batch_size = im._shape[0];
        int channels = im._shape[1];
        int height = im._shape[2];
        int width = im._shape[3];
        
        int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        
#ifdef __CUDACC__
        int num_kernels = batch_size * channels * kernel_h * kernel_w * height_col * width_col;
        int threads = 256;
        int blocks = (num_kernels + threads - 1) / threads;
        col2im_kernel<<<blocks, threads>>>(
            num_kernels, col.gpu_data->ptr,
            height, width, channels,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            height_col, width_col, im.gpu_data->ptr
        );
#endif
    }

    /// Computes the forward pass for 2D convolution on the GPU.
    template <typename T>
    Tensor<T> conv2d_forward_gpu(const Tensor<T>& input, const Tensor<T>& weight, const Tensor<T>& bias, int stride_h, int stride_w, int pad_h, int pad_w) {
        int batch_size = input._shape[0];
        int in_channels = input._shape[1];
        int in_h = input._shape[2];
        int in_w = input._shape[3];
        
        int out_channels = weight._shape[0];
        int kernel_h = weight._shape[2];
        int kernel_w = weight._shape[3];
        
        int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
        int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
        
        vc::vector<size_t> out_shape(4);
        out_shape[0] = batch_size; out_shape[1] = out_channels; out_shape[2] = out_h; out_shape[3] = out_w;
        
        Tensor<T> out = Tensor<T>::empty_gpu(out_shape);
        Tensor<T> col = im2col_gpu(input, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        
#ifdef __CUDACC__
        static thread_local cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
        }
        float alpha = 1.0f;
        float beta = 0.0f;
        
        int m = out_h * out_w;
        int n_mat = out_channels;
        int k = in_channels * kernel_h * kernel_w;
        
        for (int b = 0; b < batch_size; b++) {
            const T* B_ptr = col.gpu_data->ptr + b * (k * m);
            const T* A_ptr = weight.gpu_data->ptr;
            T* C_ptr = out.gpu_data->ptr + b * (n_mat * m);
            
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        m, n_mat, k,
                        &alpha,
                        B_ptr, m,
                        A_ptr, k,
                        &beta,
                        C_ptr, m);
        }
        
        if (bias.numel() > 0) {
            int threads = 256;
            int blocks = (out.numel() + threads - 1) / threads;
            cuda_add_conv_bias_kernel<<<blocks, threads>>>(out.gpu_data->ptr, bias.gpu_data->ptr, out_channels, out_h * out_w, out.numel());
        }
#endif
        return out;
    }

    /// Computes gradients for 2D convolution parameters and inputs.
    template <typename T>
    void conv2d_backward_gpu(const Tensor<T>& out_grad, const Tensor<T>& input, const Tensor<T>& weight, Tensor<T>& grad_input, Tensor<T>& grad_weight, Tensor<T>& grad_bias, int stride_h, int stride_w, int pad_h, int pad_w) {
        int batch_size = input._shape[0];
        int in_channels = input._shape[1];
        
        int out_channels = weight._shape[0];
        int kernel_h = weight._shape[2];
        int kernel_w = weight._shape[3];
        
        int out_h = out_grad._shape[2];
        int out_w = out_grad._shape[3];
        
        int m = out_h * out_w; 
        int n_mat = out_channels; 
        int k = in_channels * kernel_h * kernel_w; 
        
        Tensor<T> col = im2col_gpu(input, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        
        vc::vector<size_t> grad_col_shape(3);
        grad_col_shape[0] = batch_size;
        grad_col_shape[1] = in_channels * kernel_h * kernel_w;
        grad_col_shape[2] = out_h * out_w;
        
        Tensor<T> grad_col = Tensor<T>::empty_gpu(grad_col_shape);
#ifdef __CUDACC__
        cudaMemset(grad_col.gpu_data->ptr, 0, grad_col.numel() * sizeof(T));
        static thread_local cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
        }
        float alpha = 1.0f;
        float beta_weight = 1.0f;
        float beta_col = 0.0f;
        
        for (int b = 0; b < batch_size; b++) {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        k, n_mat, m,
                        &alpha,
                        col.gpu_data->ptr + b * (k * m), m,
                        out_grad.gpu_data->ptr + b * (n_mat * m), m,
                        &beta_weight,
                        grad_weight.gpu_data->ptr, k);
                        
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        m, k, n_mat,
                        &alpha,
                        out_grad.gpu_data->ptr + b * (n_mat * m), m,
                        weight.gpu_data->ptr, k,
                        &beta_col,
                        grad_col.gpu_data->ptr + b * (k * m), m);
        }
        
        if (grad_bias.numel() > 0) {
            int threads = 256;
            int blocks = (out_grad.numel() + threads - 1) / threads;
            cuda_grad_bias_kernel<<<blocks, threads>>>(out_grad.gpu_data->ptr, grad_bias.gpu_data->ptr, out_channels, out_h * out_w, out_grad.numel());
        }
#endif
        col2im_gpu(grad_col, grad_input, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    }
}

namespace vc {
    template Tensor<float> conv2d_forward_gpu<float>(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&, int, int, int, int);
    template void conv2d_backward_gpu<float>(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&, Tensor<float>&, Tensor<float>&, Tensor<float>&, int, int, int, int);
}

namespace vc {
    template class Tensor<float>;
}
