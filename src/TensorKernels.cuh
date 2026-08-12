#pragma once
#include "vecore/Tensor.h"
#include <cblas.h>
namespace vc {
#ifdef __CUDACC__
#define MAX_DIMS 8
struct CudaBroadcastInfo {
    int rank;
    size_t shape[MAX_DIMS];
    size_t stridesA[MAX_DIMS];
    size_t stridesB[MAX_DIMS];
};

struct CudaSumInfo {
    int rank;
    size_t in_shape[MAX_DIMS];
    size_t out_strides[MAX_DIMS];
};

template <typename T>
__global__ void cuda_sum_kernel_atomic(const T* in, T* out, CudaSumInfo info, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    size_t temp = idx;
    size_t out_idx = 0;
    for (int d = info.rank - 1; d >= 0; --d) {
        size_t c = temp % info.in_shape[d];
        out_idx += c * info.out_strides[d];
        temp /= info.in_shape[d];
    }
    
    atomicAdd(&out[out_idx], in[idx]);
}

template <typename T>
__global__ void cuda_sum_kernel_2d_dim0(const T* in, T* out, size_t B, size_t D) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < D) {
        T s = 0;
        for (size_t b = 0; b < B; b++) {
            s += in[b * D + d];
        }
        out[d] = s;
    }
}

template <typename T>
__global__ void cuda_broadcast_add_kernel(const T* A, const T* B, T* C, CudaBroadcastInfo info, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t temp = idx;
    size_t idxA = 0, idxB = 0;
    for (int d = info.rank - 1; d >= 0; --d) {
        size_t c = temp % info.shape[d];
        idxA += c * info.stridesA[d];
        idxB += c * info.stridesB[d];
        temp /= info.shape[d];
    }
    C[idx] = A[idxA] + B[idxB];
}

template <typename T>
__global__ void cuda_add_bias_2d_kernel(const T* A, const T* B, T* C, size_t total, size_t D_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        C[idx] = A[idx] + B[idx % D_dim];
    }
}

template <typename T>
__global__ void cuda_broadcast_sub_kernel(const T* A, const T* B, T* C, CudaBroadcastInfo info, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int coords[10]; int temp = idx;
    for (int i = info.rank - 1; i >= 0; i--) {
        coords[i] = temp % info.shape[i];
        temp /= info.shape[i];
    }
    int idxA = 0, idxB = 0;
    for (int i = 0; i < info.rank; i++) {
        idxA += coords[i] * info.stridesA[i];
        idxB += coords[i] * info.stridesB[i];
    }
    C[idx] = A[idxA] - B[idxB];
}

template <typename T>
__global__ void cuda_add_inplace_kernel(T* A, const T* B, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) A[idx] += B[idx];
}

template <typename T>
__global__ void cuda_sub_inplace_kernel(T* A, const T* B, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) A[idx] -= B[idx];
}

template <typename T>
__global__ void cuda_relu_forward_kernel(const T* A, T* C, size_t total) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        C[i] = A[i] > 0 ? A[i] : 0;
    }
}

template <typename T>
__global__ void cuda_softmax_forward_kernel(const T* in, T* out, size_t batch_size, size_t num_classes) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        T max_val = in[b * num_classes];
        for (size_t j = 1; j < num_classes; j++) {
            if (in[b * num_classes + j] > max_val) max_val = in[b * num_classes + j];
        }
        T sum_exp = 0;
        for (size_t j = 0; j < num_classes; j++) {
            T exp_val = exp(in[b * num_classes + j] - max_val);
            out[b * num_classes + j] = exp_val;
            sum_exp += exp_val;
        }
        for (size_t j = 0; j < num_classes; j++) {
            out[b * num_classes + j] /= sum_exp;
        }
    }
}
template <typename T>
__global__ void cuda_relu_backward_kernel(const T* A, const T* out_grad, T* A_grad, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) A_grad[idx] += (A[idx] > 0) ? out_grad[idx] : 0;
}

template <typename T>
__global__ void cuda_sigmoid_forward_kernel(const T* A, T* C, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) C[idx] = 1.0f / (1.0f + expf(-A[idx]));
}
template <typename T>
__global__ void cuda_sigmoid_backward_kernel(const T* A, const T* out_grad, T* A_grad, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        T sig = 1.0f / (1.0f + expf(-A[idx]));
        A_grad[idx] += out_grad[idx] * sig * (1.0f - sig);
    }
}

template <typename T>
__global__ void cuda_tanh_forward_kernel(const T* A, T* C, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) C[idx] = tanhf(A[idx]);
}
template <typename T>
__global__ void cuda_tanh_backward_kernel(const T* A, const T* out_grad, T* A_grad, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        T t = tanhf(A[idx]);
        A_grad[idx] += out_grad[idx] * (1.0f - t * t);
    }
}

template <typename T>
__global__ void cuda_leaky_relu_forward_kernel(const T* A, T* C, float alpha, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) C[idx] = A[idx] > 0 ? A[idx] : A[idx] * alpha;
}
template <typename T>
__global__ void cuda_leaky_relu_backward_kernel(const T* A, const T* out_grad, T* A_grad, float alpha, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) A_grad[idx] += out_grad[idx] * (A[idx] > 0 ? 1.0f : alpha);
}

template <typename T>
__global__ void cuda_gelu_forward_kernel(const T* A, T* C, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        T x = A[idx];
        C[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}
template <typename T>
__global__ void cuda_gelu_backward_kernel(const T* A, const T* out_grad, T* A_grad, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        T x = A[idx];
        T x3 = x * x * x;
        T t = tanhf(0.7978845608f * (x + 0.044715f * x3));
        T sech2 = 1.0f - t * t;
        T grad_val = 0.5f * (1.0f + t) + 0.5f * x * sech2 * (0.7978845608f * (1.0f + 0.134145f * x * x));
        A_grad[idx] += out_grad[idx] * grad_val;
    }
}

template <typename T>
__global__ void cuda_silu_forward_kernel(const T* A, T* C, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        T x = A[idx];
        C[idx] = x / (1.0f + expf(-x));
    }
}
template <typename T>
__global__ void cuda_silu_backward_kernel(const T* A, const T* out_grad, T* A_grad, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        T x = A[idx];
        T sig = 1.0f / (1.0f + expf(-x));
        A_grad[idx] += out_grad[idx] * (sig + x * sig * (1.0f - sig));
    }
}

template <typename T>
__global__ void cuda_sgd_kernel(T* weight, const T* grad, float lr, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    weight[idx] -= lr * grad[idx];
}

static __global__ void cuda_cross_entropy_metrics_kernel(const float* logits, const float* target, float* grad, float* batch_loss, int* batch_correct, int batch_size, int num_classes) {
    __shared__ float s_loss[256];
    __shared__ int s_correct[256];
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float local_loss = 0.0f;
    int local_correct = 0;
    
    if (b < batch_size) {
        float max_val = -1e9;
        int max_idx = 0;
        int target_idx = 0;
        
        for (int j = 0; j < num_classes; j++) {
            float val = logits[b * num_classes + j];
            if (val > max_val) { max_val = val; max_idx = j; }
            if (target[b * num_classes + j] == 1.0f) target_idx = j;
        }
        
        if (max_idx == target_idx) local_correct = 1;
        
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += exp(logits[b * num_classes + j] - max_val);
        }
        
        float prob_target = exp(logits[b * num_classes + target_idx] - max_val) / sum_exp;
        local_loss = -logf(prob_target + 1e-7f);
        
        if (grad != nullptr) {
            for (int j = 0; j < num_classes; j++) {
                float prob = exp(logits[b * num_classes + j] - max_val) / sum_exp;
                grad[b * num_classes + j] = (prob - target[b * num_classes + j]) / batch_size;
            }
        }
    }
    
    s_loss[tid] = local_loss;
    s_correct[tid] = local_correct;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_loss[tid] += s_loss[tid + stride];
            s_correct[tid] += s_correct[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(batch_loss, s_loss[0]);
        atomicAdd(batch_correct, s_correct[0]);
    }
}
#endif

    template <typename T> 
    struct BroadcastResult {
        vc::vector<size_t> stridesA;
        vc::vector<size_t> stridesB;
        vc::vector<size_t> res_shape;
    };

    template <typename T>
    BroadcastResult<T> compute_broadcast(const vc::vector<size_t>& shapeA,
                                       const vc::vector<size_t>& shapeB,
                                       const vc::vector<size_t>& stridesA,
                                       const vc::vector<size_t>& stridesB) {
        size_t rankA = shapeA.size();
        size_t rankB = shapeB.size();
        size_t max_rank = (rankA > rankB) ? rankA : rankB;

        BroadcastResult<T> res;

        res.res_shape = vc::vector<size_t>(max_rank);
        res.stridesA = vc::vector<size_t>(max_rank);
        res.stridesB = vc::vector<size_t>(max_rank);

        for (int i = 0; i < max_rank; i++) {
            int dimA_idx = rankA - 1 - i;                                                                                                                                                                                                  
            int dimB_idx = rankB - 1 - i;                                                                                                                                                                                                  
            int res_idx = max_rank - 1 - i;  

            size_t sizeA = (dimA_idx >= 0) ? shapeA[dimA_idx] : 1;                                                                                                                                                                         
            size_t sizeB = (dimB_idx >= 0) ? shapeB[dimB_idx] : 1;                                                                                                                                                                         
                                                                                                                                                                                                                                            
            size_t strideA = (dimA_idx >= 0) ? stridesA[dimA_idx] : 0;                                                                                                                                                                     
            size_t strideB = (dimB_idx >= 0) ? stridesB[dimB_idx] : 0;  

            if (sizeA != sizeB && sizeA != 1 && sizeB != 1) {                                                                                                                                                                              
                throw std::invalid_argument("Shape Mismatch Error: Tensors cannot be broadcasted together.");                                                                                                                              
            }    

            res.res_shape[res_idx] = (sizeA > sizeB) ? sizeA : sizeB;
            res.stridesA[res_idx] = (sizeA == 1 && res.res_shape[res_idx] > 1) ? 0 : strideA;                                                                                                                                              
            res.stridesB[res_idx] = (sizeB == 1 && res.res_shape[res_idx] > 1) ? 0 : strideB; 
        }

        return res;
    }

    template <typename T, typename Func>                                                                           
    Tensor<T> broadcast_operation(const Tensor<T>& A, const Tensor<T>& B, Func operation) {                        
        BroadcastResult<T> b_info = compute_broadcast<T>(A._shape, B._shape, A._strides, B._strides);              
                                                                                                                   
        Tensor<T> result(b_info.res_shape);                                                                        
        size_t total_elements = result.numel();                                                               
                                                                                                                   
        vc::vector<size_t> current_coords(b_info.res_shape.size(), 0);                                  
                                                                                                                   
        size_t idxA = 0;                                                                                           
        size_t idxB = 0;                                                                                           
                                                                                                                   
        for (size_t i = 0; i < total_elements; ++i) {                                                              
            // 1. Apply the operation using the lambda function                                                    
            (*result.data)[i] = operation((*A.data)[idxA], (*B.data)[idxB]);                                       
                                                                                                                   
            // 2. Fast Coordinate and Stride Advancement                                                           
            // We step backwards through the dimensions to mimic row-major traversal                               
            for (int dim = b_info.res_shape.size() - 1; dim >= 0; --dim) {                                         
                current_coords[dim]++;                                                                             
                idxA += b_info.stridesA[dim];                                                                      
                idxB += b_info.stridesB[dim];                                                                      
                                                                                                                   
                if (current_coords[dim] < b_info.res_shape[dim]) {                                                 
                    break; // No wrap-around needed, we are done advancing                                         
                }                                                                                                  
                                                                                                                   
                // Wrap-around happened! Reset this dimension to 0                                                 
                current_coords[dim] = 0;                                                                           
                // Roll back the pointers by exactly the amount we advanced in this dimension                      
                idxA -= b_info.stridesA[dim] * b_info.res_shape[dim];                                              
                idxB -= b_info.stridesB[dim] * b_info.res_shape[dim];                                              
            }                                                                                                      
        }                                                                                                          
                                                                                                                   
        result.ctx->requires_grad = AutogradContext<T>::grad_mode && (A.ctx->requires_grad || B.ctx->requires_grad);                                
        return result;                                                                                             
    }                        


}
