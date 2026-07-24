#include "vecore/Tensor.h"
#include <iostream>
#include <stdexcept>
#include <cblas.h>
#include <type_traits>

namespace vc {
    template <typename T>
    bool AutogradContext<T>::grad_mode = true;

    template <typename T>
    Tensor<T>::Tensor() { 
        _numel = 0;
        ctx = std::make_shared<AutogradContext<T>>(); 
    }

    template <typename T>
    Tensor<T>::Tensor(vc::vector<size_t> s, vc::vector<size_t> st, std::shared_ptr<vc::vector<T>> d, std::shared_ptr<AutogradContext<T>> c, bool cuda, std::shared_ptr<GPUData<T>> gd) 
        : _shape(s), _strides(st), data(d), ctx(c), is_cuda(cuda), gpu_data(gd) {
        _numel = 1;
        for (size_t dim : _shape) _numel *= dim;
        if (_shape.size() == 0) _numel = 0;
    }

    template <typename T>
    Tensor<T>::Tensor(vc::vector<size_t> shape) : Tensor(shape, true) {}

    template <typename T>
    Tensor<T>::Tensor(vc::vector<size_t> shape, bool zero_memory, bool allocate_cpu) : _shape(shape) {
        _numel = 1;
        for (size_t dim : _shape) _numel *= dim;
        if (_shape.size() == 0) _numel = 0;
        
        size_t size = _numel;
        if (allocate_cpu) {
            data = std::make_shared<vc::vector<T>>(size);
            if (zero_memory) {
                for(size_t i = 0; i < size; i++) (*data)[i] = T(); // Zero memory!
            }
        } else {
            data = nullptr;
        }
        
        ctx = std::make_shared<AutogradContext<T>>();
        
        _strides = vc::vector<size_t>(_shape.size());
        if(!_shape.isEmpty()) {
            size_t last_idx = _shape.size();
            _strides[last_idx-1] = 1;
            for(int i = last_idx - 2; i >= 0; i--) 
                _strides[i] = _strides[i+1] * _shape[i+1];
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::empty_gpu(const vc::vector<size_t>& shape) {
        Tensor<T> result(shape, false, false); // Do not allocate CPU memory!
        result.is_cuda = true;
        result.gpu_data = std::make_shared<GPUData<T>>(result.numel());
        return result;
    }

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

__global__ void cuda_cross_entropy_metrics_kernel(const float* logits, const float* target, float* grad, float* batch_loss, int* batch_correct, int batch_size, int num_classes) {
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
        local_loss = -log(prob_target);
        
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

    template <typename T>
    Tensor<T> Tensor<T>::to(const std::string& device) const {
        Tensor<T> result(this->_shape);
        if (device == "cuda" && !this->is_cuda) {
            result.is_cuda = true;
            result.gpu_data = std::make_shared<GPUData<T>>(this->numel());
            cudaMemcpy(result.gpu_data->ptr, this->data->begin(), this->numel() * sizeof(T), cudaMemcpyHostToDevice);
            // IMPORTANT: We must also preserve the CPU data (e.g. Xavier init) so SGD has the right starting point!
            for (size_t i = 0; i < this->numel(); i++) (*result.data)[i] = (*this->data)[i];
        } else if (device == "cpu" && this->is_cuda) {
            result.is_cuda = false;
            cudaMemcpy(result.data->begin(), this->gpu_data->ptr, this->numel() * sizeof(T), cudaMemcpyDeviceToHost);
        } else {
            return *this;
        }
        result.ctx->requires_grad = AutogradContext<T>::grad_mode && this->ctx->requires_grad;
        if (result.ctx->requires_grad) {
            result.ctx->creator = std::make_shared<ToNode<T>>(*this, result);
        }
        return result;
    }

    template <typename T>
    T& Tensor<T>::operator()(const vc::vector<size_t>& coordinates) { 
        size_t idx = 0;
        for (size_t i = 0; i < coordinates.size(); i++) idx += coordinates[i] * _strides[i];

        return (*data)[idx];
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose() const {
        vc::vector<size_t> s(_shape.size());
        vc::vector<size_t> st(_strides.size());

        for (int i = _shape.size() - 1, j = 0; i >= 0; i--, j++) {
            s[j] = _shape[i];
            st[j] = _strides[i]; 
        }
        
        return Tensor(s, st, this->data, this->ctx, this->is_cuda, this->gpu_data);
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
        Tensor<T> tensor;
        if (this->is_cuda && other.is_cuda) {
            BroadcastResult<T> b_info = compute_broadcast<T>(this->_shape, other._shape, this->_strides, other._strides);
            tensor = Tensor<T>::empty_gpu(b_info.res_shape);
#ifdef __CUDACC__
            CudaBroadcastInfo info;
            info.rank = tensor._shape.size();
            for (int i=0; i<info.rank; i++) {
                info.shape[i] = b_info.res_shape[i];
                info.stridesA[i] = b_info.stridesA[i];
                info.stridesB[i] = b_info.stridesB[i];
            }
            int threads = 256;
            int blocks = (tensor.numel() + threads - 1) / threads;
            if (info.rank == 2 && info.stridesA[0] == info.shape[1] && info.stridesA[1] == 1 && info.stridesB[0] == 0 && info.stridesB[1] == 1) {
                cuda_add_bias_2d_kernel<<<blocks, threads>>>(this->gpu_data->ptr, other.gpu_data->ptr, tensor.gpu_data->ptr, tensor.numel(), info.shape[1]);
            } else {
                cuda_broadcast_add_kernel<<<blocks, threads>>>(this->gpu_data->ptr, other.gpu_data->ptr, tensor.gpu_data->ptr, info, tensor.numel());
            }
#endif
        } else {
            BroadcastResult<T> b_info = compute_broadcast<T>(this->_shape, other._shape, this->_strides, other._strides);
            tensor = Tensor<T>(b_info.res_shape);
            for (int i=0; i<tensor.numel(); i++) (*tensor.data)[i] = 0;
            tensor = broadcast_operation(*this, other, [](T a, T b) { return a + b; });
        }

        if (AutogradContext<T>::grad_mode && (this->ctx->requires_grad || other.ctx->requires_grad)) {
            tensor.ctx->requires_grad = true;
            tensor.ctx->creator = std::make_shared<AddNode<T>>(*this, other, tensor);
        }

        return tensor;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other) {
        if (this->is_cuda && other.is_cuda) {
            // Since this is primarily used for gradients, assume same shape for extreme performance
#ifdef __CUDACC__
            int threads = 256;
            int blocks = (this->numel() + threads - 1) / threads;
            cuda_add_inplace_kernel<<<blocks, threads>>>(this->gpu_data->ptr, other.gpu_data->ptr, this->numel());
#endif
        } else {
            for (size_t i = 0; i < this->numel(); i++) {
                (*this->data)[i] += (*other.data)[i];
            }
        }
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other) {
        if (this->is_cuda && other.is_cuda) {
#ifdef __CUDACC__
            int threads = 256;
            int blocks = (this->numel() + threads - 1) / threads;
            cuda_sub_inplace_kernel<<<blocks, threads>>>(this->gpu_data->ptr, other.gpu_data->ptr, this->numel());
#endif
        } else {
            for (size_t i = 0; i < this->numel(); i++) {
                (*this->data)[i] -= (*other.data)[i];
            }
        }
        return *this;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
        Tensor<T> tensor = broadcast_operation(*this, other, [](T a, T b) { return a - b; });

        if (tensor.ctx->requires_grad)
            tensor.ctx->creator = std::make_shared<SubNode<T>>(*this, other, tensor);

        return tensor;
    }


    template <typename T>                                                                                                       
    Tensor<T> Tensor<T>::matmul(const Tensor<T>& other, bool transA_arg, bool transB_arg) const {                                                              
        if (this->_shape.size() < 2 || other._shape.size() < 2) {                                                               
            throw std::invalid_argument("Tensors must be at least 2D for matmul.");                                             
        }                                                                                                                       
        size_t M = transA_arg ? this->_shape[this->_shape.size() - 1] : this->_shape[this->_shape.size() - 2];
        size_t K = transA_arg ? this->_shape[this->_shape.size() - 2] : this->_shape[this->_shape.size() - 1];
                                                                                                                                
        size_t other_K = transB_arg ? other._shape[other._shape.size() - 1] : other._shape[other._shape.size() - 2];
        size_t N = transB_arg ? other._shape[other._shape.size() - 2] : other._shape[other._shape.size() - 1];
                                                                                                                                
        if (K != other_K) throw std::invalid_argument("Inner dimensions must match for matmul.");                               
                                                                                                                                
        // Calculate total batches and build the result shape                                                                   
        size_t batches = 1;                                                                                                     
        vc::vector<size_t> result_shape(this->_shape.size());                                                                 
        for (size_t i = 0; i < this->_shape.size() - 2; i++) {                                                                  
            if (this->_shape[i] != other._shape[i]) throw std::invalid_argument("Batch shapes must match.");                    
            batches *= this->_shape[i];                                                                                         
            result_shape[i] = this->_shape[i];                                                                                  
        }                                                                                                                       
        result_shape[result_shape.size() - 2] = M;
        result_shape[result_shape.size() - 1] = N;
        
        Tensor<T> result;                                                                                                                                
        // If a 2D tensor has _strides[0] == 1, it means it is a transposed view!
        bool transA = transA_arg || (this->_shape.size() == 2 && this->_strides[0] == 1);
        bool transB = transB_arg || (other._shape.size() == 2 && other._strides[0] == 1);
        
        // The leading dimension is always the major stride (columns of original physical layout)
        int ldA = this->_strides[0];
        int ldB = other._strides[0];

        // THE CUBLAS MAGIC
        if (this->is_cuda && other.is_cuda) {
            result = Tensor<T>::empty_gpu(result_shape);
            
            static thread_local cublasHandle_t handle = nullptr;
            if (handle == nullptr) {
                cublasCreate(&handle);
                cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
            }
            float alpha = 1.0f;
            float beta = 0.0f;
            
            cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
            
            for (size_t b = 0; b < batches; b++) {
                // cuBLAS is column-major. To compute C = A * B in Row-Major:
                // We compute C^T = B^T * A^T in Column-Major. 
                // So we pass opB, opA.
                cublasSgemm(handle, 
                             opB, 
                             opA,
                             N, M, K,
                             &alpha, 
                             other.gpu_data->ptr + (b * K * N), ldB,
                             this->gpu_data->ptr + (b * M * K), ldA,
                             &beta,
                             result.gpu_data->ptr + (b * M * N), N);
            }
            // No cublasDestroy(handle) - we keep it alive to avoid sync overhead!
            // No cudaMemcpyDeviceToHost! Data stays on GPU!
        }
        // THE OPENBLAS MAGIC                                                                                                   
        else if constexpr (std::is_same_v<T, float>) {
            result = Tensor<T>(result_shape, false); // Don't zero memory for BLAS
            float* A_ptr = this->data->begin();                                                                                 
            float* B_ptr = other.data->begin();                                                                                 
            float* C_ptr = result.data->begin();                                                                                
            
            CBLAS_TRANSPOSE cblas_opA = transA ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE cblas_opB = transB ? CblasTrans : CblasNoTrans;

            for (size_t b = 0; b < batches; b++) {                                                                              
                cblas_sgemm(CblasRowMajor, cblas_opA, cblas_opB,                                                          
                            M, N, K,                                                                                            
                            1.0f,                                                                                               
                            A_ptr + (b * M * K), ldA,                                                                             
                            B_ptr + (b * K * N), ldB,                                                                             
                            0.0f,                                                                                               
                            C_ptr + (b * M * N), N);                                                                            
            }                                                                                                                   
        } else {                                                                                                                
            throw std::runtime_error("Only float32 matmul is supported via BLAS currently.");                               
        }

        result.ctx->requires_grad = AutogradContext<T>::grad_mode && (this->ctx->requires_grad || other.ctx->requires_grad);

        if (result.ctx->requires_grad)
            result.ctx->creator = std::make_shared<MulNode<T>>(*this, other, result, transA, transB);

        return result;
    }

    template <typename T>
    void Tensor<T>::matmul_accumulate(const Tensor<T>& other, Tensor<T>& accum, bool transA, bool transB) const {
        size_t M = transA ? this->_shape[this->_shape.size() - 1] : this->_shape[this->_shape.size() - 2];
        size_t K = transA ? this->_shape[this->_shape.size() - 2] : this->_shape[this->_shape.size() - 1];
        size_t N = transB ? other._shape[other._shape.size() - 2] : other._shape[other._shape.size() - 1];
        
        size_t batches = 1;
        if (this->_shape.size() > 2) {
            batches = this->_shape[0]; // assuming 3D max
        }

        int ldA = transA ? M : K; // Leading dimension of A (columns in row-major)
        int ldB = transB ? K : N; // Leading dimension of B
        int ldC = N;

        if (this->is_cuda && other.is_cuda) {
#ifdef __CUDACC__
            static thread_local cublasHandle_t handle = nullptr;
            if (handle == nullptr) {
                cublasCreate(&handle);
                cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
            }
            float alpha = 1.0f;
            float beta = 1.0f; // ACCUMULATE IN PLACE!
            
            cublasOperation_t opA_cublas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t opB_cublas = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
            
            for (size_t b = 0; b < batches; b++) {
                // To compute C = A * B in Row-Major, we do C^T = B^T * A^T in Col-Major
                // which means cublasSgemm(opB, opA, N, M, K, ..., B, A, C)
                cublasSgemm(handle, opB_cublas, opA_cublas,
                            N, M, K,
                            &alpha,
                            other.gpu_data->ptr + (b * (transB ? N * K : K * N)), ldB,
                            this->gpu_data->ptr + (b * (transA ? K * M : M * K)), ldA,
                            &beta,
                            accum.gpu_data->ptr + (b * M * N), ldC);
            }
#endif
        } else {
            // CPU Fallback
            float* A_ptr = this->data->begin();
            float* B_ptr = other.data->begin();
            float* C_ptr = accum.data->begin();
            
            CBLAS_TRANSPOSE cblas_opA = transA ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE cblas_opB = transB ? CblasTrans : CblasNoTrans;

            for (size_t b = 0; b < batches; b++) {
                cblas_sgemm(CblasRowMajor, cblas_opA, cblas_opB,
                            M, N, K,
                            1.0f,
                            A_ptr + (b * M * K), ldA,
                            B_ptr + (b * K * N), ldB,
                            1.0f, // ACCUMULATE
                            C_ptr + (b * M * N), N);
            }
        }
    }

    template <typename T>                                                                                                       
    void Tensor<T>::print() const {                                                                                             
        std::cout << "Tensor(shape=[";                                                                                          
        for (size_t i = 0; i < _shape.size(); i++) {                                                                            
            std::cout << _shape[i] << (i == _shape.size() - 1 ? "" : ", ");                                                     
        }                                                                                                                       
        std::cout << "])\n[ ";                                                                                                  
        for (size_t i = 0; i < data->size(); i++) {                                                                             
            std::cout << (*data)[i] << " ";                                                                                     
        }                                                                                                                       
        std::cout << "]\n\n";                                                                                                   
    }   

    template <typename T>
    Tensor<T> Tensor<T>::relu() const {
        Tensor<T> result;
        if (this->is_cuda) {
            result = Tensor<T>::empty_gpu(this->_shape);
#ifdef __CUDACC__
            int threads = 256;
            int blocks = (result.numel() + threads - 1) / threads;
            cuda_relu_forward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, result.gpu_data->ptr, result.numel());
#endif
        } else {
            result = Tensor<T>(this->_shape);
            for (size_t i = 0; i < this->numel(); i++){
                T val = (*this->data)[i];
                (*result.data)[i] = val > 0 ? val : 0;
            }
        }
        
        result.ctx->requires_grad = AutogradContext<T>::grad_mode && this->ctx->requires_grad;
        if (result.ctx->requires_grad) 
            result.ctx->creator = std::make_shared<ReLU<T>>(*this, result);
        
        return result;
    }
    template <typename T>
    Tensor<T> Tensor<T>::softmax(int dim) const {
        Tensor<T> result;
        size_t batch_size = this->_shape.size() > 1 ? this->_shape[0] : 1;
        size_t num_classes = this->_shape.size() > 1 ? this->_shape[1] : this->_shape[0];
        
        if (this->is_cuda) {
            result = Tensor<T>::empty_gpu(this->_shape);
#ifdef __CUDACC__
            int threads = 256;
            int blocks = (batch_size + threads - 1) / threads;
            cuda_softmax_forward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, result.gpu_data->ptr, batch_size, num_classes);
#endif
        } else {
            result = Tensor<T>(this->_shape);
            for (size_t b = 0; b < batch_size; b++) {
                T max_val = (*this->data)[b * num_classes];
                for (size_t j = 1; j < num_classes; j++) {
                    if ((*this->data)[b * num_classes + j] > max_val) max_val = (*this->data)[b * num_classes + j];
                }
                T sum_exp = 0.0f;
                for (size_t j = 0; j < num_classes; j++) {
                    T exp_val = std::exp((*this->data)[b * num_classes + j] - max_val);
                    (*result.data)[b * num_classes + j] = exp_val;
                    sum_exp += exp_val;
                }
                for (size_t j = 0; j < num_classes; j++) {
                    (*result.data)[b * num_classes + j] /= sum_exp;
                }
            }
        }
        
        result.ctx->requires_grad = AutogradContext<T>::grad_mode && this->ctx->requires_grad;
        if (result.ctx->requires_grad) {
            throw std::runtime_error("Softmax backward is not implemented standalone. Please use CrossEntropyLoss for training.");
        }
        
        return result;
    }
    template <typename T>
    void Tensor<T>::relu_backward(const Tensor<T>& out_grad, Tensor<T>& accum) const {
        if (this->is_cuda && out_grad.is_cuda) {
#ifdef __CUDACC__
            int threads = 256;
            int blocks = (this->numel() + threads - 1) / threads;
            cuda_relu_backward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, out_grad.gpu_data->ptr, accum.gpu_data->ptr, this->numel());
#endif
        } else {
            for (size_t i = 0; i < this->numel(); i++) {
                if ((*this->data)[i] > 0) (*accum.data)[i] += (*out_grad.data)[i];
            }
        }
    }

    template <typename T>
    void Tensor<T>::sgd_update(float lr) {
        if (this->is_cuda && this->ctx->grad && this->ctx->grad->is_cuda) {
#ifdef __CUDACC__
            int threads = 256;
            int blocks = (this->numel() + threads - 1) / threads;
            cuda_sgd_kernel<<<blocks, threads>>>(this->gpu_data->ptr, this->ctx->grad->gpu_data->ptr, lr, this->numel());
#endif
        } else {
            for (size_t i = 0; i < this->numel(); i++) {
                (*this->data)[i] -= lr * (*this->ctx->grad->data)[i];
            }
        }
    }

    template <typename T>
    Tensor<T>& Tensor<T>::zero_() {
        if (this->is_cuda) {
#ifdef __CUDACC__
            cudaMemsetAsync(this->gpu_data->ptr, 0, this->numel() * sizeof(T));
#endif
        } else {
            std::fill(this->data->begin(), this->data->end(), (T)0);
        }
        return *this;
    }

    template <typename T>
    void Tensor<T>::zero_grad_data() {
        if (this->ctx->grad) {
            if (this->ctx->grad->is_cuda) {
#ifdef __CUDACC__
                cudaMemsetAsync(this->ctx->grad->gpu_data->ptr, 0, this->numel() * sizeof(T));
#endif
            } else {
                for (size_t i = 0; i < this->ctx->grad->numel(); i++) {
                    (*this->ctx->grad->data)[i] = 0.0f;
                }
            }
        }
    }

    template <typename T>
    void build_topo(Tensor<T>& t, vc::unordered_map<AutogradNode<T>*, bool>& visited, vc::vector<Tensor<T>>& topo) {
        if (t.ctx->creator && !visited.contains(t.ctx->creator.get())) {
            visited.insert(t.ctx->creator.get(), true);
            for (Tensor<T>& parent : t.ctx->creator->get_parents()) {
                build_topo(parent, visited, topo);
            }
            topo.push_back(t);
        }
    }
    
    template <typename T>
    Tensor<T> Tensor<T>::reshape(const vc::vector<size_t>& new_shape) const {
        size_t new_size = 1;
        for (size_t i = 0; i < new_shape.size(); i++) new_size *= new_shape[i];
        
        if (new_size != this->numel()) {
            throw std::invalid_argument("Reshape cannot change the total number of elements.");
        }
        
        vc::vector<size_t> new_strides(new_shape.size());
        if (new_shape.size() > 0) {
            new_strides[new_shape.size() - 1] = 1;
            for(int i = (int)new_shape.size() - 2; i >= 0; i--) {
                new_strides[i] = new_strides[i+1] * new_shape[i+1];
            }
        }
        
        return Tensor<T>(new_shape, new_strides, this->data, this->ctx, this->is_cuda, this->gpu_data);
    }

    template <typename T>
    Tensor<T> Tensor<T>::sum(const vc::vector<int>& dims, bool keepdim) const {
        vc::vector<size_t> res_shape;
        vc::vector<bool> sum_dim(this->_shape.size());
        for(size_t i = 0; i < sum_dim.size(); i++) sum_dim[i] = false;
        for (size_t i = 0; i < dims.size(); i++) sum_dim[dims[i]] = true;
        
        for (size_t i = 0; i < this->_shape.size(); i++) {
            if (sum_dim[i]) {
                if (keepdim) res_shape.push_back(1);
            } else {
                res_shape.push_back(this->_shape[i]);
            }
        }
        
        Tensor<T> result;
        if (this->is_cuda) {
            result = Tensor<T>::empty_gpu(res_shape);
#ifdef __CUDACC__
            cudaMemsetAsync(result.gpu_data->ptr, 0, result.numel() * sizeof(T));
            CudaSumInfo info;
            info.rank = this->_shape.size();
            size_t res_dim_idx = 0;
            for (size_t d = 0; d < this->_shape.size(); d++) {
                info.in_shape[d] = this->_shape[d];
                if (sum_dim[d]) {
                    info.out_strides[d] = 0;
                    if (keepdim) res_dim_idx++;
                } else {
                    info.out_strides[d] = result._strides[res_dim_idx];
                    res_dim_idx++;
                }
            }
            if (this->_shape.size() == 2 && dims.size() == 1 && dims[0] == 0) {
                int threads = 256;
                int blocks = (this->_shape[1] + threads - 1) / threads;
                cuda_sum_kernel_2d_dim0<<<blocks, threads>>>(this->gpu_data->ptr, result.gpu_data->ptr, this->_shape[0], this->_shape[1]);
            } else {
                int threads = 256;
                int blocks = (this->numel() + threads - 1) / threads;
                cuda_sum_kernel_atomic<<<blocks, threads>>>(this->gpu_data->ptr, result.gpu_data->ptr, info, this->numel());
            }
#endif
        } else {
            result = Tensor<T>(res_shape); // Creates zeroed tensor!
        
            vc::vector<size_t> out_strides(this->_shape.size(), 0);
            size_t res_dim_idx = 0;
            for (size_t d = 0; d < this->_shape.size(); d++) {
                if (!sum_dim[d]) {
                    out_strides[d] = result._strides[res_dim_idx];
                    res_dim_idx++;
                } else if (keepdim) {
                    res_dim_idx++;
                }
            }

            vc::vector<size_t> curr_coords(this->_shape.size(), 0);
            size_t res_idx = 0;
            for (size_t i = 0; i < this->numel(); i++) {
                (*result.data)[res_idx] += (*this->data)[i];
                
                // Advance original coordinates and efficiently update res_idx
                for (int d = (int)this->_shape.size() - 1; d >= 0; d--) {
                    curr_coords[d]++;
                    if (curr_coords[d] < this->_shape[d]) {
                        res_idx += out_strides[d];
                        break;
                    }
                    curr_coords[d] = 0;
                    res_idx -= out_strides[d] * (this->_shape[d] - 1);
                }
            }
        }
        
        return result;
    }

    template <typename T>
    void Tensor<T>::backward() {
        AutogradContext<T>::grad_mode = false;
        
        if (!this->ctx->grad) {
            this->ctx->grad = std::make_shared<Tensor<T>>(this->_shape);
            for (size_t i = 0; i < this->ctx->grad->numel(); i++) 
                (*this->ctx->grad->data)[i] = 1.0f;
        }

        vc::vector<Tensor<T>> topo;
        vc::unordered_map<AutogradNode<T>*, bool> visited;
        build_topo(*this, visited, topo);

        for (int i = topo.size() - 1; i >= 0; i--) {
            topo[i].ctx->creator->backward();
        }   
        
        AutogradContext<T>::grad_mode = true;
    }

    template <typename T>
    void Tensor<T>::fast_cross_entropy_backward(const Tensor<T>& target, float* d_loss, int* d_correct) {
#ifdef __CUDACC__
        if (this->is_cuda && target.is_cuda) {
            Tensor<T> grad_tensor = Tensor<T>::empty_gpu(this->_shape);
            this->ctx->grad = std::make_shared<Tensor<T>>(grad_tensor);
            
            int batch_size = this->_shape.size() > 1 ? this->_shape[0] : 1;
            int num_classes = this->_shape[this->_shape.size() - 1];
            
            int threads = 256;
            int blocks = (batch_size + threads - 1) / threads;
            
            cuda_cross_entropy_metrics_kernel<<<blocks, threads>>>(
                this->gpu_data->ptr, target.gpu_data->ptr, this->ctx->grad->gpu_data->ptr, 
                d_loss, d_correct, batch_size, num_classes
            );
            
            this->backward(); // Trigger the full GPU backward pass!
        }
#endif
    }

    template <typename T>
    Tensor<T> Tensor<T>::sigmoid() const {
        Tensor<T> result;
        if (this->is_cuda) {
            result = Tensor<T>::empty_gpu(this->_shape);
#ifdef __CUDACC__
            int threads = 256; int blocks = (result.numel() + threads - 1) / threads;
            cuda_sigmoid_forward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, result.gpu_data->ptr, result.numel());
#endif
        } else {
            result = Tensor<T>(this->_shape);
            for (size_t i = 0; i < this->numel(); i++) (*result.data)[i] = 1.0f / (1.0f + std::exp(-(*this->data)[i]));
        }
        result.ctx->requires_grad = AutogradContext<T>::grad_mode && this->ctx->requires_grad;
        if (result.ctx->requires_grad) result.ctx->creator = std::make_shared<SigmoidNode<T>>(*this, result);
        return result;
    }

    template <typename T>
    void Tensor<T>::sigmoid_backward(const Tensor<T>& out_grad, Tensor<T>& accum) const {
        if (this->is_cuda) {
#ifdef __CUDACC__
            int threads = 256; int blocks = (this->numel() + threads - 1) / threads;
            cuda_sigmoid_backward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, out_grad.gpu_data->ptr, accum.gpu_data->ptr, this->numel());
#endif
        } else {
            for (size_t i = 0; i < this->numel(); i++) {
                T sig = 1.0f / (1.0f + std::exp(-(*this->data)[i]));
                (*accum.data)[i] += (*out_grad.data)[i] * sig * (1.0f - sig);
            }
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::tanh() const {
        Tensor<T> result;
        if (this->is_cuda) {
            result = Tensor<T>::empty_gpu(this->_shape);
#ifdef __CUDACC__
            int threads = 256; int blocks = (result.numel() + threads - 1) / threads;
            cuda_tanh_forward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, result.gpu_data->ptr, result.numel());
#endif
        } else {
            result = Tensor<T>(this->_shape);
            for (size_t i = 0; i < this->numel(); i++) (*result.data)[i] = std::tanh((*this->data)[i]);
        }
        result.ctx->requires_grad = AutogradContext<T>::grad_mode && this->ctx->requires_grad;
        if (result.ctx->requires_grad) result.ctx->creator = std::make_shared<TanhNode<T>>(*this, result);
        return result;
    }

    template <typename T>
    void Tensor<T>::tanh_backward(const Tensor<T>& out_grad, Tensor<T>& accum) const {
        if (this->is_cuda) {
#ifdef __CUDACC__
            int threads = 256; int blocks = (this->numel() + threads - 1) / threads;
            cuda_tanh_backward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, out_grad.gpu_data->ptr, accum.gpu_data->ptr, this->numel());
#endif
        } else {
            for (size_t i = 0; i < this->numel(); i++) {
                T t = std::tanh((*this->data)[i]);
                (*accum.data)[i] += (*out_grad.data)[i] * (1.0f - t * t);
            }
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::leaky_relu(float alpha) const {
        Tensor<T> result;
        if (this->is_cuda) {
            result = Tensor<T>::empty_gpu(this->_shape);
#ifdef __CUDACC__
            int threads = 256; int blocks = (result.numel() + threads - 1) / threads;
            cuda_leaky_relu_forward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, result.gpu_data->ptr, alpha, result.numel());
#endif
        } else {
            result = Tensor<T>(this->_shape);
            for (size_t i = 0; i < this->numel(); i++) {
                T val = (*this->data)[i];
                (*result.data)[i] = val > 0 ? val : val * alpha;
            }
        }
        result.ctx->requires_grad = AutogradContext<T>::grad_mode && this->ctx->requires_grad;
        if (result.ctx->requires_grad) result.ctx->creator = std::make_shared<LeakyReLUNode<T>>(*this, result, alpha);
        return result;
    }

    template <typename T>
    void Tensor<T>::leaky_relu_backward(const Tensor<T>& out_grad, Tensor<T>& accum, float alpha) const {
        if (this->is_cuda) {
#ifdef __CUDACC__
            int threads = 256; int blocks = (this->numel() + threads - 1) / threads;
            cuda_leaky_relu_backward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, out_grad.gpu_data->ptr, accum.gpu_data->ptr, alpha, this->numel());
#endif
        } else {
            for (size_t i = 0; i < this->numel(); i++) {
                (*accum.data)[i] += (*out_grad.data)[i] * ((*this->data)[i] > 0 ? 1.0f : alpha);
            }
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::gelu() const {
        Tensor<T> result;
        if (this->is_cuda) {
            result = Tensor<T>::empty_gpu(this->_shape);
#ifdef __CUDACC__
            int threads = 256; int blocks = (result.numel() + threads - 1) / threads;
            cuda_gelu_forward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, result.gpu_data->ptr, result.numel());
#endif
        } else {
            result = Tensor<T>(this->_shape);
            for (size_t i = 0; i < this->numel(); i++) {
                T x = (*this->data)[i];
                (*result.data)[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            }
        }
        result.ctx->requires_grad = AutogradContext<T>::grad_mode && this->ctx->requires_grad;
        if (result.ctx->requires_grad) result.ctx->creator = std::make_shared<GELUNode<T>>(*this, result);
        return result;
    }

    template <typename T>
    void Tensor<T>::gelu_backward(const Tensor<T>& out_grad, Tensor<T>& accum) const {
        if (this->is_cuda) {
#ifdef __CUDACC__
            int threads = 256; int blocks = (this->numel() + threads - 1) / threads;
            cuda_gelu_backward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, out_grad.gpu_data->ptr, accum.gpu_data->ptr, this->numel());
#endif
        } else {
            for (size_t i = 0; i < this->numel(); i++) {
                T x = (*this->data)[i];
                T x3 = x * x * x;
                T t = std::tanh(0.7978845608f * (x + 0.044715f * x3));
                T sech2 = 1.0f - t * t;
                T grad_val = 0.5f * (1.0f + t) + 0.5f * x * sech2 * (0.7978845608f * (1.0f + 0.134145f * x * x));
                (*accum.data)[i] += (*out_grad.data)[i] * grad_val;
            }
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::silu() const {
        Tensor<T> result;
        if (this->is_cuda) {
            result = Tensor<T>::empty_gpu(this->_shape);
#ifdef __CUDACC__
            int threads = 256; int blocks = (result.numel() + threads - 1) / threads;
            cuda_silu_forward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, result.gpu_data->ptr, result.numel());
#endif
        } else {
            result = Tensor<T>(this->_shape);
            for (size_t i = 0; i < this->numel(); i++) {
                T x = (*this->data)[i];
                (*result.data)[i] = x / (1.0f + std::exp(-x));
            }
        }
        result.ctx->requires_grad = AutogradContext<T>::grad_mode && this->ctx->requires_grad;
        if (result.ctx->requires_grad) result.ctx->creator = std::make_shared<SiLUNode<T>>(*this, result);
        return result;
    }

    template <typename T>
    void Tensor<T>::silu_backward(const Tensor<T>& out_grad, Tensor<T>& accum) const {
        if (this->is_cuda) {
#ifdef __CUDACC__
            int threads = 256; int blocks = (this->numel() + threads - 1) / threads;
            cuda_silu_backward_kernel<<<blocks, threads>>>(this->gpu_data->ptr, out_grad.gpu_data->ptr, accum.gpu_data->ptr, this->numel());
#endif
        } else {
            for (size_t i = 0; i < this->numel(); i++) {
                T x = (*this->data)[i];
                T sig = 1.0f / (1.0f + std::exp(-x));
                (*accum.data)[i] += (*out_grad.data)[i] * (sig + x * sig * (1.0f - sig));
            }
        }
    }
} 

template class vc::Tensor<float>;
