#include "vecore/Tensor.h"
#include "vecore/nn.h"
#include "../TensorKernels.cuh"

namespace vc {
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

}

namespace vc {
    template class Tensor<float>;
}
