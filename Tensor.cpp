#include "Tensor.h"
#include <iostream>
#include <stdexcept>
#include <cblas.h>
#include <type_traits>

namespace mstd {
    template <typename T>
    Tensor<T>::Tensor() { 
        ctx = std::make_shared<AutogradContext<T>>(); 
    }

    template <typename T>
    Tensor<T>::Tensor(mstd::vector<size_t> s, mstd::vector<size_t> st, std::shared_ptr<mstd::vector<T>> d, std::shared_ptr<AutogradContext<T>> c, bool cuda, std::shared_ptr<GPUData<T>> gd) 
        : _shape(s), _strides(st), data(d), ctx(c), is_cuda(cuda), gpu_data(gd) {}

    template <typename T>
    Tensor<T>::Tensor(mstd::vector<size_t> shape) : _shape(shape) {
        size_t size = 1;
        for (size_t dim : _shape) size *= dim;
        data = std::make_shared<mstd::vector<T>>(size);
        for(size_t i = 0; i < size; i++) (*data)[i] = T(); // Zero memory!
        
        ctx = std::make_shared<AutogradContext<T>>();
        
        _strides = mstd::vector<size_t>(_shape.size());
        if(!_shape.isEmpty()) {
            size_t last_idx = _shape.size();
            _strides[last_idx-1] = 1;
            for(int i = last_idx - 2; i >= 0; i--) 
                _strides[i] = _strides[i+1] * _shape[i+1];
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::to(const std::string& device) const {
        Tensor<T> result(this->_shape);
        if (device == "cuda" && !this->is_cuda) {
            result.is_cuda = true;
            result.gpu_data = std::make_shared<GPUData<T>>(this->data->size());
            cudaMemcpy(result.gpu_data->ptr, this->data->begin(), this->data->size() * sizeof(T), cudaMemcpyHostToDevice);
            // IMPORTANT: We must also preserve the CPU data (e.g. Xavier init) so SGD has the right starting point!
            for (size_t i = 0; i < this->data->size(); i++) (*result.data)[i] = (*this->data)[i];
        } else if (device == "cpu" && this->is_cuda) {
            result.is_cuda = false;
            cudaMemcpy(result.data->begin(), this->gpu_data->ptr, this->data->size() * sizeof(T), cudaMemcpyDeviceToHost);
        } else {
            return *this;
        }
        result.ctx->requires_grad = this->ctx->requires_grad;
        return result;
    }

    template <typename T>
    T& Tensor<T>::operator()(const mstd::vector<size_t>& dims) { 
        size_t idx = 0;
        for (size_t i = 0; i < dims.size(); i++) idx += dims[i] * _strides[i];

        return (*data)[idx];
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose() const {
        mstd::vector<size_t> s(_shape.size());
        mstd::vector<size_t> st(_strides.size());

        for (int i = _shape.size() - 1, j = 0; i >= 0; i--, j++) {
            s[j] = _shape[i];
            st[j] = _strides[i]; 
        }
        
        return Tensor(s, st, this->data, this->ctx, this->is_cuda, this->gpu_data);
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
        if (!(this->_shape == other._shape)) throw std::invalid_argument("Shape mismatch.");

        Tensor<T> tensor(this->_shape);
        for (size_t i = 0; i < data->size(); i++)
            (*tensor.data)[i] = (*this->data)[i] + (*other.data)[i];

        tensor.ctx->requires_grad = (this->ctx->requires_grad || other.ctx->requires_grad);

        if (tensor.ctx->requires_grad)
            tensor.ctx->creator = std::make_shared<AddNode<T>>(*this, other, tensor);

        return tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
        if (!(this->_shape == other._shape)) throw std::invalid_argument("Shape mismatch.");

        Tensor<T> tensor(this->_shape);
        for (size_t i = 0; i < data->size(); i++)
            (*tensor.data)[i] = (*this->data)[i] - (*other.data)[i];

        tensor.ctx->requires_grad = (this->ctx->requires_grad || other.ctx->requires_grad);

        if (tensor.ctx->requires_grad)
            tensor.ctx->creator = std::make_shared<AddNode<T>>(*this, other, tensor);

        return tensor;
    }


    template <typename T>                                                                                                       
    Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {                                                              
        if (this->_shape.size() < 2 || other._shape.size() < 2) {                                                               
            throw std::invalid_argument("Tensors must be at least 2D for matmul.");                                             
        }                                                                                                                       
                                                                                                                                
        size_t M = this->_shape[this->_shape.size() - 2];                                                                       
        size_t K = this->_shape[this->_shape.size() - 1]; // Last dimension of A                                                                  
                                                                                                                                
        size_t other_K = other._shape[other._shape.size() - 2];                                                                 
        size_t N = other._shape[other._shape.size() - 1]; // Last dimension of B                                                                  
                                                                                                                                
        if (K != other_K) throw std::invalid_argument("Inner dimensions must match for matmul.");                               
                                                                                                                                
        // Calculate total batches and build the result shape                                                                   
        size_t batches = 1;                                                                                                     
        mstd::vector<size_t> result_shape(this->_shape.size());                                                                 
        for (size_t i = 0; i < this->_shape.size() - 2; i++) {                                                                  
            if (this->_shape[i] != other._shape[i]) throw std::invalid_argument("Batch shapes must match.");                    
            batches *= this->_shape[i];                                                                                         
            result_shape[i] = this->_shape[i];                                                                                  
        }                                                                                                                       
        result_shape[result_shape.size() - 2] = M;                                                                              
        result_shape[result_shape.size() - 1] = N;                                                                              
                                                                                                                                
        Tensor<T> result(result_shape);                                                                                         
                                                                                                                                
        // If a 2D tensor has _strides[0] == 1, it means it is a transposed view!
        bool transA = (this->_shape.size() == 2 && this->_strides[0] == 1);
        bool transB = (other._shape.size() == 2 && other._strides[0] == 1);
        
        // The leading dimension is always the major stride (columns of original physical layout)
        int ldA = transA ? this->_strides[1] : this->_strides[0];
        int ldB = transB ? other._strides[1] : other._strides[0];

        // THE CUBLAS MAGIC
        if (this->is_cuda && other.is_cuda) {
            result.is_cuda = true;
            result.gpu_data = std::make_shared<GPUData<T>>(result.data->size());
            
            cublasHandle_t handle;
            cublasCreate(&handle);
            float alpha = 1.0f;
            float beta = 0.0f;
            
            cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
            
            for (size_t b = 0; b < batches; b++) {
                // cuBLAS is column-major. To compute C = A * B in Row-Major:
                // We compute C^T = B^T * A^T in Column-Major. 
                // So we pass opB, opA.
                cublasSgemm(handle, opB, opA,
                            N, M, K,
                            &alpha,
                            other.gpu_data->ptr + (b * K * N), ldB,
                            this->gpu_data->ptr + (b * M * K), ldA,
                            &beta,
                            result.gpu_data->ptr + (b * M * N), N);
            }
            cublasDestroy(handle);
            
            // Sync to CPU
            cudaMemcpy(result.data->begin(), result.gpu_data->ptr, result.data->size() * sizeof(T), cudaMemcpyDeviceToHost);
        }
        // THE OPENBLAS MAGIC                                                                                                   
        else if constexpr (std::is_same_v<T, float>) {                                                                               
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

        result.ctx->requires_grad = (this->ctx->requires_grad || other.ctx->requires_grad);

        if (result.ctx->requires_grad)
            result.ctx->creator = std::make_shared<MulNode<T>>(*this, other, result);

        return result;
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
        Tensor<T> result(this->_shape);
        for (size_t i = 0; i < this->data->size(); i++){
            T val = (*this->data)[i];
            (*result.data)[i] = val > 0 ? val : 0;
        }
        
        result.ctx->requires_grad = this->ctx->requires_grad;
        if (result.ctx->requires_grad) 
            result.ctx->creator = std::make_shared<ReLU<T>>(*this, result);
        
        return result;
    }

    template <typename T>
    void build_topo(Tensor<T>& t, mstd::unordered_map<AutogradNode<T>*, bool>& visited, mstd::vector<Tensor<T>>& topo) {
        if (t.ctx->creator && !visited.contains(t.ctx->creator.get())) {
            visited.insert(t.ctx->creator.get(), true);
            for (Tensor<T>& parent : t.ctx->creator->get_parents()) {
                build_topo(parent, visited, topo);
            }
            topo.push_back(t);
        }
    }

    template <typename T>
    void Tensor<T>::backward() {
        this->ctx->grad = std::make_shared<Tensor<T>>(this->_shape);
        for (size_t i = 0; i < this->ctx->grad->data->size(); i++) 
            (*this->ctx->grad->data)[i] = 1.0f;

        mstd::vector<Tensor<T>> topo;
        mstd::unordered_map<AutogradNode<T>*, bool> visited;
        build_topo(*this, visited, topo);

        for (int i = topo.size() - 1; i >= 0; i--) {
            topo[i].ctx->creator->backward();
        }   
    }
} 

template class mstd::Tensor<float>;
