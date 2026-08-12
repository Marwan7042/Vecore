#include "vecore/Tensor.h"
#include "vecore/nn.h"
#include "../TensorKernels.cuh"

namespace vc {
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

}

namespace vc {
    template class Tensor<float>;
}
