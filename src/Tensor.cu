#include "vecore/Tensor.h"
#include <iostream>
#include <stdexcept>
#include <cblas.h>
#include <type_traits>
#include "TensorKernels.cuh"

namespace vc {
    /// Stores the global autograd mode flag.
    template <typename T>
    bool AutogradContext<T>::grad_mode = true;

    /// Creates an empty tensor with a fresh autograd context.
    template <typename T>
    Tensor<T>::Tensor() { 
        _numel = 0;
        ctx = std::make_shared<AutogradContext<T>>(); 
    }

    /// Constructs a tensor from pre-existing storage, shape, and context.
    template <typename T>
    Tensor<T>::Tensor(vc::vector<size_t> s, vc::vector<size_t> st, std::shared_ptr<vc::vector<T>> d, std::shared_ptr<AutogradContext<T>> c, bool cuda, std::shared_ptr<GPUData<T>> gd) 
        : _shape(s), _strides(st), data(d), ctx(c), is_cuda(cuda), gpu_data(gd) {
        _numel = 1;
        for (size_t dim : _shape) _numel *= dim;
        if (_shape.size() == 0) _numel = 0;
    }

    /// Allocates a zero-initialized tensor for the requested shape.
    template <typename T>
    Tensor<T>::Tensor(vc::vector<size_t> shape) : Tensor(shape, true) {}

    /// Allocates a tensor, optionally skipping CPU allocation and zeroing.
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

    /// Allocates an empty tensor directly on the GPU.
    template <typename T>
    Tensor<T> Tensor<T>::empty_gpu(const vc::vector<size_t>& shape) {
        Tensor<T> result(shape, false, false); // Do not allocate CPU memory!
        result.is_cuda = true;
        result.gpu_data = std::make_shared<GPUData<T>>(result.numel());
        return result;
    }
    /// Moves tensor storage to the requested device.
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

    /// Returns a mutable element reference using multidimensional coordinates.
    template <typename T>
    T& Tensor<T>::operator()(const vc::vector<size_t>& coordinates) { 
        size_t idx = 0;
        for (size_t i = 0; i < coordinates.size(); i++) idx += coordinates[i] * _strides[i];

        return (*data)[idx];
    }

    /// Returns a transposed view by reversing the tensor dimensions.
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

    template class Tensor<float>;
} // namespace vc
