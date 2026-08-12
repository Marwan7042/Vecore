#include "vecore/Tensor.h"
#include "vecore/nn.h"
#include "../TensorKernels.cuh"

namespace vc {
    /// Builds a reverse topological ordering of the autograd graph.
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

/// Starts reverse-mode automatic differentiation from this tensor.
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

    /// Fast fused backward pass for cross-entropy loss and accuracy metrics.
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

    /// Applies the sigmoid activation function.
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

    /// Backpropagates through sigmoid into the provided accumulation tensor.
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

    /// Applies the tanh activation function.
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

    /// Backpropagates through tanh into the provided accumulation tensor.
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

    /// Applies leaky ReLU with the given negative slope.
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

    /// Backpropagates through leaky ReLU into the provided accumulation tensor.
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

    /// Applies the GELU activation function.
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

    /// Backpropagates through GELU into the provided accumulation tensor.
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

    /// Applies the SiLU / Swish activation function.
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

    /// Backpropagates through SiLU / Swish into the provided accumulation tensor.
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

    template class Tensor<float>;
} // namespace vc
