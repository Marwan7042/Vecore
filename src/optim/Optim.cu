#include "vecore/Tensor.h"
#include "vecore/optim.h"

#ifdef __CUDACC__
template <typename T>
__global__ void cuda_adam_update_kernel(T* weight, const T* grad, T* m, T* v, float lr, float beta1, float beta2, float eps, int t, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];
        float m_hat = m[idx] / (1.0f - powf(beta1, (float)t));
        float v_hat = v[idx] / (1.0f - powf(beta2, (float)t));
        weight[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}
#endif

namespace vc {
/// Applies Adam parameter updates using the running first and second moments.
template <typename T>
void Tensor<T>::adam_update(Tensor<T>& m, Tensor<T>& v, float lr, float beta1, float beta2, float eps, int t) {
#ifdef __CUDACC__
    if (this->is_cuda) {
        int threads = 256;
        int blocks = (this->numel() + threads - 1) / threads;
        cuda_adam_update_kernel<<<blocks, threads>>>(this->gpu_data->ptr, this->ctx->grad->gpu_data->ptr, m.gpu_data->ptr, v.gpu_data->ptr, lr, beta1, beta2, eps, t, this->numel());
    } else 
#endif
    {
        for (size_t i = 0; i < this->numel(); i++) {
            (*m.data)[i] = beta1 * (*m.data)[i] + (1.0f - beta1) * (*this->ctx->grad->data)[i];
            (*v.data)[i] = beta2 * (*v.data)[i] + (1.0f - beta2) * (*this->ctx->grad->data)[i] * (*this->ctx->grad->data)[i];
            float m_hat = (*m.data)[i] / (1.0f - std::pow(beta1, t));
            float v_hat = (*v.data)[i] / (1.0f - std::pow(beta2, t));
            (*this->data)[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}
}

namespace vc {
    template class Tensor<float>;
}
