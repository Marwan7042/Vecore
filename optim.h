#ifndef OPTIM_H
#define OPTIM_H

#include "Tensor.h"

namespace mstd {
    namespace optim {
        template <typename T>
        class SGD {
        private:
            mstd::vector<Tensor<T>*> parameters;
            float lr;
        public:
            SGD(mstd::vector<Tensor<T>*> params, float learning_rate) 
                : parameters(params), lr(learning_rate) {}

            // Resets all the gradients to 0 so they don't accumulate forever
            void zero_grad() {
                for (size_t p = 0; p < parameters.size(); p++) {
                    Tensor<T>* param = parameters[p];
                    if (param->ctx->grad) {
                        for (size_t j = 0; j < param->ctx->grad->data->size(); j++) {
                            (*param->ctx->grad->data)[j] = 0.0f;
                        }
                    }
                }
            }

            // Steps down the slope: New Weight = Old Weight - (lr * gradient)
            void step() {
                for (size_t p = 0; p < parameters.size(); p++) {
                    Tensor<T>* param = parameters[p];
                    if (param->ctx->grad) {
                        for (size_t j = 0; j < param->data->size(); j++) {
                            (*param->data)[j] -= lr * (*param->ctx->grad->data)[j];
                        }
                        // If the tensor is on the GPU, we must push the newly updated weights to VRAM!
                        if (param->is_cuda && param->gpu_data) {
                            cudaMemcpy(param->gpu_data->ptr, param->data->begin(), param->data->size() * sizeof(T), cudaMemcpyHostToDevice);
                        }
                    }
                }
            }
        };
    }
}
#endif
