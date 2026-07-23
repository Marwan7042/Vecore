#ifndef OPTIM_H
#define OPTIM_H

#include "Tensor.h"

namespace vc {
    namespace optim {
        template <typename T>
        class SGD {
        private:
            vc::vector<Tensor<T>*> parameters;
            float lr;
        public:
            SGD(vc::vector<Tensor<T>*> params, float learning_rate) 
                : parameters(params), lr(learning_rate) {}

            // Resets all the gradients to 0 so they don't accumulate forever
            void zero_grad() {
                for (size_t p = 0; p < parameters.size(); p++) {
                    parameters[p]->zero_grad_data();
                }
            }

            // Steps down the slope: New Weight = Old Weight - (lr * gradient)
            void step() {
                for (size_t p = 0; p < parameters.size(); p++) {
                    parameters[p]->sgd_update(lr);
                }
            }
        };
    }
}
#endif
