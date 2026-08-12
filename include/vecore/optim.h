#ifndef OPTIM_H
#define OPTIM_H

#include "Tensor.h"

namespace vc {
    namespace optim {
        /// Stochastic gradient descent optimizer.
        template <typename T>
        class SGD {
        private:
            vc::vector<Tensor<T>*> parameters;
            float lr;
        public:
            /// Builds an SGD optimizer over the provided parameters.
            SGD(vc::vector<Tensor<T>*> params, float learning_rate) 
                : parameters(params), lr(learning_rate) {}

            /// Updates the learning rate used by the optimizer.
            void set_lr(float new_lr) {
                lr = new_lr;
            }

            /// Zeros all parameter gradients.
            void zero_grad() {
                for (size_t p = 0; p < parameters.size(); p++) {
                    parameters[p]->zero_grad_data();
                }
            }

            /// Applies one SGD update step.
            void step() {
                for (size_t p = 0; p < parameters.size(); p++) {
                    parameters[p]->sgd_update(lr);
                }
            }
        };

        /// Adam optimizer.
        template <typename T>
        class Adam {
        private:
            vc::vector<Tensor<T>*> parameters;
            vc::vector<Tensor<T>> m_slots;
            vc::vector<Tensor<T>> v_slots;
            float lr, beta1, beta2, eps;
            int t;
        public:
            /// Builds an Adam optimizer with standard default hyperparameters.
            Adam(vc::vector<Tensor<T>*> params, float learning_rate = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float e = 1e-8f) 
                : parameters(params), m_slots(params.size()), v_slots(params.size()), lr(learning_rate), beta1(b1), beta2(b2), eps(e), t(0) {
                
                for (size_t i = 0; i < parameters.size(); i++) {
                    Tensor<T> cpu_t(parameters[i]->_shape);
                    for (size_t j = 0; j < cpu_t.numel(); j++) (*cpu_t.data)[j] = 0.0f;
                    
                    if (parameters[i]->is_cuda) {
                        m_slots[i] = cpu_t.to("cuda");
                        v_slots[i] = cpu_t.to("cuda");
                    } else {
                        m_slots[i] = cpu_t;
                        v_slots[i] = cpu_t;
                    }
                }
            }

            /// Updates the learning rate used by the optimizer.
            void set_lr(float new_lr) { lr = new_lr; }

            /// Zeros all parameter gradients.
            void zero_grad() {
                for (size_t p = 0; p < parameters.size(); p++) {
                    parameters[p]->zero_grad_data();
                }
            }

            /// Applies one Adam update step.
            void step() {
                t++;
                for (size_t p = 0; p < parameters.size(); p++) {
                    parameters[p]->adam_update(m_slots[p], v_slots[p], lr, beta1, beta2, eps, t);
                }
            }
        };
    }
}
#endif
