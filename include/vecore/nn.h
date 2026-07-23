#ifndef NN_H
#define NN_H

#include "Tensor.h"
#include <random>
#include <cmath>
#include <string>


namespace vc {
    namespace nn {
        template <typename T>
        class Dense {
        public:
            Tensor<T> weight;
            Tensor<T> bias;
            std::string activation;

            Dense() {}

            // Initialize the Dense layer with random weights and optional activation
            Dense(size_t inputs, size_t neurons, std::string act = "none") : activation(act) {
                vc::vector<size_t> w_shape(2);
                w_shape[0] = inputs; 
                w_shape[1] = neurons;
                weight = Tensor<T>(w_shape);

                vc::vector<size_t> b_shape(2);
                b_shape[0] = 1; 
                b_shape[1] = neurons;
                bias = Tensor<T>(b_shape);
                
                weight.ctx->requires_grad = true;
                bias.ctx->requires_grad = true;

                // Mathematical Xavier Initialization for fast learning
                std::random_device rd;
                std::mt19937 gen(rd());
                float limit = std::sqrt(6.0f / (inputs + neurons));
                std::uniform_real_distribution<float> dis(-limit, limit);

                for (size_t i = 0; i < weight.numel(); i++) {
                    (*weight.data)[i] = dis(gen);
                }
                for (size_t i = 0; i < bias.numel(); i++) {
                    (*bias.data)[i] = 0.0f; // biases start at 0
                }
            }

            // The Forward Pass for this layer!
            Tensor<T> operator()(const Tensor<T>& x) {
                Tensor<T> out = (x * weight) + bias;
                if (activation == "relu") {
                    return out.relu();
                }
                return out;
            }
            
            // Helper to get all parameters so our Optimizer can update them
            vc::vector<Tensor<T>*> parameters() {
                vc::vector<Tensor<T>*> params(2);
                params[0] = &weight;
                params[1] = &bias;
                return params;
            }

            Dense<T> to(const std::string& device) {
                weight = weight.to(device);
                bias = bias.to(device);
                return *this;
            }
        };



    template <typename T>
    class MSELossNode : public AutogradNode<T> {
    private:
        Tensor<T> pred, target;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        MSELossNode(Tensor<T> p, Tensor<T> t, Tensor<T> o) : pred(p), target(t), out_ctx(o.ctx) {}
        
        vc::vector<Tensor<T>> get_parents() override {
            vc::vector<Tensor<T>> parents(2);
            parents[0] = pred; parents[1] = target;
            return parents;
        }

        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (pred.ctx->requires_grad) {
                if (!pred.ctx->grad) {
                    Tensor<T> zero_grad(pred._shape);
                    pred.ctx->grad = std::make_shared<Tensor<T>>(pred.is_cuda ? zero_grad.to("cuda") : zero_grad);
                }
                for (size_t i = 0; i < pred.ctx->grad->numel(); i++) {
                    // The derivative of (pred - target)^2 is 2 * (pred - target)
                    (*pred.ctx->grad->data)[i] += 2.0f * ((*pred.data)[i] - (*target.data)[i]) * (*out_ctx_shared->grad->data)[0];
                }
            }
        }
    };

        template <typename T>
        class MSELoss {
        public:
            Tensor<T> operator()(const Tensor<T>& predictions, const Tensor<T>& targets) {
                vc::vector<size_t> out_shape(2); out_shape[0] = 1; out_shape[1] = 1;
                Tensor<T> result(out_shape);
                
                float sum = 0.0f;
                for (size_t i = 0; i < predictions.numel(); i++) {
                    float err = (*predictions.data)[i] - (*targets.data)[i];
                    sum += err * err;
                }
                (*result.data)[0] = sum / predictions.numel(); // Mean Squared Error

                result.ctx->requires_grad = predictions.ctx->requires_grad;
                if (result.ctx->requires_grad) {
                    result.ctx->creator = std::make_shared<MSELossNode<T>>(predictions, targets, result);
                }
                return result;
            }
        };

    template <typename T>
    class CrossEntropyLossNode : public AutogradNode<T> {
    private:
        Tensor<T> logits, target;
        std::weak_ptr<AutogradContext<T>> out_ctx;
        vc::vector<T> softmax_probs;
    public:
        CrossEntropyLossNode(Tensor<T> l, Tensor<T> t, Tensor<T> o, vc::vector<T> probs) 
            : logits(l), target(t), out_ctx(o.ctx), softmax_probs(probs) {}
        
        vc::vector<Tensor<T>> get_parents() override {
            vc::vector<Tensor<T>> parents(2);
            parents[0] = logits; parents[1] = target;
            return parents;
        }

        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (logits.ctx->requires_grad) {
                if (!logits.ctx->grad) {
                    Tensor<T> zero_grad(logits._shape);
                    logits.ctx->grad = std::make_shared<Tensor<T>>(logits.is_cuda ? zero_grad.to("cuda") : zero_grad);
                }
                
                size_t batch_size = logits._shape.size() > 1 ? logits._shape[0] : 1;
                for (size_t i = 0; i < logits.ctx->grad->numel(); i++) {
                    (*logits.ctx->grad->data)[i] += ((softmax_probs[i] - (*target.data)[i]) / batch_size) * (*out_ctx_shared->grad->data)[0];
                }
            }
        }
    };

        template <typename T>
        class CrossEntropyLoss {
        public:
            Tensor<T> operator()(const Tensor<T>& logits, const Tensor<T>& targets) {
                size_t batch_size = logits._shape.size() > 1 ? logits._shape[0] : 1;
                size_t num_classes = logits._shape.size() > 1 ? logits._shape[1] : logits._shape[0];
                
                vc::vector<size_t> out_shape(2); out_shape[0] = 1; out_shape[1] = 1;
                Tensor<T> result(out_shape);
                
                vc::vector<T> probs(logits.numel());
                T total_loss = 0.0f;
                
                for (size_t b = 0; b < batch_size; b++) {
                    T max_logit = (*logits.data)[b * num_classes];
                    for (size_t j = 1; j < num_classes; j++) {
                        if ((*logits.data)[b * num_classes + j] > max_logit) 
                            max_logit = (*logits.data)[b * num_classes + j];
                    }
                    
                    T sum_exp = 0.0f;
                    for (size_t j = 0; j < num_classes; j++) {
                        probs[b * num_classes + j] = std::exp((*logits.data)[b * num_classes + j] - max_logit);
                        sum_exp += probs[b * num_classes + j];
                    }
                    
                    for (size_t j = 0; j < num_classes; j++) {
                        probs[b * num_classes + j] /= sum_exp;
                        if ((*targets.data)[b * num_classes + j] > 0.5f) {
                            total_loss -= std::log(probs[b * num_classes + j] + 1e-7f);
                        }
                    }
                }
                
                (*result.data)[0] = total_loss / batch_size;

                result.ctx->requires_grad = logits.ctx->requires_grad;
                if (result.ctx->requires_grad) 
                    result.ctx->creator = std::make_shared<CrossEntropyLossNode<T>>(logits, targets, result, probs);

                return result;
            }
        };
    }
}

#endif
