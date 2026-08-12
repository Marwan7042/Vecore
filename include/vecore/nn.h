#ifndef NN_H
#define NN_H

#include "Tensor.h"

#include <random>
#include <cmath>
#include <string>


namespace vc {
    namespace nn {
        /// Base interface for trainable or callable neural network layers.
        template <typename T>
        class Layer {
        public:
            /// Releases the layer resources.
            virtual ~Layer() {}
            /// Runs the layer on an input tensor.
            virtual Tensor<T> operator()(const Tensor<T>& x) = 0;
            /// Returns pointers to the layer parameters.
            virtual vc::vector<Tensor<T>*> parameters() { return vc::vector<Tensor<T>*>(); }
            /// Moves the layer's parameters to the requested device.
            virtual void to(const std::string& device) {}
        };

        /// Fully connected layer with an optional activation function.
        template <typename T>
        class Dense : public Layer<T> {
        public:
            Tensor<T> weight;
            Tensor<T> bias;
            std::string activation;

            Dense() {}

            /// Initializes the dense layer with Xavier-initialized weights and zero bias.
            Dense(size_t inputs, size_t neurons, std::string act = "none") : activation(act) {
                vc::vector<size_t> w_shape(2);
                w_shape[0] = neurons; 
                w_shape[1] = inputs;
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

            /// Applies the linear transform and optional activation.
            Tensor<T> operator()(const Tensor<T>& x) {
                Tensor<T> out = x.matmul(weight, false, true);
                if (bias._shape.size() > 0) {
                    out = out + bias;
                }
                if (activation == "relu") {
                    return out.relu();
                } else if (activation == "softmax") {
                    return out.softmax();
                } else if (activation == "sigmoid") {
                    return out.sigmoid();
                } else if (activation == "tanh") {
                    return out.tanh();
                } else if (activation == "leaky_relu") {
                    return out.leaky_relu();
                } else if (activation == "gelu") {
                    return out.gelu();
                } else if (activation == "silu" || activation == "swish") {
                    return out.silu();
                }
                return out;
            }
            
            /// Returns pointers to the weight and bias tensors.
            vc::vector<Tensor<T>*> parameters() override {
                vc::vector<Tensor<T>*> params(2);
                params[0] = &weight;
                params[1] = &bias;
                return params;
            }

            void to(const std::string& device) override {
                weight = weight.to(device);
                bias = bias.to(device);
            }
        };

        /// 2D convolution layer.
        template <typename T>
        class Conv2d : public Layer<T> {
        public:
            Tensor<T> weight;
            Tensor<T> bias;
            int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;

            Conv2d() {}

                        /// Builds a 2D convolution layer with square kernels.
            Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
                : in_channels(in_channels), out_channels(out_channels), kernel_h(kernel_size), kernel_w(kernel_size),
                  stride_h(stride), stride_w(stride), pad_h(padding), pad_w(padding) {
                vc::vector<size_t> w_shape(4);
                w_shape[0] = out_channels; w_shape[1] = in_channels; w_shape[2] = kernel_h; w_shape[3] = kernel_w;
                weight = Tensor<T>(w_shape);
                
                vc::vector<size_t> b_shape(1);
                b_shape[0] = out_channels;
                bias = Tensor<T>(b_shape);
                
                weight.ctx->requires_grad = true;
                bias.ctx->requires_grad = true;

                float limit = std::sqrt(1.0f / (in_channels * kernel_h * kernel_w));
                for (size_t i = 0; i < weight.numel(); i++) (*weight.data)[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
                for (size_t i = 0; i < bias.numel(); i++) (*bias.data)[i] = 0.0f;
            }

            /// Applies convolution followed by autograd bookkeeping.
            Tensor<T> operator()(const Tensor<T>& x) {
                Tensor<T> out = vc::conv2d_forward_gpu(x, weight, bias, stride_h, stride_w, pad_h, pad_w);
                out.ctx->requires_grad = AutogradContext<T>::grad_mode && (x.ctx->requires_grad || weight.ctx->requires_grad || bias.ctx->requires_grad);
                if (out.ctx->requires_grad) {
                    out.ctx->creator = std::make_shared<vc::Conv2dNode<T>>(x, weight, bias, stride_h, stride_w, pad_h, pad_w, out);
                }
                return out;
            }
            
            /// Returns pointers to the convolution weights and bias.
            vc::vector<Tensor<T>*> parameters() override {
                vc::vector<Tensor<T>*> params(2);
                params[0] = &weight; params[1] = &bias;
                return params;
            }

            /// Moves the convolution parameters to the requested device.
            void to(const std::string& device) override {
                weight = weight.to(device);
                bias = bias.to(device);
            }
        };

        /// Max pooling layer.
        template <typename T>
        class MaxPool2d : public Layer<T> {
        public:
            int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;

            MaxPool2d() {}

                        /// Builds a max pooling layer with square kernels.
            MaxPool2d(int kernel_size, int stride = -1, int padding = 0)
                : kernel_h(kernel_size), kernel_w(kernel_size),
                  stride_h(stride == -1 ? kernel_size : stride), stride_w(stride == -1 ? kernel_size : stride),
                  pad_h(padding), pad_w(padding) {}

                        /// Applies max pooling and stores indices for backward propagation.
            Tensor<T> operator()(const Tensor<T>& x) {
                int batch_size = x._shape[0];
                int channels = x._shape[1];
                int height = x._shape[2];
                int width = x._shape[3];
                int pooled_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
                int pooled_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
                
                vc::vector<size_t> out_shape(4);
                out_shape[0] = batch_size; out_shape[1] = channels; out_shape[2] = pooled_height; out_shape[3] = pooled_width;
                Tensor<T> out = Tensor<T>::empty_gpu(out_shape);
                Tensor<T> argmax = Tensor<T>::empty_gpu(out_shape);
                
                vc::maxpool2d_forward_gpu(x, out, argmax, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
                
                out.ctx->requires_grad = AutogradContext<T>::grad_mode && x.ctx->requires_grad;
                if (out.ctx->requires_grad) {
                    out.ctx->creator = std::make_shared<vc::MaxPool2dNode<T>>(x, argmax, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, out);
                }
                return out;
            }
        };
    /// Autograd node for flattening a tensor into 2D.
    template <typename T>
    class FlattenNode : public AutogradNode<T> {
    private:
        Tensor<T> input;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        FlattenNode(Tensor<T> in, Tensor<T> out) : input(in), out_ctx(out.ctx) {}
        
        vc::vector<Tensor<T>> get_parents() override {
            vc::vector<Tensor<T>> p(1); p[0] = input; return p;
        }
        
        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (input.ctx->requires_grad) {
                if (!input.ctx->grad) {
                    Tensor<T> reshaped_grad = out_ctx_shared->grad->reshape(input._shape);
                    input.ctx->grad = std::make_shared<Tensor<T>>(reshaped_grad);
                }
            }
        }
    };

    /// Flattens each sample to a 2D batch x feature matrix.
    template <typename T>
    class Flatten : public Layer<T> {
    public:
        Flatten() {}
        /// Returns a flattened view that participates in autograd.
        Tensor<T> operator()(const Tensor<T>& x) override {
            vc::vector<size_t> new_shape(2);
            new_shape[0] = x._shape[0];
            size_t features = 1;
            for (size_t i = 1; i < x._shape.size(); ++i) features *= x._shape[i];
            new_shape[1] = features;
            
            Tensor<T> out = x.reshape(new_shape);
            out.ctx = std::make_shared<AutogradContext<T>>(); // New context for graph decoupling
            out.ctx->requires_grad = AutogradContext<T>::grad_mode && x.ctx->requires_grad;
            
            if (out.ctx->requires_grad) {
                out.ctx->creator = std::make_shared<FlattenNode<T>>(x, out);
            }
            return out;
        }
    };

    /// Convenience layer that applies ReLU to its input.
    template <typename T>
    class ReLU : public Layer<T> {
    public:
        ReLU() {}
        
        /// Applies ReLU to the input tensor.
        Tensor<T> operator()(const Tensor<T>& x) override {
            return x.relu();
        }
    };

    /// Autograd node for mean squared error loss.
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
                T out_grad_val = out_ctx_shared->grad->item();
                for (size_t i = 0; i < pred.ctx->grad->numel(); i++) {
                    // The derivative of (pred - target)^2 is 2 * (pred - target)
                    (*pred.ctx->grad->data)[i] += 2.0f * ((*pred.data)[i] - (*target.data)[i]) * out_grad_val;
                }
            }
        }
    };

        /// Mean squared error loss.
        template <typename T>
        class MSELoss {
        public:
            /// Computes the average squared error between predictions and targets.
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
    /// Autograd node for categorical cross entropy loss.
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
                T out_grad_val = out_ctx_shared->grad->item();
                for (size_t i = 0; i < logits.ctx->grad->numel(); i++) {
                    (*logits.ctx->grad->data)[i] += ((softmax_probs[i] - (*target.data)[i]) / batch_size) * out_grad_val;
                }
            }
        }
    };

        /// Sparse categorical cross entropy loss configuration helper.
        template <typename T>
        class SparseCategoricalCrossentropy {
        public:
            bool from_logits;
            /// Stores whether the provided inputs are logits.
            SparseCategoricalCrossentropy(bool from_logits = false) : from_logits(from_logits) {}
        };

        /// Categorical cross entropy loss.
        template <typename T>
        class CrossEntropyLoss {
        public:
            /// Computes softmax cross entropy and attaches the backward node.
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

namespace tf {
    namespace nn {
        /// Convenience softmax wrapper in the tf::nn namespace.
        template <typename T>
        vc::Tensor<T> softmax(const vc::Tensor<T>& logits) {
            return logits.softmax();
        }
    }
}

#endif // NN_H
