#ifndef Tensor_H
#define Tensor_H

#include "vector.h"
#include "unordered_map.h"
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace vc{
    template <typename T> class AutogradNode; 
    template <typename T> struct GPUData;
    template <typename T> struct AutogradContext;
    template <typename T> class Tensor;

    /// Core n-dimensional tensor type with CPU/GPU storage and autograd metadata.
    template <typename T>
    class Tensor {
    public:
        vc::vector<size_t> _shape;
        vc::vector<size_t> _strides;
        size_t _numel;
        std::shared_ptr<vc::vector<T>> data;
        std::shared_ptr<AutogradContext<T>> ctx;
        
        bool is_cuda = false;
        std::shared_ptr<GPUData<T>> gpu_data = nullptr;

        /// Creates an empty tensor with a fresh autograd context.
        Tensor();

        /// Builds a tensor from existing shape, strides, storage, context, and optional GPU state.
        Tensor(vc::vector<size_t> s, vc::vector<size_t> st, std::shared_ptr<vc::vector<T>> d, std::shared_ptr<AutogradContext<T>> c, bool cuda=false, std::shared_ptr<GPUData<T>> gd=nullptr);

        /// Allocates a zero-initialized tensor for the given shape.
        Tensor(vc::vector<size_t> shape);
        /// Allocates a tensor for the given shape, optionally skipping zeroing or CPU storage.
        Tensor(vc::vector<size_t> shape, bool zero_memory, bool allocate_cpu = true);

        /// Returns a copy of this tensor on the requested device.
        Tensor<T> to(const std::string& device) const;

        /// Returns a mutable reference to the element at the given multidimensional index.
        T& operator()(const vc::vector<size_t>& dims);
        
        /// Returns a view with dimensions reversed.
        Tensor<T> transpose() const;    

        /// Adds two tensors element-wise and records the operation for autograd.
        Tensor<T> operator+(const Tensor<T>& other) const;
        
        /// Adds another tensor in place without recording a graph edge.
        Tensor<T>& operator+=(const Tensor<T>& other);

        /// Subtracts another tensor element-wise and records the operation for autograd.
        Tensor<T> operator-(const Tensor<T>& other) const;
        
        /// Subtracts another tensor in place without recording a graph edge.
        Tensor<T>& operator-=(const Tensor<T>& other);

        /// Multiplies tensors using matrix multiplication semantics.
        Tensor<T> operator*(const Tensor<T>& other) const { return this->matmul(other); }

        /// Performs matrix multiplication with optional transposition flags.
        Tensor<T> matmul(const Tensor<T>& other, bool transA = false, bool transB = false) const;
        /// Accumulates a matrix product directly into an existing output tensor.
        void matmul_accumulate(const Tensor<T>& other, Tensor<T>& accum, bool transA = false, bool transB = false) const;
        /// Sums the specified dimensions, optionally keeping the reduced axes.
        Tensor<T> sum(const vc::vector<int>& dims, bool keepdim = false) const;
        /// Applies ReLU element-wise.
        Tensor<T> relu() const;
        /// Backpropagates through ReLU into an accumulation tensor.
        void relu_backward(const Tensor<T>& out_grad, Tensor<T>& accum) const;
        /// Applies a numerically stable softmax along the last dimension.
        Tensor<T> softmax(int dim = -1) const;
        /// Applies the logistic sigmoid element-wise.
        Tensor<T> sigmoid() const;
        /// Backpropagates through sigmoid into an accumulation tensor.
        void sigmoid_backward(const Tensor<T>& out_grad, Tensor<T>& accum) const;
        /// Applies tanh element-wise.
        Tensor<T> tanh() const;
        /// Backpropagates through tanh into an accumulation tensor.
        void tanh_backward(const Tensor<T>& out_grad, Tensor<T>& accum) const;
        /// Applies leaky ReLU element-wise.
        Tensor<T> leaky_relu(float alpha = 0.01f) const;
        /// Backpropagates through leaky ReLU into an accumulation tensor.
        void leaky_relu_backward(const Tensor<T>& out_grad, Tensor<T>& accum, float alpha = 0.01f) const;
        /// Applies GELU element-wise.
        Tensor<T> gelu() const;
        /// Backpropagates through GELU into an accumulation tensor.
        void gelu_backward(const Tensor<T>& out_grad, Tensor<T>& accum) const;
        /// Applies SiLU / Swish element-wise.
        Tensor<T> silu() const;
        /// Backpropagates through SiLU / Swish into an accumulation tensor.
        void silu_backward(const Tensor<T>& out_grad, Tensor<T>& accum) const;
        
        /// Returns the number of elements in the tensor.
        size_t numel() const { return _numel; }

        /// Allocates an empty tensor directly on the GPU.
        static Tensor<T> empty_gpu(const vc::vector<size_t>& shape);
        
        /// Fast fused backward path for cross-entropy training.
        void fast_cross_entropy_backward(const Tensor<T>& target, float* d_loss, int* d_correct);
        
        /// Applies a vanilla SGD update using the stored gradient.
        void sgd_update(float lr);
        /// Applies an Adam update using the provided moment buffers.
        void adam_update(Tensor<T>& m, Tensor<T>& v, float lr, float beta1, float beta2, float eps, int t);
        /// Zeros the gradient tensor, if one exists.
        void zero_grad();
        /// Zeros only the gradient storage while preserving the tensor data.
        void zero_grad_data();
        /// Fills the tensor storage with zeros.
        Tensor<T>& zero_();

        /// Returns a reshaped view with the same underlying storage.
        Tensor<T> reshape(const vc::vector<size_t>& new_shape) const;

        /// Starts reverse-mode automatic differentiation from this tensor.
        void backward();

        /// Prints the tensor metadata and values to standard output.
        void print() const;
        
        /// Returns the scalar value of a one-element tensor.
        T item() const {
            if (numel() != 1) throw std::runtime_error("item() only valid for scalar tensors");
            return this->is_cuda ? (*this->to("cpu").data)[0] : (*this->data)[0];
        }
    };

    template <typename T>
    struct CachingAllocator {
        static thread_local std::unordered_map<size_t, std::vector<T*>> free_blocks;
        static T* allocate(size_t size) {
            auto& blocks = free_blocks[size];
            if (!blocks.empty()) {
                T* ptr = blocks.back();
                blocks.pop_back();
                return ptr;
            }
            T* ptr;
            cudaMalloc((void**)&ptr, size * sizeof(T));
            return ptr;
        }
        static void free(T* ptr, size_t size) {
            free_blocks[size].push_back(ptr);
        }
    };

    template <typename T> 
    inline thread_local std::unordered_map<size_t, std::vector<T*>> CachingAllocator<T>::free_blocks;

    template <typename T>
    struct GPUData {
        T* ptr = nullptr;
        size_t size = 0;
        bool owns_memory = true;
        GPUData(size_t s) : size(s) {
            ptr = CachingAllocator<T>::allocate(size);
        }
        GPUData(T* p, size_t s, bool owns) : ptr(p), size(s), owns_memory(owns) {}
        ~GPUData() noexcept {
            if (ptr && owns_memory) {
                try {
                    CachingAllocator<T>::free(ptr, size);
                } catch(...) {}
            }
        }
    };

    template <typename T>
    struct AutogradContext {
        static bool grad_mode;
        bool requires_grad = false;
        std::shared_ptr<Tensor<T>> grad = nullptr;
        std::shared_ptr<AutogradNode<T>> creator = nullptr;
    };

    // Base class for all operations in the computation graph. 
    // Defines the contract that every mathematical operation must fulfill to support backpropagation.
    template <typename T>
    class AutogradNode {
    public:
        virtual void backward() = 0;
        virtual vc::vector<Tensor<T>> get_parents() = 0;
        virtual ~AutogradNode() = default;
    };
    
    // Graph node representing element-wise addition.
    // Propagates the upstream gradient equally to both input tensors.
    template <typename T>
    class AddNode : public AutogradNode<T> {
    private:
        Tensor<T> a, b;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        AddNode(Tensor<T> a, Tensor<T> b, Tensor<T> out) : a(a), b(b), out_ctx(out.ctx) {}

        vc::vector<Tensor<T>> get_parents() override {
            vc::vector<Tensor<T>> parents(2);
            parents[0] = a;
            parents[1] = b;
            return parents;
        }

        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            auto reduce_grad = [](const Tensor<T>& input, const Tensor<T>& out_grad) {
                if (input._shape == out_grad._shape) return out_grad;
                
                vc::vector<int> dims_to_sum;
                int rank_diff = (int)out_grad._shape.size() - (int)input._shape.size();
                for (int i = 0; i < rank_diff; i++) dims_to_sum.push_back(i);
                
                for (size_t i = 0; i < input._shape.size(); i++) {
                    if (input._shape[i] == 1 && out_grad._shape[i + rank_diff] > 1) {
                        dims_to_sum.push_back(i + rank_diff);
                    }
                }
                
                if (dims_to_sum.size() > 0) {
                    return out_grad.sum(dims_to_sum, true).reshape(input._shape);
                }
                return out_grad;
            };

            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) {
                    if (a.is_cuda) {
                        Tensor<T> grad_tensor = Tensor<T>::empty_gpu(a._shape);
#ifdef __CUDACC__
                        cudaMemset(grad_tensor.gpu_data->ptr, 0, grad_tensor.numel() * sizeof(T));
#endif
                        a.ctx->grad = std::make_shared<Tensor<T>>(grad_tensor);
                    } else {
                        Tensor<T> zero_grad(a._shape);
                        a.ctx->grad = std::make_shared<Tensor<T>>(zero_grad);
                    }
                }
                Tensor<T> reduced = reduce_grad(a, *out_ctx_shared->grad);
                *(a.ctx->grad) += reduced;
            }
            if (b.ctx->requires_grad) {
                Tensor<T> reduced = reduce_grad(b, *out_ctx_shared->grad);
                if (!b.ctx->grad) {
                    b.ctx->grad = std::make_shared<Tensor<T>>(reduced);
                } else {
                    *(b.ctx->grad) += reduced;
                }
            }
        }
    };

    // Graph node representing element-wise subtraction.
    // Propagates the upstream gradient positively to the left operand and negatively to the right operand.
    template <typename T>
    class SubNode : public AutogradNode<T> {
    private:
        Tensor<T> a, b;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        SubNode(Tensor<T> a, Tensor<T> b, Tensor<T> out) : a(a), b(b), out_ctx(out.ctx) {}

        vc::vector<Tensor<T>> get_parents() override {
            vc::vector<Tensor<T>> parents(2);
            parents[0] = a;
            parents[1] = b;
            return parents;
        }

        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            auto reduce_grad = [](const Tensor<T>& input, const Tensor<T>& out_grad) {
                if (input._shape == out_grad._shape) return out_grad;
                
                vc::vector<int> dims_to_sum;
                int rank_diff = (int)out_grad._shape.size() - (int)input._shape.size();
                for (int i = 0; i < rank_diff; i++) dims_to_sum.push_back(i);
                
                for (size_t i = 0; i < input._shape.size(); i++) {
                    if (input._shape[i] == 1 && out_grad._shape[i + rank_diff] > 1) {
                        dims_to_sum.push_back(i + rank_diff);
                    }
                }
                
                if (dims_to_sum.size() > 0) {
                    return out_grad.sum(dims_to_sum, true).reshape(input._shape);
                }
                return out_grad;
            };

            if (a.ctx->requires_grad) {
                Tensor<T> reduced = reduce_grad(a, *out_ctx_shared->grad);
                if (!a.ctx->grad) {
                    a.ctx->grad = std::make_shared<Tensor<T>>(reduced);
                } else {
                    *(a.ctx->grad) += reduced;
                }
            }
            if (b.ctx->requires_grad) {
                if (!b.ctx->grad) {
                    if (b.is_cuda) {
                        Tensor<T> grad_tensor = Tensor<T>::empty_gpu(b._shape);
#ifdef __CUDACC__
                        cudaMemset(grad_tensor.gpu_data->ptr, 0, grad_tensor.numel() * sizeof(T));
#endif
                        b.ctx->grad = std::make_shared<Tensor<T>>(grad_tensor);
                    } else {
                        Tensor<T> zero_grad(b._shape);
                        b.ctx->grad = std::make_shared<Tensor<T>>(zero_grad);
                    }
                }
                Tensor<T> reduced = reduce_grad(b, *out_ctx_shared->grad);
                // Subtraction flips the gradient sign for the right operand
                *(b.ctx->grad) -= reduced;
            }
        }
    };

    // Graph node representing matrix multiplication.
    // Uses the matrix calculus chain rule (A_grad = out_grad * B^T, B_grad = A^T * out_grad) to propagate gradients.
    template <typename T>
    class MulNode : public AutogradNode<T> {
    private:
        Tensor<T> a, b;
        bool transA, transB;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        MulNode(Tensor<T> a, Tensor<T> b, Tensor<T> out, bool transA = false, bool transB = false) 
            : a(a), b(b), transA(transA), transB(transB), out_ctx(out.ctx) {}

        vc::vector<Tensor<T>> get_parents() override {
            vc::vector<Tensor<T>> parents(2);
            parents[0] = a;
            parents[1] = b;
            return parents;
        }

        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) {
                    a.ctx->grad = std::make_shared<Tensor<T>>(a.is_cuda ? Tensor<T>::empty_gpu(a._shape).zero_() : Tensor<T>(a._shape, true));
                }
                // C = opA(A) * opB(B)
                // A_grad = C_grad * opB(B)^T
                // If opA is T, we need A_grad^T, so we actually want to accumulate into A_grad
                if (!transA) {
                    (*out_ctx_shared->grad).matmul_accumulate(b, *(a.ctx->grad), false, !transB);
                } else {
                    b.matmul_accumulate(*(out_ctx_shared->grad), *(a.ctx->grad), transB, true);
                }
            }

            if (b.ctx->requires_grad) {
                if (!b.ctx->grad) {
                    b.ctx->grad = std::make_shared<Tensor<T>>(b.is_cuda ? Tensor<T>::empty_gpu(b._shape).zero_() : Tensor<T>(b._shape, true));
                }
                // B_grad = opA(A)^T * C_grad
                if (!transB) {
                    a.matmul_accumulate(*(out_ctx_shared->grad), *(b.ctx->grad), !transA, false);
                } else {
                    (*out_ctx_shared->grad).matmul_accumulate(a, *(b.ctx->grad), true, transA);
                }
            }
        }
    };

    // Graph node representing the Rectified Linear Unit (ReLU) activation.
    // Propagates gradients only if the original forward pass value was greater than zero.
    template <typename T>
    class ReLU : public AutogradNode<T> {
    private:
        Tensor<T> a;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        ReLU(Tensor<T> a, Tensor<T> out) : a(a), out_ctx(out.ctx) {}

        vc::vector<Tensor<T>> get_parents() override {
            vc::vector<Tensor<T>> parents(1);
            parents[0] = a;
            return parents;
        }

        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) {
                    if (a.is_cuda) {
                        Tensor<T> grad_tensor = Tensor<T>::empty_gpu(a._shape);
#ifdef __CUDACC__
                        cudaMemset(grad_tensor.gpu_data->ptr, 0, grad_tensor.numel() * sizeof(T));
#endif
                        a.ctx->grad = std::make_shared<Tensor<T>>(grad_tensor);
                    } else {
                        Tensor<T> zero_grad(a._shape);
                        a.ctx->grad = std::make_shared<Tensor<T>>(zero_grad);
                    }
                }
                a.relu_backward(*(out_ctx_shared->grad), *(a.ctx->grad));
            }
        }
    };

    /// Autograd node for sigmoid.
    template <typename T>
    class SigmoidNode : public AutogradNode<T> {
    private:
        Tensor<T> a;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        SigmoidNode(Tensor<T> a, Tensor<T> out) : a(a), out_ctx(out.ctx) {}
        vc::vector<Tensor<T>> get_parents() override { vc::vector<Tensor<T>> parents(1); parents[0] = a; return parents; }
        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) {
                    if (a.is_cuda) {
                        Tensor<T> grad_tensor = Tensor<T>::empty_gpu(a._shape);
#ifdef __CUDACC__
                        cudaMemset(grad_tensor.gpu_data->ptr, 0, grad_tensor.numel() * sizeof(T));
#endif
                        a.ctx->grad = std::make_shared<Tensor<T>>(grad_tensor);
                    } else { a.ctx->grad = std::make_shared<Tensor<T>>(Tensor<T>(a._shape)); }
                }
                a.sigmoid_backward(*(out_ctx_shared->grad), *(a.ctx->grad));
            }
        }
    };

    /// Autograd node for tanh.
    template <typename T>
    class TanhNode : public AutogradNode<T> {
    private:
        Tensor<T> a;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        TanhNode(Tensor<T> a, Tensor<T> out) : a(a), out_ctx(out.ctx) {}
        vc::vector<Tensor<T>> get_parents() override { vc::vector<Tensor<T>> parents(1); parents[0] = a; return parents; }
        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) {
                    if (a.is_cuda) {
                        Tensor<T> grad_tensor = Tensor<T>::empty_gpu(a._shape);
#ifdef __CUDACC__
                        cudaMemset(grad_tensor.gpu_data->ptr, 0, grad_tensor.numel() * sizeof(T));
#endif
                        a.ctx->grad = std::make_shared<Tensor<T>>(grad_tensor);
                    } else { a.ctx->grad = std::make_shared<Tensor<T>>(Tensor<T>(a._shape)); }
                }
                a.tanh_backward(*(out_ctx_shared->grad), *(a.ctx->grad));
            }
        }
    };

    /// Autograd node for leaky ReLU.
    template <typename T>
    class LeakyReLUNode : public AutogradNode<T> {
    private:
        Tensor<T> a;
        float alpha;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        LeakyReLUNode(Tensor<T> a, Tensor<T> out, float alpha = 0.01f) : a(a), alpha(alpha), out_ctx(out.ctx) {}
        vc::vector<Tensor<T>> get_parents() override { vc::vector<Tensor<T>> parents(1); parents[0] = a; return parents; }
        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) {
                    if (a.is_cuda) {
                        Tensor<T> grad_tensor = Tensor<T>::empty_gpu(a._shape);
#ifdef __CUDACC__
                        cudaMemset(grad_tensor.gpu_data->ptr, 0, grad_tensor.numel() * sizeof(T));
#endif
                        a.ctx->grad = std::make_shared<Tensor<T>>(grad_tensor);
                    } else { a.ctx->grad = std::make_shared<Tensor<T>>(Tensor<T>(a._shape)); }
                }
                a.leaky_relu_backward(*(out_ctx_shared->grad), *(a.ctx->grad), alpha);
            }
        }
    };

    /// Autograd node for GELU.
    template <typename T>
    class GELUNode : public AutogradNode<T> {
    private:
        Tensor<T> a;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        GELUNode(Tensor<T> a, Tensor<T> out) : a(a), out_ctx(out.ctx) {}
        vc::vector<Tensor<T>> get_parents() override { vc::vector<Tensor<T>> parents(1); parents[0] = a; return parents; }
        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) {
                    if (a.is_cuda) {
                        Tensor<T> grad_tensor = Tensor<T>::empty_gpu(a._shape);
#ifdef __CUDACC__
                        cudaMemset(grad_tensor.gpu_data->ptr, 0, grad_tensor.numel() * sizeof(T));
#endif
                        a.ctx->grad = std::make_shared<Tensor<T>>(grad_tensor);
                    } else { a.ctx->grad = std::make_shared<Tensor<T>>(Tensor<T>(a._shape)); }
                }
                a.gelu_backward(*(out_ctx_shared->grad), *(a.ctx->grad));
            }
        }
    };

    /// Autograd node for SiLU / Swish.
    template <typename T>
    class SiLUNode : public AutogradNode<T> {
    private:
        Tensor<T> a;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        SiLUNode(Tensor<T> a, Tensor<T> out) : a(a), out_ctx(out.ctx) {}
        vc::vector<Tensor<T>> get_parents() override { vc::vector<Tensor<T>> parents(1); parents[0] = a; return parents; }
        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) {
                    if (a.is_cuda) {
                        Tensor<T> grad_tensor = Tensor<T>::empty_gpu(a._shape);
#ifdef __CUDACC__
                        cudaMemset(grad_tensor.gpu_data->ptr, 0, grad_tensor.numel() * sizeof(T));
#endif
                        a.ctx->grad = std::make_shared<Tensor<T>>(grad_tensor);
                    } else { a.ctx->grad = std::make_shared<Tensor<T>>(Tensor<T>(a._shape)); }
                }
                a.silu_backward(*(out_ctx_shared->grad), *(a.ctx->grad));
            }
        }
    };

    /// Autograd node for device transfers.
    template <typename T>
    class ToNode : public AutogradNode<T> {
    private:
        Tensor<T> a;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        ToNode(Tensor<T> a, Tensor<T> out) : a(a), out_ctx(out.ctx) {}

        vc::vector<Tensor<T>> get_parents() override {
            vc::vector<Tensor<T>> parents(1);
            parents[0] = a;
            return parents;
        }

        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) {
                    if (a.is_cuda) {
                        Tensor<T> grad_tensor = Tensor<T>::empty_gpu(a._shape);
#ifdef __CUDACC__
                        cudaMemset(grad_tensor.gpu_data->ptr, 0, grad_tensor.numel() * sizeof(T));
#endif
                        a.ctx->grad = std::make_shared<Tensor<T>>(grad_tensor);
                    } else {
                        Tensor<T> zero_grad(a._shape);
                        a.ctx->grad = std::make_shared<Tensor<T>>(zero_grad);
                    }
                }
                // Push the gradient back to the source device!
                Tensor<T> grad_moved = out_ctx_shared->grad->to(a.is_cuda ? "cuda" : "cpu");
                *(a.ctx->grad) += grad_moved;
            }
        }
    };

    /// GPU helper that extracts image patches into column form.
    template <typename T>
    Tensor<T> im2col_gpu(const Tensor<T>& im, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
    /// GPU helper that folds column data back into an image tensor.
    template <typename T>
    void col2im_gpu(const Tensor<T>& col, Tensor<T>& im, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
    /// Forward pass for max pooling on the GPU.
    template <typename T>
    void maxpool2d_forward_gpu(const Tensor<T>& bottom, Tensor<T>& top, Tensor<T>& argmax, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
    /// Backward pass for max pooling on the GPU.
    template <typename T>
    void maxpool2d_backward_gpu(const Tensor<T>& top_diff, const Tensor<T>& argmax, Tensor<T>& bottom_diff, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
    
    template <typename T>
    Tensor<T> im2col_gpu(const Tensor<T>& im, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
    template <typename T>
    void col2im_gpu(const Tensor<T>& col, Tensor<T>& im, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
    template <typename T>
    void maxpool2d_forward_gpu(const Tensor<T>& bottom, Tensor<T>& top, Tensor<T>& argmax, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
    template <typename T>
    void maxpool2d_backward_gpu(const Tensor<T>& top_diff, const Tensor<T>& argmax, Tensor<T>& bottom_diff, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);

    /// Forward pass for 2D convolution on the GPU.
    template <typename T>
    Tensor<T> conv2d_forward_gpu(const Tensor<T>& input, const Tensor<T>& weight, const Tensor<T>& bias, int stride_h, int stride_w, int pad_h, int pad_w);
    /// Backward pass for 2D convolution on the GPU.
    template <typename T>
    void conv2d_backward_gpu(const Tensor<T>& out_grad, const Tensor<T>& input, const Tensor<T>& weight, Tensor<T>& grad_input, Tensor<T>& grad_weight, Tensor<T>& grad_bias, int stride_h, int stride_w, int pad_h, int pad_w);

    /// Autograd node for 2D convolution.
    template <typename T>
    class Conv2dNode : public AutogradNode<T> {
    private:
        Tensor<T> input, weight, bias;
        int stride_h, stride_w, pad_h, pad_w;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        Conv2dNode(Tensor<T> input, Tensor<T> weight, Tensor<T> bias, int stride_h, int stride_w, int pad_h, int pad_w, Tensor<T> out)
            : input(input), weight(weight), bias(bias), stride_h(stride_h), stride_w(stride_w), pad_h(pad_h), pad_w(pad_w), out_ctx(out.ctx) {}
            
        vc::vector<Tensor<T>> get_parents() override {
            vc::vector<Tensor<T>> parents;
            parents.push_back(input); parents.push_back(weight); 
            if (bias.numel() > 0) parents.push_back(bias);
            return parents;
        }

        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            
            Tensor<T> grad_input, grad_weight, grad_bias;
            
            if (input.ctx->requires_grad) {
                if (!input.ctx->grad) input.ctx->grad = std::make_shared<Tensor<T>>(Tensor<T>::empty_gpu(input._shape).zero_());
                grad_input = *input.ctx->grad;
            } else {
                grad_input = Tensor<T>::empty_gpu(input._shape).zero_();
            }
            if (weight.ctx->requires_grad) {
                if (!weight.ctx->grad) weight.ctx->grad = std::make_shared<Tensor<T>>(Tensor<T>::empty_gpu(weight._shape).zero_());
                grad_weight = *weight.ctx->grad;
            } else {
                grad_weight = Tensor<T>::empty_gpu(weight._shape).zero_();
            }
            if (bias.numel() > 0) {
                if (bias.ctx->requires_grad) {
                    if (!bias.ctx->grad) bias.ctx->grad = std::make_shared<Tensor<T>>(Tensor<T>::empty_gpu(bias._shape).zero_());
                    grad_bias = *bias.ctx->grad;
                } else {
                    grad_bias = Tensor<T>::empty_gpu(bias._shape).zero_();
                }
            }
            
            conv2d_backward_gpu(*out_ctx_shared->grad, input, weight, grad_input, grad_weight, grad_bias, stride_h, stride_w, pad_h, pad_w);
        }
    };

    /// Autograd node for max pooling.
    template <typename T>
    class MaxPool2dNode : public AutogradNode<T> {
    private:
        Tensor<T> input, argmax;
        int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        MaxPool2dNode(Tensor<T> input, Tensor<T> argmax, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, Tensor<T> out)
            : input(input), argmax(argmax), kernel_h(kernel_h), kernel_w(kernel_w), stride_h(stride_h), stride_w(stride_w), pad_h(pad_h), pad_w(pad_w), out_ctx(out.ctx) {}
            
        vc::vector<Tensor<T>> get_parents() override {
            vc::vector<Tensor<T>> parents(1); parents[0] = input; return parents;
        }

        void backward() override {
            auto out_ctx_shared = out_ctx.lock();
            if (!out_ctx_shared || !out_ctx_shared->grad) return;
            if (input.ctx->requires_grad) {
                if (!input.ctx->grad) input.ctx->grad = std::make_shared<Tensor<T>>(Tensor<T>::empty_gpu(input._shape).zero_());
                maxpool2d_backward_gpu(*out_ctx_shared->grad, argmax, *input.ctx->grad, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
            }
        }
    };
}

#endif