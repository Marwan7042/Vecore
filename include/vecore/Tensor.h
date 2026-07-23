#ifndef Tensor_H
#define Tensor_H

#include "vector.h"
#include "unordered_map.h"
#include <stdexcept>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace vc{
    template <typename T> class AutogradNode; 
    template <typename T> struct GPUData;
    template <typename T> struct AutogradContext;
    template <typename T> class Tensor;

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

        // Default constructor: Initializes an empty tensor with a new AutogradContext
        Tensor();

        // Parameterized internal constructor: Used to construct a tensor from pre-existing data, context, and GPU pointers
        Tensor(vc::vector<size_t> s, vc::vector<size_t> st, std::shared_ptr<vc::vector<T>> d, std::shared_ptr<AutogradContext<T>> c, bool cuda=false, std::shared_ptr<GPUData<T>> gd=nullptr);

        // Shape constructor: Allocates memory for a tensor of the given shape and initializes it with zeros
        Tensor(vc::vector<size_t> shape);
        Tensor(vc::vector<size_t> shape, bool zero_memory, bool allocate_cpu = true);

        // Transfers the tensor's memory to the specified device (e.g., "cuda" or "cpu"). Returns a new tensor on that device.
        Tensor<T> to(const std::string& device) const;

        // Accesses an element in the tensor by its multidimensional index. Returns a reference to the element.
        T& operator()(const vc::vector<size_t>& dims);
        
        // Returns a new tensor that represents the transposed version of this tensor (swaps the last two dimensions).
        Tensor<T> transpose() const;    

        // Adds two tensors element-wise and records the operation in the computation graph. Returns the sum tensor.
        Tensor<T> operator+(const Tensor<T>& other) const;

        // Subtracts 'other' from this tensor element-wise and records the operation in the computation graph. Returns the difference tensor.
        Tensor<T> operator-(const Tensor<T>& other) const;

        // Performs matrix multiplication between this tensor and 'other', recording it in the graph. Returns the product tensor.
        Tensor<T> operator*(const Tensor<T>& other) const;

        // Applies the Rectified Linear Unit (ReLU) activation function element-wise. Returns the activated tensor.
        Tensor<T> matmul(const Tensor<T>& other, bool transA = false, bool transB = false) const;
        Tensor<T> sum(const vc::vector<int>& dims, bool keepdim = false) const;
        Tensor<T> relu() const;
        Tensor<T> relu_backward(const Tensor<T>& out_grad) const;
        
        size_t numel() const {
            return _numel;
        }

        static Tensor<T> empty_gpu(const vc::vector<size_t>& shape);
        
        void fast_cross_entropy_backward(const Tensor<T>& target, float* d_loss, int* d_correct);
        
        // Optimizer hooks
        void sgd_update(float lr);
        void zero_grad_data();

        // Reshapes the tensor to a new shape without modifying underlying data.
        Tensor<T> reshape(const vc::vector<size_t>& new_shape) const;

        // Initiates the reverse-mode auto-differentiation (backpropagation) from this tensor, computing gradients for all dependent tensors.
        void backward();

        // Prints the shape, strides, and actual data contents of the tensor to standard output.
        void print() const;
    };

    #include <unordered_map>
    #include <vector>

    template <typename T>
    struct CachingAllocator {
        static std::unordered_map<size_t, std::vector<T*>> free_blocks;
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
    inline std::unordered_map<size_t, std::vector<T*>> CachingAllocator<T>::free_blocks;

    template <typename T>
    struct GPUData {
        T* ptr = nullptr;
        size_t size = 0;
        GPUData(size_t s) : size(s) {
            ptr = CachingAllocator<T>::allocate(size);
        }
        ~GPUData() {
            if (ptr) CachingAllocator<T>::free(ptr, size);
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
                *(a.ctx->grad) = *(a.ctx->grad) + reduced;
            }
            if (b.ctx->requires_grad) {
                Tensor<T> reduced = reduce_grad(b, *out_ctx_shared->grad);
                if (!b.ctx->grad) {
                    b.ctx->grad = std::make_shared<Tensor<T>>(reduced);
                } else {
                    *(b.ctx->grad) = *(b.ctx->grad) + reduced;
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
                    *(a.ctx->grad) = *(a.ctx->grad) + reduced;
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
                for (size_t i = 0; i < b.ctx->grad->numel(); i++) (*b.ctx->grad->data)[i] -= (*reduced.data)[i];
            }
        }
    };

    // Graph node representing matrix multiplication.
    // Uses the matrix calculus chain rule (A_grad = out_grad * B^T, B_grad = A^T * out_grad) to propagate gradients.
    template <typename T>
    class MulNode : public AutogradNode<T> {
    private:
        Tensor<T> a, b;
        std::weak_ptr<AutogradContext<T>> out_ctx;
    public:
        MulNode(Tensor<T> a, Tensor<T> b, Tensor<T> out) : a(a), b(b), out_ctx(out.ctx) {}

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
                Tensor<T> a_grad_update = *(out_ctx_shared->grad) * b.transpose();
                if (!a.ctx->grad) {
                    a.ctx->grad = std::make_shared<Tensor<T>>(a_grad_update);
                } else {
                    *(a.ctx->grad) = *(a.ctx->grad) + a_grad_update;
                }
            }
            if (b.ctx->requires_grad) {
                Tensor<T> b_grad_update = a.transpose() * *(out_ctx_shared->grad);
                if (!b.ctx->grad) {
                    b.ctx->grad = std::make_shared<Tensor<T>>(b_grad_update);
                } else {
                    *(b.ctx->grad) = *(b.ctx->grad) + b_grad_update;
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
                *(a.ctx->grad) = *(a.ctx->grad) + a.relu_backward(*(out_ctx_shared->grad));
            }
        }
    };

    // Graph node representing device transfers (.to("cuda") or .to("cpu"))
    // Automatically propagates gradients across the PCIe bus during backward pass.
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
                *(a.ctx->grad) = *(a.ctx->grad) + grad_moved;
            }
        }
    };

}

#endif