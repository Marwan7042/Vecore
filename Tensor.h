#ifndef Tensor_H
#define Tensor_H

#include "vector.h"
#include "unordered_map.h"
#include <stdexcept>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace mstd{
    template <typename T> class AutogradNode; 
    template <typename T> struct GPUData;
    template <typename T> struct AutogradContext;
    template <typename T> class Tensor;

    template <typename T>
    class Tensor {
    public:
        mstd::vector<size_t> _shape;
        mstd::vector<size_t> _strides;
        std::shared_ptr<mstd::vector<T>> data;
        std::shared_ptr<AutogradContext<T>> ctx;
        
        bool is_cuda = false;
        std::shared_ptr<GPUData<T>> gpu_data = nullptr;

        // Default constructor: Initializes an empty tensor with a new AutogradContext
        Tensor();

        // Parameterized internal constructor: Used to construct a tensor from pre-existing data, context, and GPU pointers
        Tensor(mstd::vector<size_t> s, mstd::vector<size_t> st, std::shared_ptr<mstd::vector<T>> d, std::shared_ptr<AutogradContext<T>> c, bool cuda=false, std::shared_ptr<GPUData<T>> gd=nullptr);

        // Shape constructor: Allocates memory for a tensor of the given shape and initializes it with zeros
        Tensor(mstd::vector<size_t> shape);

        // Transfers the tensor's memory to the specified device (e.g., "cuda" or "cpu"). Returns a new tensor on that device.
        Tensor<T> to(const std::string& device) const;

        // Accesses an element in the tensor by its multidimensional index. Returns a reference to the element.
        T& operator()(const mstd::vector<size_t>& dims);
        
        // Returns a new tensor that represents the transposed version of this tensor (swaps the last two dimensions).
        Tensor<T> transpose() const;    

        // Adds two tensors element-wise and records the operation in the computation graph. Returns the sum tensor.
        Tensor<T> operator+(const Tensor<T>& other) const;

        // Subtracts 'other' from this tensor element-wise and records the operation in the computation graph. Returns the difference tensor.
        Tensor<T> operator-(const Tensor<T>& other) const;

        // Performs matrix multiplication between this tensor and 'other', recording it in the graph. Returns the product tensor.
        Tensor<T> operator*(const Tensor<T>& other) const;

        // Applies the Rectified Linear Unit (ReLU) activation function element-wise. Returns the activated tensor.
        Tensor<T> relu() const;

        // Initiates the reverse-mode auto-differentiation (backpropagation) from this tensor, computing gradients for all dependent tensors.
        void backward();

        // Prints the shape, strides, and actual data contents of the tensor to standard output.
        void print() const;
    };

    template <typename T>
    struct GPUData {
        T* ptr = nullptr;
        size_t size = 0;
        GPUData(size_t s) : size(s) {
            cudaMalloc((void**)&ptr, size * sizeof(T));
        }
        ~GPUData() {
            if (ptr) cudaFree(ptr);
        }
    };

    template <typename T>
    struct AutogradContext {
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
        virtual mstd::vector<Tensor<T>> get_parents() = 0;
        virtual ~AutogradNode() = default;
    };
    
    // Graph node representing element-wise addition.
    // Propagates the upstream gradient equally to both input tensors.
    template <typename T>
    class AddNode : public AutogradNode<T> {
    private:
        Tensor<T> a, b, out;
    public:
        AddNode(Tensor<T> a, Tensor<T> b, Tensor<T> out) : a(a), b(b), out(out) {}

        mstd::vector<Tensor<T>> get_parents() override {
            mstd::vector<Tensor<T>> parents(2);
            parents[0] = a;
            parents[1] = b;
            return parents;
        }

        void backward() override {
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) a.ctx->grad = std::make_shared<Tensor<T>>(a._shape);
                for (size_t i = 0; i < a.ctx->grad->data->size(); i++) (*a.ctx->grad->data)[i] += (*out.ctx->grad->data)[i];
            }
            if (b.ctx->requires_grad) {
                if (!b.ctx->grad) b.ctx->grad = std::make_shared<Tensor<T>>(b._shape);
                for (size_t i = 0; i < b.ctx->grad->data->size(); i++) (*b.ctx->grad->data)[i] += (*out.ctx->grad->data)[i];
            }
        }
    };

    // Graph node representing element-wise subtraction.
    // Propagates the upstream gradient positively to the left operand and negatively to the right operand.
    template <typename T>
    class SubNode : public AutogradNode<T> {
    private:
        Tensor<T> a, b, out;
    public:
        SubNode(Tensor<T> a, Tensor<T> b, Tensor<T> out) : a(a), b(b), out(out) {}

        mstd::vector<Tensor<T>> get_parents() override {
            mstd::vector<Tensor<T>> parents(2);
            parents[0] = a;
            parents[1] = b;
            return parents;
        }

        void backward() override {
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) a.ctx->grad = std::make_shared<Tensor<T>>(a._shape);
                for (size_t i = 0; i < a.ctx->grad->data->size(); i++) (*a.ctx->grad->data)[i] -= (*out.ctx->grad->data)[i];
            }
            if (b.ctx->requires_grad) {
                if (!b.ctx->grad) b.ctx->grad = std::make_shared<Tensor<T>>(b._shape);
                for (size_t i = 0; i < b.ctx->grad->data->size(); i++) (*b.ctx->grad->data)[i] -= (*out.ctx->grad->data)[i];
            }
        }
    };

    // Graph node representing matrix multiplication.
    // Uses the matrix calculus chain rule (A_grad = out_grad * B^T, B_grad = A^T * out_grad) to propagate gradients.
    template <typename T>
    class MulNode : public AutogradNode<T> {
    private:
        Tensor<T> a, b, out;
    public:
        MulNode(Tensor<T> a, Tensor<T> b, Tensor<T> out) : a(a), b(b), out(out) {}

        mstd::vector<Tensor<T>> get_parents() override {
            mstd::vector<Tensor<T>> parents(2);
            parents[0] = a;
            parents[1] = b;
            return parents;
        }

        void backward() override {
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) a.ctx->grad = std::make_shared<Tensor<T>>(a._shape);
                Tensor<T> a_grad_update = *(out.ctx->grad) * b.transpose();
                *(a.ctx->grad) = *(a.ctx->grad) + a_grad_update;
            }
            if (b.ctx->requires_grad) {
                if (!b.ctx->grad) b.ctx->grad = std::make_shared<Tensor<T>>(b._shape);
                Tensor<T> b_grad_update = a.transpose() * *(out.ctx->grad);
                *(b.ctx->grad) = *(b.ctx->grad) + b_grad_update;
            }
        }
    };

    // Graph node representing the Rectified Linear Unit (ReLU) activation.
    // Propagates gradients only if the original forward pass value was greater than zero.
    template <typename T>
    class ReLU : public AutogradNode<T> {
    private:
        Tensor<T> a, out;
    public:
        ReLU(Tensor<T> a, Tensor<T> out) : a(a), out(out) {}

        mstd::vector<Tensor<T>> get_parents() override {
            mstd::vector<Tensor<T>> parents(1);
            parents[0] = a;
            return parents;
        }

        void backward() override {
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) a.ctx->grad = std::make_shared<Tensor<T>>(a._shape);
                for (size_t i = 0; i < a.ctx->grad->data->size(); i++) if ((*a.data)[i] > 0) (*a.ctx->grad->data)[i] += (*out.ctx->grad->data)[i];
            }
        }
    };

}

#endif