/// High-level model wrapper that chains layers and drives training.
#pragma once
#include "vecore/nn.h"
#include "vecore/optim.h"
#include "vecore/data.h"
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cuda_runtime.h>


class Sequential {
private:
    float lr = 0.01;
    std::string path;
    
public:
    /// Training samples used by the wrapper.
    vc::vector<Sample> dataset;
    /// Layer stack in forward order.
    vc::vector<std::shared_ptr<vc::nn::Layer<float>>> layers;
    /// Per-epoch loss history.
    vc::vector<float> history_loss;
    /// Per-epoch accuracy history.
    vc::vector<float> history_acc;

    /// Creates an empty Sequential model.
    Sequential() {}

    /// Constructs a Sequential model from a list of layer instances.
    template<typename... Args>
    Sequential(Args... args) {
        (layers.push_back(std::make_shared<Args>(args)), ...);
    }
    
    /// Appends a layer to the model.
    void add(std::shared_ptr<vc::nn::Layer<float>> layer) {
        layers.push_back(layer);
    }
    
    /// Moves every layer to the requested device.
    Sequential& to(const std::string& device) {
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i]->to(device);
        }
        return *this;
    }

    /// Runs a forward pass through the full model.
    vc::Tensor<float> operator()(const vc::Tensor<float>& x) {
        vc::Tensor<float> h = x;
        for (size_t i = 0; i < layers.size(); i++) {
            h = (*layers[i])(h);
        }
        return h;
    }

    /// Returns pointers to every trainable parameter in the model.
    vc::vector<vc::Tensor<float>*> parameters() {
        vc::vector<vc::Tensor<float>*> params;
        for (size_t i = 0; i < layers.size(); i++) {
            auto p = layers[i]->parameters();
            for (size_t j = 0; j < p.size(); j++) {
                params.push_back(p[j]);
            }
        }
        return params;
    }

    /// Placeholder dataset loader hook.
    void data_loader(std::string path) {
        // Skip for now
    }
    /// Selected optimizer name used by compile() and fit().
    std::string opt_name = "adam";

    /// Stores the chosen optimizer configuration.
    void compile(const std::string& optimizer_name, vc::nn::SparseCategoricalCrossentropy<float> loss = vc::nn::SparseCategoricalCrossentropy<float>()) {
        opt_name = optimizer_name;
    }
        
    /// Trains the model on a dataset for a number of epochs.
    void fit(std::vector<Sample>& train_dataset, int epochs = 50, float learning_rate = 0.001f, int batch_size = 1024, int patience = 0, float min_delta = 1e-5f, std::vector<Sample>* val_dataset = nullptr) {
        lr = learning_rate;
        
        vc::optim::SGD<float> sgd(this->parameters(), lr);
        vc::optim::Adam<float> adam(this->parameters(), lr);
        
        vc::nn::CrossEntropyLoss<float> criterion;

        history_loss.clear();
        history_acc.clear();
        
        vc::Tensor<float> X_all_cuda;
        vc::Tensor<float> Y_all_cuda;

        if (train_dataset.size() > 0) {
            size_t x_dim = train_dataset[0].image.numel();
            size_t y_dim = train_dataset[0].target.numel();
            
            float* h_X_all = (float*)std::malloc(train_dataset.size() * x_dim * sizeof(float));
            float* h_Y_all = (float*)std::malloc(train_dataset.size() * y_dim * sizeof(float));
            
            for (size_t b = 0; b < train_dataset.size(); b++) {
                std::memcpy(h_X_all + b * x_dim, train_dataset[b].image.data->begin(), x_dim * sizeof(float));
                std::memcpy(h_Y_all + b * y_dim, train_dataset[b].target.data->begin(), y_dim * sizeof(float));
            }

            vc::vector<size_t> all_x_shape(2);
            all_x_shape[0] = train_dataset.size();
            all_x_shape[1] = x_dim;
            
            vc::vector<size_t> all_y_shape(2);
            all_y_shape[0] = train_dataset.size();
            all_y_shape[1] = y_dim;
            
            X_all_cuda = vc::Tensor<float>::empty_gpu(all_x_shape);
            Y_all_cuda = vc::Tensor<float>::empty_gpu(all_y_shape);

            cudaMemcpy(X_all_cuda.gpu_data->ptr, h_X_all, train_dataset.size() * x_dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(Y_all_cuda.gpu_data->ptr, h_Y_all, train_dataset.size() * y_dim * sizeof(float), cudaMemcpyHostToDevice);

            std::free(h_X_all);
            std::free(h_Y_all);
        }

        vc::Tensor<float> X_val_all_cuda;
        vc::Tensor<float> Y_val_all_cuda;
        if (val_dataset != nullptr && val_dataset->size() > 0) {
            size_t x_dim = (*val_dataset)[0].image.numel();
            size_t y_dim = (*val_dataset)[0].target.numel();
            
            float* h_X_val = (float*)std::malloc(val_dataset->size() * x_dim * sizeof(float));
            float* h_Y_val = (float*)std::malloc(val_dataset->size() * y_dim * sizeof(float));
            
            for (size_t b = 0; b < val_dataset->size(); b++) {
                std::memcpy(h_X_val + b * x_dim, (*val_dataset)[b].image.data->begin(), x_dim * sizeof(float));
                std::memcpy(h_Y_val + b * y_dim, (*val_dataset)[b].target.data->begin(), y_dim * sizeof(float));
            }

            vc::vector<size_t> all_x_shape(2);
            all_x_shape[0] = val_dataset->size();
            all_x_shape[1] = x_dim;
            
            vc::vector<size_t> all_y_shape(2);
            all_y_shape[0] = val_dataset->size();
            all_y_shape[1] = y_dim;
            
            X_val_all_cuda = vc::Tensor<float>::empty_gpu(all_x_shape);
            Y_val_all_cuda = vc::Tensor<float>::empty_gpu(all_y_shape);

            cudaMemcpy(X_val_all_cuda.gpu_data->ptr, h_X_val, val_dataset->size() * x_dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(Y_val_all_cuda.gpu_data->ptr, h_Y_val, val_dataset->size() * y_dim * sizeof(float), cudaMemcpyHostToDevice);

            std::free(h_X_val);
            std::free(h_Y_val);
        }

        float* d_epoch_loss = nullptr;
        int* d_epoch_correct = nullptr;
        cudaMalloc(&d_epoch_loss, sizeof(float));
        cudaMalloc(&d_epoch_correct, sizeof(int));

        float best_loss = 1e9f;
        int epochs_no_improve = 0;
        std::vector<std::vector<float>> best_params_cpu;

        for (int epoch = 0; epoch <= epochs; epoch++) {
            auto t_epoch_start = std::chrono::high_resolution_clock::now();
            cudaMemsetAsync(d_epoch_loss, 0, sizeof(float));
            cudaMemsetAsync(d_epoch_correct, 0, sizeof(int));
            
            float epoch_loss = 0.0f;
            int correct = 0;
            
            int num_batches = (train_dataset.size() + batch_size - 1) / batch_size;
            
            std::chrono::high_resolution_clock::time_point t_step_end;
            for (int b = 0; b < num_batches; b++) {                auto t0 = std::chrono::high_resolution_clock::now();
                size_t start_idx = b * batch_size;
                int current_batch_size = std::min((int)train_dataset.size() - (int)start_idx, batch_size);
                vc::vector<size_t> x_shape;
                if (train_dataset[0].image._shape.size() > 0 && train_dataset[0].image._shape[0] == 1) {
                    x_shape = train_dataset[0].image._shape;
                    x_shape[0] = current_batch_size;
                } else {
                    x_shape.push_back(current_batch_size);
                    for (size_t i = 0; i < train_dataset[0].image._shape.size(); i++) {
                        x_shape.push_back(train_dataset[0].image._shape[i]);
                    }
                }
                
                vc::vector<size_t> y_shape;
                if (train_dataset[0].target._shape.size() > 0 && train_dataset[0].target._shape[0] == 1) {
                    y_shape = train_dataset[0].target._shape;
                    y_shape[0] = current_batch_size;
                } else {
                    y_shape.push_back(current_batch_size);
                    for (size_t i = 0; i < train_dataset[0].target._shape.size(); i++) {
                        y_shape.push_back(train_dataset[0].target._shape[i]);
                    }
                }
                
                vc::vector<size_t> x_strides(x_shape.size());
                size_t c_s = 1;
                for (int i = x_shape.size() - 1; i >= 0; i--) {
                    x_strides[i] = c_s;
                    c_s *= x_shape[i];
                }
                
                vc::vector<size_t> y_strides(y_shape.size());
                c_s = 1;
                for (int i = y_shape.size() - 1; i >= 0; i--) {
                    y_strides[i] = c_s;
                    c_s *= y_shape[i];
                }
                
                size_t x_dim = train_dataset[0].image.numel();
                size_t y_dim = train_dataset[0].target.numel();
                
                auto t1 = std::chrono::high_resolution_clock::now();

                float* x_ptr = X_all_cuda.gpu_data->ptr + (start_idx * x_dim);
                float* y_ptr = Y_all_cuda.gpu_data->ptr + (start_idx * y_dim);
                
                auto x_gpu_data = std::make_shared<vc::GPUData<float>>(x_ptr, current_batch_size * x_dim, false);
                auto y_gpu_data = std::make_shared<vc::GPUData<float>>(y_ptr, current_batch_size * y_dim, false);
                vc::Tensor<float> X_batch_cuda(x_shape, x_strides, nullptr, std::make_shared<vc::AutogradContext<float>>(), true, x_gpu_data);
                vc::Tensor<float> Y_batch_cuda(y_shape, y_strides, nullptr, std::make_shared<vc::AutogradContext<float>>(), true, y_gpu_data);
                vc::Tensor<float> Y_pred_gpu = (*this)(X_batch_cuda);
                
                if (opt_name == "adam") adam.zero_grad();
                else sgd.zero_grad();
                
                Y_pred_gpu.fast_cross_entropy_backward(Y_batch_cuda, d_epoch_loss, d_epoch_correct);
                
                if (opt_name == "adam") adam.step();
                else sgd.step();
            }
            
            cudaMemcpy(&epoch_loss, d_epoch_loss, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&correct, d_epoch_correct, sizeof(int), cudaMemcpyDeviceToHost);
            
            float avg_loss = epoch_loss / train_dataset.size();
            float accuracy = (float)correct / train_dataset.size() * 100.0f;
            
            float val_avg_loss = avg_loss;
            float val_accuracy = accuracy;
            
            if (val_dataset != nullptr && val_dataset->size() > 0) {
                cudaMemsetAsync(d_epoch_loss, 0, sizeof(float));
                cudaMemsetAsync(d_epoch_correct, 0, sizeof(int));
                
                int val_num_batches = (val_dataset->size() + batch_size - 1) / batch_size;
                vc::AutogradContext<float>::grad_mode = false;
                for (int b = 0; b < val_num_batches; b++) {
                    size_t start_idx = b * batch_size;
                    int current_batch_size = std::min((int)val_dataset->size() - (int)start_idx, batch_size);
                    vc::vector<size_t> x_shape;
                    if ((*val_dataset)[0].image._shape.size() > 0 && (*val_dataset)[0].image._shape[0] == 1) {
                        x_shape = (*val_dataset)[0].image._shape;
                        x_shape[0] = current_batch_size;
                    } else {
                        x_shape.push_back(current_batch_size);
                        for (size_t i = 0; i < (*val_dataset)[0].image._shape.size(); i++) x_shape.push_back((*val_dataset)[0].image._shape[i]);
                    }
                    vc::vector<size_t> y_shape;
                    if ((*val_dataset)[0].target._shape.size() > 0 && (*val_dataset)[0].target._shape[0] == 1) {
                        y_shape = (*val_dataset)[0].target._shape;
                        y_shape[0] = current_batch_size;
                    } else {
                        y_shape.push_back(current_batch_size);
                        for (size_t i = 0; i < (*val_dataset)[0].target._shape.size(); i++) y_shape.push_back((*val_dataset)[0].target._shape[i]);
                    }
                    
                    vc::vector<size_t> x_strides(x_shape.size());
                    size_t c_s = 1; for (int i = x_shape.size() - 1; i >= 0; i--) { x_strides[i] = c_s; c_s *= x_shape[i]; }
                    vc::vector<size_t> y_strides(y_shape.size());
                    c_s = 1; for (int i = y_shape.size() - 1; i >= 0; i--) { y_strides[i] = c_s; c_s *= y_shape[i]; }
                    
                    size_t x_dim = (*val_dataset)[0].image.numel();
                    size_t y_dim = (*val_dataset)[0].target.numel();
                    
                    float* x_ptr = X_val_all_cuda.gpu_data->ptr + (start_idx * x_dim);
                    float* y_ptr = Y_val_all_cuda.gpu_data->ptr + (start_idx * y_dim);
                    
                    auto x_gpu_data = std::make_shared<vc::GPUData<float>>(x_ptr, current_batch_size * x_dim, false);
                    auto y_gpu_data = std::make_shared<vc::GPUData<float>>(y_ptr, current_batch_size * y_dim, false);
                    vc::Tensor<float> X_batch_cuda(x_shape, x_strides, nullptr, std::make_shared<vc::AutogradContext<float>>(), true, x_gpu_data);
                    vc::Tensor<float> Y_batch_cuda(y_shape, y_strides, nullptr, std::make_shared<vc::AutogradContext<float>>(), true, y_gpu_data);
                    
                    vc::Tensor<float> Y_pred_gpu = (*this)(X_batch_cuda);
                    Y_pred_gpu.fast_cross_entropy_backward(Y_batch_cuda, d_epoch_loss, d_epoch_correct);
                }
                vc::AutogradContext<float>::grad_mode = true;
                
                float v_epoch_loss = 0.0f;
                int v_correct = 0;
                cudaMemcpy(&v_epoch_loss, d_epoch_loss, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&v_correct, d_epoch_correct, sizeof(int), cudaMemcpyDeviceToHost);
                val_avg_loss = v_epoch_loss / val_dataset->size();
                val_accuracy = (float)v_correct / val_dataset->size() * 100.0f;
            }
            
            auto t_epoch_end = std::chrono::high_resolution_clock::now();
            if (epoch <= 2) {
                std::cout << "EPOCH " << epoch << " TOTAL TIME: " << std::chrono::duration<double, std::milli>(t_epoch_end - t_epoch_start).count() << "ms\n";
            }
            
            std::cout << "Epoch " << epoch << "/" << epochs 
                      << " - Loss: " << avg_loss 
                      << " - Acc: " << accuracy << "%";
            if (val_dataset != nullptr && val_dataset->size() > 0) {
                std::cout << " - Val Loss: " << val_avg_loss 
                          << " - Val Acc: " << val_accuracy << "%";
            }
            std::cout << std::endl;
            
            history_loss.push_back(val_avg_loss);
            history_acc.push_back(val_accuracy);
            
            if (patience > 0) {
                if (val_avg_loss < best_loss - min_delta) {
                    best_loss = val_avg_loss;
                    epochs_no_improve = 0;
                    
                    best_params_cpu.clear();
                    vc::vector<vc::Tensor<float>*> params = this->parameters();
                    for (size_t i = 0; i < params.size(); i++) {
                        std::vector<float> p_copy(params[i]->numel());
                        if (params[i]->is_cuda) {
#ifdef __CUDACC__
                            cudaMemcpy(p_copy.data(), params[i]->gpu_data->ptr, p_copy.size() * sizeof(float), cudaMemcpyDeviceToHost);
#endif
                        } else {
                            std::memcpy(p_copy.data(), params[i]->data->begin(), p_copy.size() * sizeof(float));
                        }
                        best_params_cpu.push_back(p_copy);
                    }
                } else {
                    epochs_no_improve++;
                    if (epochs_no_improve >= patience) {
                        std::cout << "Early stopping triggered at epoch " << epoch << " (no improvement for " << patience << " epochs)." << std::endl;
                        break;
                    }
                }
            }
        }
        
        cudaFree(d_epoch_loss);
        cudaFree(d_epoch_correct);
        
        if (patience > 0 && epochs_no_improve > 0) {
            std::cout << "Restoring best model weights (Loss: " << best_loss << ")" << std::endl;
            vc::vector<vc::Tensor<float>*> params = this->parameters();
            for (size_t i = 0; i < params.size(); i++) {
                if (params[i]->is_cuda) {
#ifdef __CUDACC__
                    cudaMemcpy(params[i]->gpu_data->ptr, best_params_cpu[i].data(), best_params_cpu[i].size() * sizeof(float), cudaMemcpyHostToDevice);
#endif
                } else {
                    std::memcpy(params[i]->data->begin(), best_params_cpu[i].data(), best_params_cpu[i].size() * sizeof(float));
                }
            }
        }
    }

    /// Saves all model parameters to a binary file.
    void save(const std::string& filepath) {
        vc::vector<vc::Tensor<float>*> params = this->parameters();
        
        std::ofstream weights_file(filepath, std::ios::binary);
        if (!weights_file) {
            std::cerr << "Failed to open " << filepath << " for writing\n";
            return;
        }
        
        for (size_t i = 0; i < params.size(); i++) {
            vc::Tensor<float> cpu_t = params[i]->to("cpu");
            size_t size = cpu_t.numel() * sizeof(float);
            weights_file.write(reinterpret_cast<const char*>(cpu_t.data->begin()), size);
        }
        
        weights_file.close();
        std::cout << "Model weights successfully exported to " << filepath << " in pure binary format!" << std::endl;
    }
    
    /// Loads all model parameters from a binary file.
    void load(const std::string& filepath) {
        vc::vector<vc::Tensor<float>*> params = this->parameters();

        std::ifstream weights_file(filepath, std::ios::binary);
        if (!weights_file) {
            std::cerr << "Failed to open " << filepath << " for reading\n";
            return;
        }
        for (size_t i = 0; i < params.size(); i++) {
            vc::Tensor<float> cpu_t = params[i]->to("cpu");
            size_t size = cpu_t.numel() * sizeof(float);
            weights_file.read(reinterpret_cast<char*>(cpu_t.data->begin()), size);

            if (params[i]->is_cuda) {
#ifdef __CUDACC__
                cudaMemcpy(params[i]->gpu_data->ptr, cpu_t.data->begin(), size, cudaMemcpyHostToDevice);
#endif
            } else {
                std::memcpy(params[i]->data->begin(), cpu_t.data->begin(), size);
            }
        }
        weights_file.close();
        std::cout << "Model weights successfully loaded from " << filepath << " (binary format)!" << std::endl;
    }

    /// Evaluates the model on a test dataset and prints the final metrics.
    void evaluate(std::vector<Sample>& test_dataset, int batch_size = 64) {
        if (test_dataset.empty()) return;
        
        size_t x_dim = test_dataset[0].image.numel();
        size_t y_dim = test_dataset[0].target.numel();
        float* h_X_test = (float*)std::malloc(test_dataset.size() * x_dim * sizeof(float));
        float* h_Y_test = (float*)std::malloc(test_dataset.size() * y_dim * sizeof(float));
        for (size_t b = 0; b < test_dataset.size(); b++) {
            std::memcpy(h_X_test + b * x_dim, test_dataset[b].image.data->begin(), x_dim * sizeof(float));
            std::memcpy(h_Y_test + b * y_dim, test_dataset[b].target.data->begin(), y_dim * sizeof(float));
        }

        vc::vector<size_t> all_x_shape(2); all_x_shape[0] = test_dataset.size(); all_x_shape[1] = x_dim;
        vc::vector<size_t> all_y_shape(2); all_y_shape[0] = test_dataset.size(); all_y_shape[1] = y_dim;
        
        vc::Tensor<float> X_test_cuda = vc::Tensor<float>::empty_gpu(all_x_shape);
        vc::Tensor<float> Y_test_cuda = vc::Tensor<float>::empty_gpu(all_y_shape);

        cudaMemcpy(X_test_cuda.gpu_data->ptr, h_X_test, test_dataset.size() * x_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Y_test_cuda.gpu_data->ptr, h_Y_test, test_dataset.size() * y_dim * sizeof(float), cudaMemcpyHostToDevice);
        std::free(h_X_test); std::free(h_Y_test);

        float* d_test_loss = nullptr;
        int* d_test_correct = nullptr;
        cudaMalloc(&d_test_loss, sizeof(float));
        cudaMalloc(&d_test_correct, sizeof(int));
        cudaMemsetAsync(d_test_loss, 0, sizeof(float));
        cudaMemsetAsync(d_test_correct, 0, sizeof(int));

        int num_batches = (test_dataset.size() + batch_size - 1) / batch_size;
        vc::AutogradContext<float>::grad_mode = false;
        
        for (int b = 0; b < num_batches; b++) {
            size_t start_idx = b * batch_size;
            int current_batch_size = std::min((int)test_dataset.size() - (int)start_idx, batch_size);
            
            vc::vector<size_t> x_shape;
            if (test_dataset[0].image._shape.size() > 0 && test_dataset[0].image._shape[0] == 1) {
                x_shape = test_dataset[0].image._shape; x_shape[0] = current_batch_size;
            } else {
                x_shape.push_back(current_batch_size);
                for (size_t i = 0; i < test_dataset[0].image._shape.size(); i++) x_shape.push_back(test_dataset[0].image._shape[i]);
            }
            vc::vector<size_t> y_shape;
            if (test_dataset[0].target._shape.size() > 0 && test_dataset[0].target._shape[0] == 1) {
                y_shape = test_dataset[0].target._shape; y_shape[0] = current_batch_size;
            } else {
                y_shape.push_back(current_batch_size);
                for (size_t i = 0; i < test_dataset[0].target._shape.size(); i++) y_shape.push_back(test_dataset[0].target._shape[i]);
            }
            
            vc::vector<size_t> x_strides(x_shape.size());
            size_t c_s = 1; for (int i = x_shape.size() - 1; i >= 0; i--) { x_strides[i] = c_s; c_s *= x_shape[i]; }
            vc::vector<size_t> y_strides(y_shape.size());
            c_s = 1; for (int i = y_shape.size() - 1; i >= 0; i--) { y_strides[i] = c_s; c_s *= y_shape[i]; }
            
            float* x_ptr = X_test_cuda.gpu_data->ptr + (start_idx * x_dim);
            float* y_ptr = Y_test_cuda.gpu_data->ptr + (start_idx * y_dim);
            
            auto x_gpu_data = std::make_shared<vc::GPUData<float>>(x_ptr, current_batch_size * x_dim, false);
            auto y_gpu_data = std::make_shared<vc::GPUData<float>>(y_ptr, current_batch_size * y_dim, false);
            vc::Tensor<float> X_batch_cuda(x_shape, x_strides, nullptr, std::make_shared<vc::AutogradContext<float>>(), true, x_gpu_data);
            vc::Tensor<float> Y_batch_cuda(y_shape, y_strides, nullptr, std::make_shared<vc::AutogradContext<float>>(), true, y_gpu_data);
            
            vc::Tensor<float> Y_pred_gpu = (*this)(X_batch_cuda);
            Y_pred_gpu.fast_cross_entropy_backward(Y_batch_cuda, d_test_loss, d_test_correct);
        }
        vc::AutogradContext<float>::grad_mode = true;
        
        float test_loss = 0.0f;
        int test_correct = 0;
        cudaMemcpy(&test_loss, d_test_loss, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&test_correct, d_test_correct, sizeof(int), cudaMemcpyDeviceToHost);
        
        float avg_loss = test_loss / test_dataset.size();
        float accuracy = (float)test_correct / test_dataset.size() * 100.0f;
        
        std::cout << "\n=============================================" << std::endl;
        std::cout << "TEST SET EVALUATION RESULTS:" << std::endl;
        std::cout << "Test Loss: " << avg_loss << " - Test Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "=============================================\n" << std::endl;
        
        cudaFree(d_test_loss);
        cudaFree(d_test_correct);
    }
};