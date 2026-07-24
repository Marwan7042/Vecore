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
    std::vector<Sample> dataset;
    vc::vector<vc::nn::Dense<float>> layers;
    std::vector<float> history_loss;
    std::vector<float> history_acc;

    Sequential() {}

    Sequential(std::initializer_list<vc::nn::Dense<float>> init_layers) {
        for (const auto& layer : init_layers) {
            layers.push_back(layer);
        }
    }
    
    Sequential& to(const std::string& device) {
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].to(device);
        }
        return *this;
    }

    vc::Tensor<float> operator()(const vc::Tensor<float>& x) {
        vc::Tensor<float> h = x;
        for (size_t i = 0; i < layers.size(); i++) {
            h = layers[i](h);
        }
        return h;
    }

    vc::vector<vc::Tensor<float>*> parameters() {
        vc::vector<vc::Tensor<float>*> params;
        for (size_t i = 0; i < layers.size(); i++) {
            auto p = layers[i].parameters();
            params.push_back(p[0]); 
            params.push_back(p[1]);
        }
        return params;
    }

    void data_loader(std::string path) {
        // Skip for now
    }
        
    void fit(vc::data::InMemoryDataset& dataset_obj, int epochs = 150, float learning_rate = 0.1f, int batch_size = 512, int patience = 0, float min_delta = 1e-5f) {
        std::vector<Sample>& train_dataset = dataset_obj.get_raw_data();
        lr = learning_rate;
        vc::optim::SGD<float> optimizer(this->parameters(), lr);
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

        float* d_epoch_loss = nullptr;
        int* d_epoch_correct = nullptr;
        cudaMalloc(&d_epoch_loss, sizeof(float));
        cudaMalloc(&d_epoch_correct, sizeof(int));

        float best_loss = 1e9f;
        int epochs_no_improve = 0;
        std::vector<std::vector<float>> best_weights_cpu;
        std::vector<std::vector<float>> best_bias_cpu;

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
                size_t x_dim = train_dataset[0].image.numel();
                size_t y_dim = train_dataset[0].target.numel();
                
                vc::vector<size_t> x_shape(2); x_shape[0] = current_batch_size; x_shape[1] = x_dim;
                vc::vector<size_t> y_shape(2); y_shape[0] = current_batch_size; y_shape[1] = y_dim;
                vc::vector<size_t> x_strides(2); x_strides[0] = x_dim; x_strides[1] = 1;
                vc::vector<size_t> y_strides(2); y_strides[0] = y_dim; y_strides[1] = 1;
                auto t1 = std::chrono::high_resolution_clock::now();

                float* x_ptr = X_all_cuda.gpu_data->ptr + (start_idx * x_dim);
                float* y_ptr = Y_all_cuda.gpu_data->ptr + (start_idx * y_dim);
                
                auto x_gpu_data = std::make_shared<vc::GPUData<float>>(x_ptr, current_batch_size * x_dim, false);
                auto y_gpu_data = std::make_shared<vc::GPUData<float>>(y_ptr, current_batch_size * y_dim, false);
                vc::Tensor<float> X_batch_cuda(x_shape, x_strides, nullptr, std::make_shared<vc::AutogradContext<float>>(), true, x_gpu_data);
                vc::Tensor<float> Y_batch_cuda(y_shape, y_strides, nullptr, std::make_shared<vc::AutogradContext<float>>(), true, y_gpu_data);
                vc::Tensor<float> Y_pred_gpu = (*this)(X_batch_cuda);
                optimizer.zero_grad();
                Y_pred_gpu.fast_cross_entropy_backward(Y_batch_cuda, d_epoch_loss, d_epoch_correct);
                optimizer.step();
}
            
            cudaMemcpy(&epoch_loss, d_epoch_loss, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&correct, d_epoch_correct, sizeof(int), cudaMemcpyDeviceToHost);
            
            float avg_loss = epoch_loss / train_dataset.size();
            float accuracy = (float)correct / train_dataset.size() * 100.0f;
            
            auto t_epoch_end = std::chrono::high_resolution_clock::now();
            if (epoch <= 2) {
                std::cout << "EPOCH " << epoch << " TOTAL TIME: " << std::chrono::duration<double, std::milli>(t_epoch_end - t_epoch_start).count() << "ms\n";
            }
            
            std::cout << "Epoch " << epoch << "/" << epochs 
                      << " - Loss: " << avg_loss 
                      << " - Acc: " << accuracy << "%" << std::endl;
            
            history_loss.push_back(avg_loss);
            history_acc.push_back(accuracy);
            
            if (patience > 0) {
                if (avg_loss < best_loss - min_delta) {
                    best_loss = avg_loss;
                    epochs_no_improve = 0;
                    
                    best_weights_cpu.clear();
                    best_bias_cpu.clear();
                    for (auto& layer : layers) {
                        std::vector<float> w(layer.weight.numel());
                        std::vector<float> b(layer.bias.numel());
                        if (layer.weight.is_cuda) {
#ifdef __CUDACC__
                            cudaMemcpy(w.data(), layer.weight.gpu_data->ptr, w.size() * sizeof(float), cudaMemcpyDeviceToHost);
#endif
                        } else {
                            std::memcpy(w.data(), layer.weight.data->begin(), w.size() * sizeof(float));
                        }
                        if (layer.bias.is_cuda) {
#ifdef __CUDACC__
                            cudaMemcpy(b.data(), layer.bias.gpu_data->ptr, b.size() * sizeof(float), cudaMemcpyDeviceToHost);
#endif
                        } else {
                            std::memcpy(b.data(), layer.bias.data->begin(), b.size() * sizeof(float));
                        }
                        best_weights_cpu.push_back(w);
                        best_bias_cpu.push_back(b);
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
            for (size_t l = 0; l < layers.size(); l++) {
                if (layers[l].weight.is_cuda) {
#ifdef __CUDACC__
                    cudaMemcpy(layers[l].weight.gpu_data->ptr, best_weights_cpu[l].data(), best_weights_cpu[l].size() * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(layers[l].bias.gpu_data->ptr, best_bias_cpu[l].data(), best_bias_cpu[l].size() * sizeof(float), cudaMemcpyHostToDevice);
#endif
                } else {
                    std::memcpy(layers[l].weight.data->begin(), best_weights_cpu[l].data(), best_weights_cpu[l].size() * sizeof(float));
                    std::memcpy(layers[l].bias.data->begin(), best_bias_cpu[l].data(), best_bias_cpu[l].size() * sizeof(float));
                }
            }
        }
    }

    void save(const std::string& filepath) {
        // Bring weights back to CPU so we can read them safely
        this->to("cpu");
        
        std::ofstream weights_file(filepath);
        weights_file << std::setprecision(8);
        weights_file << "{\n";
        
        for (size_t l = 0; l < layers.size(); l++) {
            // Write weight tensor
            vc::Tensor<float> w_save = layers[l].weight.transpose();
            weights_file << "  \"w" << (l+1) << "\": {";
            weights_file << "\"shape\": [" << w_save._shape[0] << ", " << w_save._shape[1] << "], ";
            weights_file << "\"data\": [";
            for (size_t i = 0; i < w_save.numel(); i++) {
                weights_file << (*w_save.data)[i];
                if (i + 1 < w_save.numel()) weights_file << ",";
            }
            weights_file << "]}";
            weights_file << ",\n";
            
            // Write bias tensor
            weights_file << "  \"b" << (l+1) << "\": {";
            weights_file << "\"shape\": [" << layers[l].bias._shape[0] << ", " << layers[l].bias._shape[1] << "], ";
            weights_file << "\"data\": [";
            for (size_t i = 0; i < layers[l].bias.numel(); i++) {
                weights_file << (*layers[l].bias.data)[i];
                if (i + 1 < layers[l].bias.numel()) weights_file << ",";
            }
            weights_file << "]}";
            
            if (l + 1 < layers.size()) weights_file << ",\n";
            else weights_file << "\n";
        }
        
        weights_file << "}\n";
        weights_file.close();
        std::cout << "Model weights successfully exported to " << filepath << "!" << std::endl;
    }
};