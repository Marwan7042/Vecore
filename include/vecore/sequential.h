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
        
    void fit(vc::data::InMemoryDataset& dataset_obj, int epochs = 150, float learning_rate = 0.1f, int batch_size = 512) {
        std::vector<Sample>& train_dataset = dataset_obj.get_raw_data();
        lr = learning_rate;
        vc::optim::SGD<float> optimizer(this->parameters(), lr);
        vc::nn::CrossEntropyLoss<float> criterion;

        history_loss.clear();
        history_acc.clear();
        
        float* h_X_all = nullptr;
        float* h_Y_all = nullptr;
        cudaStream_t copy_stream;
        cudaEvent_t copy_done[2];
        bool use_pipeline = false;
        
        if (train_dataset.size() > 0) {
            size_t x_dim = train_dataset[0].image.numel();
            size_t y_dim = train_dataset[0].target.numel();
            
            h_X_all = (float*)std::malloc(train_dataset.size() * x_dim * sizeof(float));
            h_Y_all = (float*)std::malloc(train_dataset.size() * y_dim * sizeof(float));
            
            // PRE-PACK ENTIRE DATASET INTO CONTIGUOUS MEMORY ONCE!
            for (size_t b = 0; b < train_dataset.size(); b++) {
                std::memcpy(h_X_all + b * x_dim, train_dataset[b].image.data->begin(), x_dim * sizeof(float));
                std::memcpy(h_Y_all + b * y_dim, train_dataset[b].target.data->begin(), y_dim * sizeof(float));
            }
            
            // Try to pin the memory for faster async transfers
            if (cudaHostRegister(h_X_all, train_dataset.size() * x_dim * sizeof(float), cudaHostRegisterDefault) == cudaSuccess &&
                cudaHostRegister(h_Y_all, train_dataset.size() * y_dim * sizeof(float), cudaHostRegisterDefault) == cudaSuccess) {
                use_pipeline = true;
                cudaStreamCreate(&copy_stream);
                cudaEventCreate(&copy_done[0]);
                cudaEventCreate(&copy_done[1]);
            } else {
                std::cout << "Notice: Could not pin memory. Proceeding with pageable memory." << std::endl;
            }
        }

        float* d_epoch_loss = nullptr;
        int* d_epoch_correct = nullptr;
        if (use_pipeline) {
            cudaMalloc(&d_epoch_loss, sizeof(float));
            cudaMalloc(&d_epoch_correct, sizeof(int));
        }

        for (int epoch = 0; epoch <= epochs; epoch++) {
            if (use_pipeline) {
                cudaMemset(d_epoch_loss, 0, sizeof(float));
                cudaMemset(d_epoch_correct, 0, sizeof(int));
            }
            float epoch_loss = 0.0f;
            int correct = 0;
            
            int num_batches = (train_dataset.size() + batch_size - 1) / batch_size;
            vc::Tensor<float> next_X_cuda;
            vc::Tensor<float> next_Y_cuda;
            
            if (use_pipeline && num_batches > 0) {
                int current_batch_size = std::min((int)train_dataset.size(), batch_size);
                size_t x_dim = train_dataset[0].image.numel();
                size_t y_dim = train_dataset[0].target.numel();
                
                vc::vector<size_t> x_shape(2); x_shape[0] = current_batch_size; x_shape[1] = x_dim;
                next_X_cuda = vc::Tensor<float>::empty_gpu(x_shape);
                
                vc::vector<size_t> y_shape(2); y_shape[0] = current_batch_size; y_shape[1] = y_dim;
                next_Y_cuda = vc::Tensor<float>::empty_gpu(y_shape);
                
                cudaMemcpyAsync(next_X_cuda.gpu_data->ptr, h_X_all, current_batch_size * x_dim * sizeof(float), cudaMemcpyHostToDevice, copy_stream);
                cudaMemcpyAsync(next_Y_cuda.gpu_data->ptr, h_Y_all, current_batch_size * y_dim * sizeof(float), cudaMemcpyHostToDevice, copy_stream);
                cudaEventRecord(copy_done[0], copy_stream);
            }
            
            for (int b_i = 0; b_i < num_batches; b_i++) {
                size_t i = b_i * batch_size;
                int current_batch_size = std::min((int)train_dataset.size() - (int)i, batch_size);
                size_t x_dim = train_dataset[0].image.numel();
                size_t y_dim = train_dataset[0].target.numel();
                
                vc::Tensor<float> Y_cuda;
                vc::Tensor<float> Y_batch_cuda;
                
                if (use_pipeline) {
                    int curr_buf = b_i % 2;
                    int next_buf = (b_i + 1) % 2;
                    
                    cudaStreamWaitEvent(0, copy_done[curr_buf], 0);
                    Y_cuda = next_X_cuda;
                    Y_batch_cuda = next_Y_cuda;
                    
                    if (b_i + 1 < num_batches) {
                        int next_batch_size = std::min((int)train_dataset.size() - (int)(i + batch_size), batch_size);
                        
                        vc::vector<size_t> next_x_shape(2); next_x_shape[0] = next_batch_size; next_x_shape[1] = x_dim;
                        next_X_cuda = vc::Tensor<float>::empty_gpu(next_x_shape);
                        
                        vc::vector<size_t> next_y_shape(2); next_y_shape[0] = next_batch_size; next_y_shape[1] = y_dim;
                        next_Y_cuda = vc::Tensor<float>::empty_gpu(next_y_shape);
                        
                        cudaMemcpyAsync(next_X_cuda.gpu_data->ptr, h_X_all + (i + batch_size) * x_dim, next_batch_size * x_dim * sizeof(float), cudaMemcpyHostToDevice, copy_stream);
                        cudaMemcpyAsync(next_Y_cuda.gpu_data->ptr, h_Y_all + (i + batch_size) * y_dim, next_batch_size * y_dim * sizeof(float), cudaMemcpyHostToDevice, copy_stream);
                        cudaEventRecord(copy_done[next_buf], copy_stream);
                    }
                } else {
                    vc::vector<size_t> x_shape(2); x_shape[0] = current_batch_size; x_shape[1] = x_dim;
                    next_X_cuda = vc::Tensor<float>::empty_gpu(x_shape);
                    cudaMemcpy(next_X_cuda.gpu_data->ptr, h_X_all + i * x_dim, current_batch_size * x_dim * sizeof(float), cudaMemcpyHostToDevice);
                    Y_cuda = next_X_cuda;
                    
                    vc::vector<size_t> y_shape(2); y_shape[0] = current_batch_size; y_shape[1] = y_dim;
                    next_Y_cuda = vc::Tensor<float>::empty_gpu(y_shape);
                    cudaMemcpy(next_Y_cuda.gpu_data->ptr, h_Y_all + i * y_dim, current_batch_size * y_dim * sizeof(float), cudaMemcpyHostToDevice);
                    Y_batch_cuda = next_Y_cuda;
                }
                
                vc::Tensor<float> Y_pred_gpu = (*this)(Y_cuda);
                
                if (use_pipeline) {
                    optimizer.zero_grad();
                    Y_pred_gpu.fast_cross_entropy_backward(Y_batch_cuda, d_epoch_loss, d_epoch_correct);
                    optimizer.step();
                } else {
                    float batch_loss = 0.0f;
                    int batch_correct = 0;
                    optimizer.zero_grad();
                    Y_pred_gpu.fast_cross_entropy_backward(Y_batch_cuda, &batch_loss, &batch_correct);
                    optimizer.step();
                    epoch_loss += batch_loss;
                    correct += batch_correct;
                }
            }
            
            if (use_pipeline) {
                // Synchronize ONCE per epoch!
                cudaMemcpy(&epoch_loss, d_epoch_loss, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&correct, d_epoch_correct, sizeof(int), cudaMemcpyDeviceToHost);
            }
            
            float avg_loss = epoch_loss / train_dataset.size();
            float accuracy = (float)correct / train_dataset.size() * 100.0f;
            
            std::cout << "Epoch " << epoch << "/" << epochs 
                      << " - Loss: " << avg_loss 
                      << " - Acc: " << accuracy << "%" << std::endl;
            
            history_loss.push_back(avg_loss);
            history_acc.push_back(accuracy);
        }
        
        if (use_pipeline) {
            cudaHostUnregister(h_X_all);
            cudaHostUnregister(h_Y_all);
            cudaFree(d_epoch_loss);
            cudaFree(d_epoch_correct);
            cudaStreamDestroy(copy_stream);
            cudaEventDestroy(copy_done[0]);
            cudaEventDestroy(copy_done[1]);
        }
        std::free(h_X_all);
        std::free(h_Y_all);
    }

    void save(const std::string& filepath) {
        // Bring weights back to CPU so we can read them safely
        this->to("cpu");
        
        std::ofstream weights_file(filepath);
        weights_file << std::setprecision(8);
        weights_file << "{\n";
        
        for (size_t l = 0; l < layers.size(); l++) {
            // Write weight tensor
            weights_file << "  \"w" << (l+1) << "\": {";
            weights_file << "\"shape\": [" << layers[l].weight._shape[0] << ", " << layers[l].weight._shape[1] << "], ";
            weights_file << "\"data\": [";
            for (size_t i = 0; i < layers[l].weight.numel(); i++) {
                weights_file << (*layers[l].weight.data)[i];
                if (i + 1 < layers[l].weight.numel()) weights_file << ",";
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