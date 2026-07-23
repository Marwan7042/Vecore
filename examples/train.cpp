#include "vecore/nn.h"
#include <iostream>

int main() {
    std::cout << "Starting Neural Network Training..." << std::endl;
    std::cout << "Goal: Learn the function Y = 2X + 3\n" << std::endl;
    
    // 1. Create our Neural Network (1 input feature, 1 output feature)
    vc::nn::Dense<float> layer(1, 1);
    
    // 2. Training Data
    float X_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float Y_data[4] = {5.0f, 7.0f, 9.0f, 11.0f}; // 2x + 3
    
    vc::vector<size_t> shape(2);
    shape[0] = 1; shape[1] = 1;
    
    vc::Tensor<float> X(shape);
    vc::Tensor<float> Y_true(shape);
    
    vc::vector<size_t> idx(2);
    idx[0] = 0; idx[1] = 0;

    float learning_rate = 0.001f;

    for (int epoch = 0; epoch <= 1000; epoch++) {
        float epoch_loss = 0.0f;
        
        // We use a batch size of 1 since we haven't implemented broadcasting yet!
        for (int i = 0; i < 4; i++) {
            // Load one batch
            X(idx) = X_data[i];
            Y_true(idx) = Y_data[i];
            
            // --- FORWARD PASS ---
            vc::Tensor<float> Y_pred = layer(X);
            
            // Calculate Error and Loss (Error * Error.T() gives us Sum of Squared Errors)
            vc::Tensor<float> Error = Y_pred - Y_true;
            vc::Tensor<float> Loss = Error * Error.transpose();
            
            epoch_loss += Loss(idx);
            
            // --- BACKWARD PASS ---
            Loss.backward();
            
            // --- OPTIMIZER STEP (SGD) ---
            vc::vector<vc::Tensor<float>*> params = layer.parameters();
            for (size_t p = 0; p < params.size(); p++) {
                vc::Tensor<float>* param = params[p];
                for (size_t j = 0; j < param->numel(); j++) {
                    // Update weights: W = W - lr * dL/dW
                    (*param->data)[j] -= learning_rate * (*param->ctx->grad->data)[j];
                    
                    // Zero Gradients for the next step! (Just like PyTorch optimizer.zero_grad())
                    (*param->ctx->grad->data)[j] = 0.0f;
                }
            }
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss / 4.0f << std::endl;
        }
    }
    
    std::cout << "\nTraining Finished!" << std::endl;
    std::cout << "Learned Weight (Expected ~2.0): " << (*layer.weight.data)[0] << std::endl;
    std::cout << "Learned Bias (Expected ~3.0): " << (*layer.bias.data)[0] << std::endl;
    
    return 0;
}
