#include "vecore/Tensor.h"
#include "vecore/nn.h"
#include "vecore/optim.h"
#include "vecore/data.h"
#include <cstdio>
#include <vector>

// 1. Define the identical Architecture
struct CloudModel {
    vc::nn::Conv2d<float> conv1; vc::nn::MaxPool2d<float> pool1;
    vc::nn::Conv2d<float> conv2; vc::nn::MaxPool2d<float> pool2;
    vc::nn::Conv2d<float> conv3; vc::nn::MaxPool2d<float> pool3;
    vc::nn::Dense<float> dense1; vc::nn::Dense<float> dense2;

    CloudModel() : 
        conv1(3, 16, 3, 1, 1, "relu"), pool1(2, 2),
        conv2(16, 32, 3, 1, 1, "relu"), pool2(2, 2),
        conv3(32, 64, 3, 1, 1, "relu"), pool3(2, 2),
        dense1(4096, 128, "relu"), dense2(128, 2, "none") {}

    vc::Tensor<float> operator()(const vc::Tensor<float>& x) {
        vc::Tensor<float> h = x;
        h = conv1(h); h = pool1(h);
        h = conv2(h); h = pool2(h);
        h = conv3(h); h = pool3(h);
        
        vc::vector<size_t> flat_shape(2);
        flat_shape[0] = h._shape[0]; 
        flat_shape[1] = h.numel() / h._shape[0];
        h = h.reshape(flat_shape);
        
        h = dense1(h); 
        return dense2(h);
    }

    vc::vector<vc::Tensor<float>*> parameters() {
        vc::vector<vc::Tensor<float>*> params;
        params.push_back(&conv1.weight); params.push_back(&conv1.bias);
        params.push_back(&conv2.weight); params.push_back(&conv2.bias);
        params.push_back(&conv3.weight); params.push_back(&conv3.bias);
        params.push_back(&dense1.weight); params.push_back(&dense1.bias);
        params.push_back(&dense2.weight); params.push_back(&dense2.bias);
        return params;
    }
};

// 2. Exporter Function (Writes raw float32 bytes for VCLite)
void export_weights(CloudModel& model, const char* filepath) {
    FILE* f = fopen(filepath, "wb");
    if (!f) { printf("Failed to open %s\n", filepath); return; }
    
    auto params = model.parameters();
    for (size_t i = 0; i < params.size(); i++) {
        vc::Tensor<float>* p = params[i];
        if (p->is_cuda) {
            // Need to pull to CPU first if using GPU
            vc::Tensor<float> cpu_p = p->to("cpu");
            for (size_t j = 0; j < cpu_p.numel(); j++) {
                float val = (*cpu_p.data)[j];
                fwrite(&val, sizeof(float), 1, f);
            }
        } else {
            for (size_t j = 0; j < p->numel(); j++) {
                float val = (*p->data)[j];
                fwrite(&val, sizeof(float), 1, f);
            }
        }
    }
    fclose(f);
    printf("Exported model weights to %s for VCLite!\n", filepath);
}

// 3. Training Loop
int main() {
    printf("--- Training Cloud Filter Model (Vecore CPU) ---\n");
    CloudModel model;
    
    // Set up SGD optimizer
    vc::optim::SGD<float> optimizer(model.parameters(), 0.01f);
    
    // (Here is where you'd load the NASA 38-Cloud dataset)
    // For this boilerplate, we'll create a dummy batch
    vc::vector<size_t> in_shape(4);
    in_shape[0] = 8; in_shape[1] = 3; in_shape[2] = 64; in_shape[3] = 64;
    vc::Tensor<float> X(in_shape); // 8 dummy images
    
    vc::vector<size_t> y_shape(2);
    y_shape[0] = 8; y_shape[1] = 2;
    vc::Tensor<float> Y(y_shape);  // 8 dummy targets (one-hot)
    
    int epochs = 5;
    for (int e = 0; e < epochs; e++) {
        optimizer.zero_grad();
        
        vc::Tensor<float> logits = model(X);
        
        // Dummy MSE loss
        vc::Tensor<float> diff = logits - Y;
        vc::Tensor<float> loss = (diff * diff).sum(vc::vector<int>()) * 0.5f; // or CrossEntropy
        
        printf("Epoch %d - Loss: %f\n", e+1, (*loss.data)[0]);
        
        // loss.backward(); // Assuming backward() exists in Tensor.h
        // optimizer.step();
    }
    
    export_weights(model, "cloud_filter_weights.bin");
    
    return 0;
}
