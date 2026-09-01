#include "vecore/Tensor.h"
#include "vecore/nn.h"
#include "vecore/optim.h"
#include <cstdio>
#include <vector>

struct DeforestationModel {
    vc::nn::Conv2d<float> conv1; vc::nn::MaxPool2d<float> pool1;
    vc::nn::Conv2d<float> conv2; vc::nn::MaxPool2d<float> pool2;
    vc::nn::Conv2d<float> conv3; vc::nn::MaxPool2d<float> pool3;
    vc::nn::Dense<float> dense1; vc::nn::Dense<float> dense2;

    DeforestationModel() : 
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
        flat_shape[0] = h._shape[0]; flat_shape[1] = h.numel() / h._shape[0];
        h = h.reshape(flat_shape);
        h = dense1(h); return dense2(h);
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

void export_weights(DeforestationModel& model, const char* filepath) {
    FILE* f = fopen(filepath, "wb");
    if (!f) return;
    auto params = model.parameters();
    for (size_t i = 0; i < params.size(); i++) {
        vc::Tensor<float>* p = params[i];
        for (size_t j = 0; j < p->numel(); j++) {
            float val = (*p->data)[j];
            fwrite(&val, sizeof(float), 1, f);
        }
    }
    fclose(f);
    printf("Exported model weights to %s for VCLite!\n", filepath);
}

int main() {
    printf("--- Training Deforestation Model (Vecore CPU) ---\n");
    DeforestationModel model;
    vc::optim::Adam<float> optimizer(model.parameters(), 0.001f);
    
    vc::vector<size_t> in_shape(4);
    in_shape[0] = 4; in_shape[1] = 3; in_shape[2] = 64; in_shape[3] = 64;
    vc::Tensor<float> X(in_shape);
    
    vc::vector<size_t> y_shape(2);
    y_shape[0] = 4; y_shape[1] = 2;
    vc::Tensor<float> Y(y_shape);
    
    for (int e = 0; e < 5; e++) {
        optimizer.zero_grad();
        vc::Tensor<float> logits = model(X);
        vc::Tensor<float> loss = ((logits - Y) * (logits - Y)).sum(vc::vector<int>()) * 0.5f;
        printf("Epoch %d - Loss: %f\n", e+1, (*loss.data)[0]);
    }
    
    export_weights(model, "deforestation_weights.bin");
    return 0;
}
