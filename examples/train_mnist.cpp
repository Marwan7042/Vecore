#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "vecore/nn.h"
#include "vecore/optim.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <cmath>

using namespace vc;
namespace fs = std::filesystem;

// A real Computer Vision MLP!
 class MNISTModel {                                                       
    public:                                                                                                                                 
        nn::Dense<float> layer1;
        nn::Dense<float> layer2;
        nn::Dense<float> layer3;

        MNISTModel() : 
            layer1(784, 2048), 
            layer2(2048, 1024),
            layer3(1024, 10)
        {}

        Tensor<float> operator()(const Tensor<float>& x) {
            Tensor<float> h1 = layer1(x).relu().to("cuda"); 
            Tensor<float> h2 = layer2(h1).relu().to("cuda"); 
            return layer3(h2);
        }

        vector<Tensor<float>*> parameters() {
            vector<Tensor<float>*> params;
            auto p1 = layer1.parameters();
            auto p2 = layer2.parameters();
            auto p3 = layer3.parameters();
  
            params.push_back(p1[0]); params.push_back(p1[1]);
            params.push_back(p2[0]); params.push_back(p2[1]);
            params.push_back(p3[0]); params.push_back(p3[1]);

            return params;
        }
    };


struct Sample {
    Tensor<float> image;
    Tensor<float> target;
    std::string   filepath;
    int           label;
};

std::vector<float> softmax_vec(const Tensor<float>& logits) {
    float max_val = (*logits.data)[0];
    for (int i = 1; i < 10; i++) if ((*logits.data)[i] > max_val) max_val = (*logits.data)[i];
    std::vector<float> probs(10);
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) { probs[i] = std::exp((*logits.data)[i] - max_val); sum += probs[i]; }
    for (int i = 0; i < 10; i++) probs[i] /= sum;
    return probs;
}

// Function to read an actual .png file and turn it into a Tensor!
Sample load_image(const std::string& filepath, int label) {
    int width, height, channels;
    // Read the PNG pixels!
    unsigned char *img = stbi_load(filepath.c_str(), &width, &height, &channels, 1);
    if (img == nullptr) {
        throw std::runtime_error("Error in loading the image\n");
    }

    vector<size_t> img_shape(2); img_shape[0] = 1; img_shape[1] = 784;
    Tensor<float> X(img_shape);
    for (int i = 0; i < 784; i++) {
        // Normalize pixel values from [0, 255] to [0.0, 1.0] for the neural network
        X.data->operator[](i) = (float)img[i] / 255.0f;
    }
    stbi_image_free(img);

    // One-hot encode the target label (e.g., if label is 3, make the 3rd index 1.0 and rest 0.0)
    vector<size_t> y_shape(2); y_shape[0] = 1; y_shape[1] = 10;
    Tensor<float> Y(y_shape);
    for (int i = 0; i < 10; i++) {
        Y.data->operator[](i) = (i == label) ? 1.0f : 0.0f;
    }

    return {X, Y, filepath, label};
}

int main() {
    std::cout << "Scanning MNIST PNG Directory..." << std::endl;
    
    std::vector<Sample> dataset; // (Using standard vector to hold the dataset easily)
    std::string base_dir = std::string(PROJECT_ROOT_DIR) + "/datasets/mnist_png/training/"; // Path to the images we downloaded
    
    // Load 100 images of each digit (1000 total) to train efficiently with batch size 1
    for (int label = 0; label < 10; label++) {
        std::string dir_path = base_dir + std::to_string(label);
        int count = 0;
        for (const auto & entry : fs::directory_iterator(dir_path)) {
            if (count >= 100) break;
            dataset.push_back(load_image(entry.path().string(), label));
            count++;
        }
    }
    
    std::cout << "Loaded " << dataset.size() << " actual PNG images into Tensors!" << std::endl;
    std::cout << "Building Deep MLP (784 -> 512 -> 256 -> 128 -> 64 -> 10) to learn Computer Vision..." << std::endl;
    
    MNISTModel model;
    
    // Shoot the Neural Network weights across the PCIe bus into the RTX 5050's VRAM!
    std::cout << "Transferring Weights to RTX 5050 (VRAM)..." << std::endl;
    model.layer1.weight = model.layer1.weight.to("cuda");
    model.layer2.weight = model.layer2.weight.to("cuda");
    model.layer3.weight = model.layer3.weight.to("cuda");

    
    float learning_rate = 0.1f;
    optim::SGD<float> optimizer(model.parameters(), learning_rate);
    nn::CrossEntropyLoss<float> criterion;

    std::vector<float> history_loss, history_acc;
    // int patience = 3;
    // int count_perfect = 0;
    for (int epoch = 0; epoch <= 150; epoch++) {
        float epoch_loss = 0.0f;
        int correct = 0;
        
        for (size_t i = 0; i < dataset.size(); i++) {
            // --- 1. Forward Pass ---
            // Send the image to the GPU!
            Tensor<float> X_cuda = dataset[i].image.to("cuda");
            Tensor<float> Y_pred = model(X_cuda);
            
            // --- 2. Calculate Loss ---
            Tensor<float> Loss = criterion(Y_pred, dataset[i].target);
            
            // Check if the network guessed right!
            int max_idx = 0; float max_val = -9999.0f; int target_idx = 0;
            for (int j = 0; j < 10; j++) {
                if ((*Y_pred.data)[j] > max_val) { max_val = (*Y_pred.data)[j]; max_idx = j; }
                if ((*dataset[i].target.data)[j] == 1.0f) { target_idx = j; }
            }
            if (max_idx == target_idx) correct++;
            
            vector<size_t> idx00(2); idx00[0] = 0; idx00[1] = 0;
            epoch_loss += Loss(idx00);
            
            // --- 3. Zero Gradients ---
            optimizer.zero_grad();
            
            // --- 4. Backward Pass ---
            Loss.backward();
            
            // --- 5. Optimizer Step ---
            optimizer.step();
        }
        
        float avg_loss = epoch_loss / dataset.size();
        float accuracy = (float)correct / dataset.size() * 100.0f;
        // if (accuracy == 100) count_perfect++;
        history_loss.push_back(avg_loss);
        history_acc.push_back(accuracy);
        
        std::cout << "Epoch " << epoch << " | Loss: " << avg_loss 
        << " | Accuracy: " << accuracy << "%" << std::endl;
        // if (count_perfect >= patience) break;
    }
    
    // ── Evaluate on Test Set ──────────────────────────────────────────────────
    std::cout << "\nEvaluating on Unseen Test Images..." << std::endl;
    std::vector<Sample> test_dataset;
    std::string test_dir = std::string(PROJECT_ROOT_DIR) + "/datasets/mnist_png/testing/";
    
    // Load ALL 10,000 unseen images
    for (int label = 0; label < 10; label++) {
        std::string dir_path = test_dir + std::to_string(label);
        for (const auto & entry : fs::directory_iterator(dir_path)) {
            test_dataset.push_back(load_image(entry.path().string(), label));
        }
    }
    
    int test_correct = 0;
    for (size_t i = 0; i < test_dataset.size(); i++) {
        Tensor<float> X_cuda = test_dataset[i].image.to("cuda");
        Tensor<float> Y_pred = model(X_cuda);
        
        int max_idx = 0; float max_val = -9999.0f; int target_idx = 0;
        for (int j = 0; j < 10; j++) {
            if ((*Y_pred.data)[j] > max_val) { max_val = (*Y_pred.data)[j]; max_idx = j; }
            if ((*test_dataset[i].target.data)[j] == 1.0f) { target_idx = j; }
        }
        if (max_idx == target_idx) test_correct++;
    }
    
    float test_accuracy = (float)test_correct / test_dataset.size() * 100.0f;
    std::cout << "=====================================================" << std::endl;
    std::cout << " REAL-WORLD TEST ACCURACY: " << test_accuracy << "% (" << test_correct << "/" << test_dataset.size() << ")" << std::endl;
    std::cout << "=====================================================" << std::endl;
    
    // ── Generate results.json ─────────────────────────────────────────────────
    std::cout << "\nGenerating results.json for the demo..." << std::endl;
    fs::create_directories(std::string(PROJECT_ROOT_DIR) + "/docs");
    
    std::ofstream json_file(std::string(PROJECT_ROOT_DIR) + "/docs/results.json");
    json_file << "{\n";
    
    json_file << "  \"history\": {\n    \"loss\": [";
    for (size_t i = 0; i < history_loss.size(); i++)
        json_file << history_loss[i] << (i+1 < history_loss.size() ? "," : "");
    json_file << "],\n    \"accuracy\": [";
    for (size_t i = 0; i < history_acc.size(); i++)
        json_file << history_acc[i] << (i+1 < history_acc.size() ? "," : "");
    json_file << "]\n  },\n";
    
    json_file << "  \"samples\": [\n";
    for (size_t i = 0; i < dataset.size(); i++) {
        Tensor<float> X_cuda = dataset[i].image.to("cuda");
        Tensor<float> Y_pred = model(X_cuda);
        auto probs = softmax_vec(Y_pred);
        
        int pred_label = 0; float max_p = probs[0];
        for (int j = 1; j < 10; j++) if (probs[j] > max_p) { max_p = probs[j]; pred_label = j; }
        
        json_file << "    {";
        json_file << "\"path\":\"" << dataset[i].filepath << "\",";
        json_file << "\"label\":" << dataset[i].label << ",";
        json_file << "\"prediction\":" << pred_label << ",";
        json_file << "\"confidence\":" << max_p << ",";
        json_file << "\"probs\":[";
        for (int j = 0; j < 10; j++)
            json_file << probs[j] << (j < 9 ? "," : "");
        json_file << "]}";
        if (i + 1 < dataset.size()) json_file << ",";
        json_file << "\n";
    }
    json_file << "  ]\n}\n";
    json_file.close();
    
    std::cout << "Done! Open docs/index.html to view the results." << std::endl;
    
    // ── Export Trained Weights for Browser Inference ──────────────────────────
    std::cout << "\nExporting trained weights for interactive web demo..." << std::endl;
    
    // Copy weights back from GPU to CPU before exporting
    model.layer1.weight = model.layer1.weight.to("cpu");
    model.layer2.weight = model.layer2.weight.to("cpu");
    model.layer3.weight = model.layer3.weight.to("cpu");
    
    std::ofstream weights_file(std::string(PROJECT_ROOT_DIR) + "/docs/weights.json");
    weights_file << std::setprecision(8);
    weights_file << "{\n";
    
    // Helper lambda to write a tensor as a flat JSON array
    auto write_tensor = [&](const std::string& name, const Tensor<float>& t, bool last = false) {
        weights_file << "  \"" << name << "\": {";
        weights_file << "\"shape\": [" << t._shape[0] << ", " << t._shape[1] << "], ";
        weights_file << "\"data\": [";
        for (size_t i = 0; i < t.data->size(); i++) {
            weights_file << (*t.data)[i];
            if (i + 1 < t.data->size()) weights_file << ",";
        }
        weights_file << "]}";
        if (!last) weights_file << ",";
        weights_file << "\n";
    };
    
    write_tensor("w1", model.layer1.weight);
    write_tensor("b1", model.layer1.bias);
    write_tensor("w2", model.layer2.weight);
    write_tensor("b2", model.layer2.bias);
    write_tensor("w3", model.layer3.weight);
    write_tensor("b3", model.layer3.bias, true);
    
    weights_file << "}\n";
    weights_file.close();
    std::cout << "Weights exported to docs/weights.json!" << std::endl;
    
    return 0;
}
