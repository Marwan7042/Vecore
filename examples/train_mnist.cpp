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
#include <thread>
#include <chrono>
#include "vecore/sequential.h"

using namespace vc;
namespace fs = std::filesystem;

// A real Computer Vision MLP using our new Sequential API!
Sequential build_model() {
    return Sequential({
        vc::nn::Dense<float>(784, 2048, "relu"),
        vc::nn::Dense<float>(2048, 1024, "relu"),
        vc::nn::Dense<float>(1024, 10, "none")
    });
}

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

class MNISTDataset : public vc::data::InMemoryDataset {
public:
    MNISTDataset(const std::string& base_dir) {
        std::cout << "Scanning MNIST PNG Directory..." << std::endl;
        
        std::vector<std::pair<std::string, int>> file_list;
        for (int label = 0; label < 10; label++) {
            std::string dir_path = base_dir + std::to_string(label);
            if (fs::exists(dir_path)) {
                for (const auto & entry : fs::directory_iterator(dir_path)) {
                    file_list.push_back({entry.path().string(), label});
                }
            }
        }
        
        data.resize(file_list.size());
        
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        std::cout << "Loading " << file_list.size() << " images using " << num_threads << " CPU cores..." << std::endl;
        
        std::vector<std::thread> threads;
        int chunk_size = file_list.size() / num_threads;
        
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back([this, &file_list, t, num_threads, chunk_size]() {
                int start = t * chunk_size;
                int end = (t == num_threads - 1) ? file_list.size() : (t + 1) * chunk_size;
                for (int i = start; i < end; i++) {
                    data[i] = load_image(file_list[i].first, file_list[i].second);
                }
            });
        }
        
        for (auto& th : threads) {
            th.join();
        }
        std::cout << "Done! Loaded " << data.size() << " actual PNG images into Tensors!" << std::endl;
    }
};

int main() {
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    
    MNISTDataset dataset(std::string(PROJECT_ROOT_DIR) + "/datasets/mnist_png/training/");
    std::cout << "Building Deep MLP (784 -> 2048 -> 1024 -> 10) to learn Computer Vision..." << std::endl;
    
    Sequential model = build_model();
    
    // Shoot the Neural Network weights across the PCIe bus into the RTX 5050's VRAM!
    std::cout << "Transferring Weights to RTX 5050 (VRAM)..." << std::endl;
    model.to("cuda");

    std::cout << "\nTraining..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    model.fit(dataset, 150, 0.1f, 8192);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> training_duration = end_time - start_time;
    std::cout << "\n>>> TOTAL TRAINING TIME: " << training_duration.count() << " seconds <<<" << std::endl;
    
    // ── Evaluate on Test Set ──────────────────────────────────────────────────
    std::cout << "\nEvaluating on Unseen Test Images..." << std::endl;
    MNISTDataset test_dataset_obj(std::string(PROJECT_ROOT_DIR) + "/datasets/mnist_png/testing/");
    std::vector<Sample>& test_dataset = test_dataset_obj.get_raw_data();
    
    int test_correct = 0;
    int batch_size = 32;
    for (size_t i = 0; i < test_dataset.size(); i += batch_size) {
        int current_batch_size = std::min((int)test_dataset.size() - (int)i, batch_size);
        
        vector<size_t> x_shape(2);
        x_shape[0] = current_batch_size;
        x_shape[1] = 784;
        Tensor<float> X_batch(x_shape);
        
        for (int b = 0; b < current_batch_size; b++) {
            for (int j = 0; j < 784; j++) (*X_batch.data)[b * 784 + j] = (*test_dataset[i + b].image.data)[j];
        }
        
        Tensor<float> X_cuda = X_batch.to("cuda");
        Tensor<float> Y_pred_gpu = model(X_cuda);
        Tensor<float> Y_pred = Y_pred_gpu.to("cpu");
        
        for (int b = 0; b < current_batch_size; b++) {
            int max_idx = 0; float max_val = -9999.0f; int target_idx = 0;
            for (int j = 0; j < 10; j++) {
                if ((*Y_pred.data)[b * 10 + j] > max_val) { max_val = (*Y_pred.data)[b * 10 + j]; max_idx = j; }
                if ((*test_dataset[i + b].target.data)[j] == 1.0f) { target_idx = j; }
            }
            if (max_idx == target_idx) test_correct++;
        }
    }
    
    float test_accuracy = (float)test_correct / test_dataset.size() * 100.0f;
    std::cout << "=====================================================" << std::endl;
    std::cout << " REAL-WORLD TEST ACCURACY: " << test_accuracy << "% (" << test_correct << "/" << test_dataset.size() << ")" << std::endl;
    std::cout << "=====================================================" << std::endl;
    
    // ── Generating results.json is skipped as inference happens in the browser ──
    
    // ── Export Trained Weights for Browser Inference ──────────────────────────
    std::cout << "\nExporting trained weights for interactive web demo..." << std::endl;
    model.save(std::string(PROJECT_ROOT_DIR) + "/docs/weights.json");
    
    return 0;
}
