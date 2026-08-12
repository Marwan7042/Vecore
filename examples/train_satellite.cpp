#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "vecore/nn.h"
#include "vecore/optim.h"
#include "vecore/data.h"
#include "vecore/sequential.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <thread>
#include <chrono>
#include <map>
#include <algorithm>
#include <random>

using namespace vc;
namespace fs = std::filesystem;

// The Satellite CNN Architecture
Sample load_eurosat_image(const std::string& filepath, int label) {
    int width, height, channels;
    // EuroSAT has RGB JPEGs. Force 3 channels.
    unsigned char *img = stbi_load(filepath.c_str(), &width, &height, &channels, 3);
    if (img == nullptr) {
        throw std::runtime_error("Error loading image: " + filepath);
    }

    vc::vector<size_t> img_shape(3); 
    img_shape[0] = 3; img_shape[1] = 64; img_shape[2] = 64;
    Tensor<float> X(img_shape);
    
    // Convert HWC (stb_image) to CHW (our CNN expects Channels x Height x Width)
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 64; h++) {
            for (int w = 0; w < 64; w++) {
                int src_idx = (h * 64 + w) * 3 + c;
                int dst_idx = c * 64 * 64 + h * 64 + w;
                (*X.data)[dst_idx] = (float)img[src_idx] / 255.0f;
            }
        }
    }
    stbi_image_free(img);

    // One-hot encode label
    vc::vector<size_t> y_shape(2); y_shape[0] = 1; y_shape[1] = 10;
    Tensor<float> Y(y_shape);
    for (int i = 0; i < 10; i++) {
        (*Y.data)[i] = (i == label) ? 1.0f : 0.0f;
    }

    return {X, Y, filepath, label};
}

class EuroSATDataset : public vc::data::InMemoryDataset {
public:
    EuroSATDataset(const std::string& base_dir) {
        std::cout << "Scanning EuroSAT Directory..." << std::endl;
        
        std::vector<std::string> classes = {
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", 
            "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
        };
        
        std::vector<std::pair<std::string, int>> file_list;
        for (int label = 0; label < 10; label++) {
            std::string dir_path = base_dir + "/" + classes[label];
            if (fs::exists(dir_path)) {
                for (const auto & entry : fs::directory_iterator(dir_path)) {
                    if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
                        file_list.push_back({entry.path().string(), label});
                    }
                }
            }
        }
        
        // Shuffle to ensure randomized batches
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(file_list.begin(), file_list.end(), g);
        
        // Load the ENTIRE NASA dataset (27,000 images)
        int num_to_load = file_list.size();
        data.resize(num_to_load);
        
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        std::cout << "Loading " << num_to_load << " EuroSAT images using " << num_threads << " CPU cores..." << std::endl;
        
        std::vector<std::thread> threads;
        int chunk_size = num_to_load / num_threads;
        
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back([this, &file_list, t, num_threads, chunk_size, num_to_load]() {
                int start = t * chunk_size;
                int end = (t == num_threads - 1) ? num_to_load : (t + 1) * chunk_size;
                for (int i = start; i < end; i++) {
                    data[i] = load_eurosat_image(file_list[i].first, file_list[i].second);
                }
            });
        }
        
        for (auto& th : threads) { th.join(); }
        std::cout << "Successfully loaded " << data.size() << " 64x64 RGB Satellite Tensors!" << std::endl;
    }
};

int main() {
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    
    // Initialize Dataset
    EuroSATDataset dataset("/home/marwan/.dev/vecore/datasets/eurosat");
    std::vector<Sample>& raw_data = dataset.get_raw_data();
    if (raw_data.empty()) {
        std::cerr << "Dataset empty! Check path." << std::endl;
        return 1;
    }

    std::mt19937 g(42);
    std::shuffle(raw_data.begin(), raw_data.end(), g);
    
    size_t split1 = raw_data.size() * 0.8;
    size_t split2 = raw_data.size() * 0.9;
    std::vector<Sample> train_data(raw_data.begin(), raw_data.begin() + split1);
    std::vector<Sample> val_data(raw_data.begin() + split1, raw_data.begin() + split2);
    std::vector<Sample> test_data(raw_data.begin() + split2, raw_data.end());

    std::cout << "Train dataset size: " << train_data.size() << std::endl;
    std::cout << "Validation dataset size: " << val_data.size() << std::endl;
    std::cout << "Test dataset size: " << test_data.size() << std::endl;

    // Initialize Model Polymorphically
    Sequential model;
    model.add(std::make_shared<vc::nn::Conv2d<float>>(3, 16, 3, 1, 1));
    model.add(std::make_shared<vc::nn::ReLU<float>>());
    model.add(std::make_shared<vc::nn::MaxPool2d<float>>(2, 2));
    model.add(std::make_shared<vc::nn::Conv2d<float>>(16, 32, 3, 1, 1));
    model.add(std::make_shared<vc::nn::ReLU<float>>());
    model.add(std::make_shared<vc::nn::MaxPool2d<float>>(2, 2));
    model.add(std::make_shared<vc::nn::Conv2d<float>>(32, 64, 3, 1, 1));
    model.add(std::make_shared<vc::nn::ReLU<float>>());
    model.add(std::make_shared<vc::nn::MaxPool2d<float>>(2, 2));
    model.add(std::make_shared<vc::nn::Flatten<float>>());
    model.add(std::make_shared<vc::nn::Dense<float>>(4096, 128, "relu"));
    model.add(std::make_shared<vc::nn::Dense<float>>(128, 10, "none"));

    std::cout << "Transferring CNN Weights to GPU VRAM..." << std::endl;
    model.to("cuda");

    model.compile("adam");
    model.fit(train_data, 50, 0.001f, 64, 10, 1e-5f, &val_data);

    std::cout << "\nTraining finished successfully!" << std::endl;
    
    // Evaluate on the unseen test set
    model.evaluate(test_data, 64);
    
    model.save("satellite_model_weights.bin");
    return 0;
}
