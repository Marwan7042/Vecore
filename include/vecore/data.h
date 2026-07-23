#pragma once
#include "vecore/Tensor.h"
#include <vector>
#include <string>

// We moved Sample here from sequential.h to serve as the standard data format
struct Sample {
    vc::Tensor<float> image; 
    vc::Tensor<float> target;
    std::string       filepath;
    int               label;
};

namespace vc {
    namespace data {

        // Abstract Base Class for ALL future datasets (Text, Audio, Vision)
        class Dataset {
        public:
            virtual size_t len() = 0;
            virtual Sample get_item(size_t index) = 0;
            virtual ~Dataset() = default;
        };

        // A wrapper that holds an entire dataset in memory
        class InMemoryDataset : public Dataset {
        protected:
            std::vector<Sample> data;
        public:
            size_t len() override { return data.size(); }
            Sample get_item(size_t index) override { return data[index]; }
            std::vector<Sample>& get_raw_data() { return data; }
        };

    }
}
