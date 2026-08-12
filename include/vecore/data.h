#pragma once
#include "vecore/Tensor.h"
#include <vector>
#include <string>

/// Standard sample container for image, target, and metadata fields.
struct Sample {
    vc::Tensor<float> image; 
    vc::Tensor<float> target;
    std::string       filepath;
    int               label;
};

namespace vc {
    namespace data {

        /// Abstract base class for all dataset types.
        class Dataset {
        public:
            /// Returns the number of items in the dataset.
            virtual size_t len() = 0;
            /// Returns the sample at the given index.
            virtual Sample get_item(size_t index) = 0;
            /// Releases the dataset resources.
            virtual ~Dataset() = default;
        };

        /// Dataset implementation that keeps all samples in memory.
        class InMemoryDataset : public Dataset {
        protected:
            std::vector<Sample> data;
        public:
            /// Returns the number of stored samples.
            size_t len() override { return data.size(); }
            /// Returns the stored sample at the requested index.
            Sample get_item(size_t index) override { return data[index]; }
            /// Returns mutable access to the backing sample buffer.
            std::vector<Sample>& get_raw_data() { return data; }
        };

    }
}
