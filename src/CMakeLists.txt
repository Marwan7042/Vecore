cmake_minimum_required(VERSION 4.2.3)

add_library(vecore 
    Tensor.cu
    ops/Math.cu
    ops/Activations.cu
    autograd/Engine.cu
    optim/Optim.cu
    nn/Conv2d.cu
    nn/MaxPool2d.cu
)
add_library(vecore::vecore ALIAS vecore)

target_include_directories(vecore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

target_link_libraries(vecore PUBLIC libvc::libvc BLAS::BLAS CUDA::cublas CUDA::cudart)
