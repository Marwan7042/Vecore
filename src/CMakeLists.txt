cmake_minimum_required(VERSION 4.2.3)

add_library(micrograd Tensor.cpp)
add_library(micrograd::micrograd ALIAS micrograd)

target_include_directories(micrograd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

set_source_files_properties(Tensor.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(micrograd PUBLIC libvc::libvc BLAS::BLAS CUDA::cublas CUDA::cudart)
