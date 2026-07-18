cmake_minimum_required(VERSION 4.2.3)

add_library(micrograd
    Tensor.h
    nn.h
    optim.h
)

add_library(micrograd::micrograd ALIAS micrograd)

target_link_libraries(PUBLIC libvc)

target_include_directories(micrograd PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
