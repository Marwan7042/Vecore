cmake_minimum_required(VERSION 4.2.3)

add_executable(trainmlp train_mlp.cpp)
add_executable(trainmnist train_mnist.cpp)
add_executable(train train.cpp)
add_executable(trainsatellite train_satellite.cpp)

target_include_directories(trainmnist PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
target_compile_definitions(trainmnist PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

target_link_libraries(trainmlp PRIVATE vecore::vecore)
target_link_libraries(trainmnist PRIVATE vecore::vecore)
target_link_libraries(train PRIVATE vecore::vecore)
target_link_libraries(trainsatellite PRIVATE vecore::vecore)


add_executable(train_cloud_filter train_cloud_filter.cpp)
target_link_libraries(train_cloud_filter vecore)

add_executable(train_dark_vessel train_dark_vessel.cpp)
target_link_libraries(train_dark_vessel vecore)

add_executable(train_wildfire train_wildfire.cpp)
target_link_libraries(train_wildfire vecore)

add_executable(train_deforestation train_deforestation.cpp)
target_link_libraries(train_deforestation vecore)

add_executable(train_methane train_methane.cpp)
target_link_libraries(train_methane vecore)
