cmake_minimum_required(VERSION 3.24)
project(kernel_image_processing CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_executable(kernel_image_processing
        CSV.cpp CSV.h main.cu png.cpp png.h)

set_target_properties(kernel_image_processing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(kernel_image_processing png)