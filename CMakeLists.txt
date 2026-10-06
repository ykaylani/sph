cmake_minimum_required(VERSION 4.2)
project(sph CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(
        sph src/main.cu
        src/hash/hash.cu

)

set_target_properties(sph PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
