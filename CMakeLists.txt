cmake_minimum_required(VERSION 4.2)
project(sph CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(
        sph src/main.cu
        src/kernel/hash/hash_fnc.cu
        src/kernel/hash/hash_ranges.cu
        src/kernel/pressure/density.cu
        src/kernel/pressure/pressure.cu
        src/kernel/pressure/pressure_acceleration.cu
        src/kernel/integrate/verlet.cu
        src/kernel/external_forces/gravity.cu
        src/kernel/external_forces/gravity.cuh
)

set_target_properties(sph PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(sph PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/Zc:preprocessor>
)
