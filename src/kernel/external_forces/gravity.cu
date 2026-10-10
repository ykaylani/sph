#include <cuda_runtime.h>
#include <cstdint>

__global__ void gravity(float4* acceleration, uint32_t particle_count) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= particle_count) return;
    acceleration[idx].y += 9.81;
}