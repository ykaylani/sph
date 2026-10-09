#include <cuda_runtime.h>
#include <cstdint>

__global__ void pressure(
    float* pressure,
    float* density,
    uint32_t particle_count,
    float polytropic,
    float polytropic_inv,
    float rest_density,
    float rest_density_inv,
    float sound_speed) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count) return;

    pressure[idx] = rest_density * sound_speed * sound_speed * polytropic_inv * (powf(density[idx] * rest_density_inv, polytropic) - 1.0f);
}