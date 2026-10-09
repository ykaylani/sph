#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t hashCell(int32_t x, int32_t y, int32_t z, uint32_t hashmap_size) {
    uint32_t ux = static_cast<uint32_t>(x);
    uint32_t uy = static_cast<uint32_t>(y);
    uint32_t uz = static_cast<uint32_t>(z);

    return ((ux * 73856093u) ^ (uy * 19349663u) ^ (uz * 83492791u)) % hashmap_size;
}


__global__ void hashPositions(int64_t* keys_out, uint32_t* indices_out, uint32_t hashmap_size, float4* positions, uint32_t particle_count, float inverse_smoothing) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count) return;

    float4 position = positions[idx];

    int32_t posx = floorf(position.x * inverse_smoothing);
    int32_t posy = floorf(position.y * inverse_smoothing);
    int32_t posz = floorf(position.z * inverse_smoothing);

    keys_out[idx] = hashCell(posx, posy, posz, hashmap_size);
    indices_out[idx] = idx;
}