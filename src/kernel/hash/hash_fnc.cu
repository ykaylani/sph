#include <cuda_runtime.h>
#include "hash.cuh"

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