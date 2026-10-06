#include <cuda_runtime.h>
#include <thrust/sort.h>

__global__ void hashPositions(int64_t* keys_out, uint32_t* indices_out, uint32_t hashmap_size, float4* positions, uint32_t body_count, float smoothing_radius) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= body_count) return;

    float4 position = positions[idx];
    float sm_inv = 1.0f / smoothing_radius;

    int32_t posx = floorf(position.x * sm_inv);
    int32_t posy = floorf(position.y * sm_inv);
    int32_t posz = floorf(position.z * sm_inv);

    int64_t hash = ((posx * 73856093) ^ (posy * 19349663) ^ (posz * 83492791)) % hashmap_size;

    keys_out[idx] = hash;
    indices_out[idx] = idx;
}