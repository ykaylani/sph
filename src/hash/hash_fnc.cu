#include <cuda_runtime.h>

__global__ void hashPositions(int64_t* keys_out, uint32_t* indices_out, uint32_t hashmap_size, float4* positions, uint32_t body_count, float smoothing_radius) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= body_count) return;

    float4 position = positions[idx];
    float sm_inv = 1.0f / smoothing_radius;

    int32_t posx = floorf(position.x * sm_inv);
    int32_t posy = floorf(position.y * sm_inv);
    int32_t posz = floorf(position.z * sm_inv);

    uint32_t u_posx = static_cast<uint32_t>(posx);
    uint32_t u_posy = static_cast<uint32_t>(posy);
    uint32_t u_posz = static_cast<uint32_t>(posz);

    int64_t hash = ((u_posx * 73856093) ^ (u_posy * 19349663) ^ (u_posz * 83492791)) % hashmap_size;

    keys_out[idx] = hash;
    indices_out[idx] = idx;
}