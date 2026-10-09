#include <cuda_runtime.h>
#include <cstdint>

#include "../hash/hash.cuh"

__global__ void pressureAcceleration(
    float* densities,
    float* pressures,
    int32_t* cell_starts,
    int32_t* cell_ends,
    uint32_t* indices,
    uint32_t hashmap_size,
    float inverse_smoothing,
    float inverse_smoothing_sqr,
    float smoothing_norm,
    float4* particle_accelerations,
    float4* particle_positions,
    uint32_t particle_count) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count) return;

    float4 selected_position = particle_positions[idx];

    int32_t posx = floorf(selected_position.x * inverse_smoothing);
    int32_t posy = floorf(selected_position.y * inverse_smoothing);
    int32_t posz = floorf(selected_position.z * inverse_smoothing);

    float selected_density = densities[idx];
    float4 acceleration = {0, 0, 0, 0};

    for (int32_t dz = -1; dz <= 1; ++dz) {
        for (int32_t dy = -1; dy <= 1; ++dy) {
            for (int32_t dx = -1; dx <= 1; ++dx) {
                uint32_t bucket = hashCell(posx + dx, posy + dy, posz + dz, hashmap_size);

                int32_t start = cell_starts[bucket];
                if (start == -1) continue;

                int32_t end = cell_ends[bucket];

                for (int32_t slot = start; slot < end; ++slot) {
                    uint32_t neighbor_idx = indices[slot];
                    float4 neighbor_position = particle_positions[neighbor_idx];
                    float neighbor_mass = neighbor_position.w;

                    int32_t neighbor_x = floorf(neighbor_position.x * inverse_smoothing);
                    int32_t neighbor_y = floorf(neighbor_position.y * inverse_smoothing);
                    int32_t neighbor_z = floorf(neighbor_position.z * inverse_smoothing);

                    if (neighbor_x != posx + dx || neighbor_y != posy + dy || neighbor_z != posz + dz) continue;

                    float dist_sqr =
                        (neighbor_position.x - selected_position.x) * (neighbor_position.x - selected_position.x) +
                        (neighbor_position.y - selected_position.y) * (neighbor_position.y - selected_position.y) +
                        (neighbor_position.z - selected_position.z) * (neighbor_position.z - selected_position.z);


                    float op = 1.0f - dist_sqr * inverse_smoothing_sqr;
                    if (op <= 0.0f) continue;

                    float gradient_scale = -6.0f * smoothing_norm * inverse_smoothing_sqr * op * op;

                    float3 gradW = {
                        gradient_scale * (selected_position.x - neighbor_position.x),
                        gradient_scale * (selected_position.y - neighbor_position.y),
                        gradient_scale * (selected_position.z - neighbor_position.z)};

                    float neighbor_density = densities[neighbor_idx];
                    float weight = -neighbor_mass * (pressures[idx] / (selected_density * selected_density) +pressures[neighbor_idx] / (neighbor_density * neighbor_density));

                    acceleration.x += weight * gradW.x;
                    acceleration.y += weight * gradW.y;
                    acceleration.z += weight * gradW.z;
                }
            }
        }
    }

    particle_accelerations[idx] = acceleration;
}