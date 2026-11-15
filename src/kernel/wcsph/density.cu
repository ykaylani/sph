#include <cuda_runtime.h>
#include <cstdint>

#include "../hash/hash.cuh"
#include "../../scene/constants.cuh"
#include "../smoothing_kernels.cuh"

__global__ void density(
    int32_t* cell_starts,
    int32_t* cell_ends,
    uint32_t* indices,
    uint32_t hashmap_size,

    float4* particle_positions,
    uint32_t particle_count,

    float* densities,
    float* pressures,

    float rest_density,
    float sound_speed,
    float polytropic) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count) return;

    float4 selected_position = particle_positions[idx];
    float inverse_smoothing = CONSTANTS.SMOOTHING_INV;

    int32_t posx = floorf(selected_position.x * inverse_smoothing);
    int32_t posy = floorf(selected_position.y * inverse_smoothing);
    int32_t posz = floorf(selected_position.z * inverse_smoothing);

    float density = 0;

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

                    int32_t neighbor_x = floorf(neighbor_position.x * inverse_smoothing);
                    int32_t neighbor_y = floorf(neighbor_position.y * inverse_smoothing);
                    int32_t neighbor_z = floorf(neighbor_position.z * inverse_smoothing);
                    if (neighbor_x != posx + dx || neighbor_y != posy + dy || neighbor_z != posz + dz) continue;
                    float neighbor_mass = neighbor_position.w;

                    float dist_sqr =
                        (neighbor_position.x - selected_position.x) * (neighbor_position.x - selected_position.x) +
                        (neighbor_position.y - selected_position.y) * (neighbor_position.y - selected_position.y) +
                        (neighbor_position.z - selected_position.z) * (neighbor_position.z - selected_position.z);

                    density += neighbor_mass * SmoothingKernels::wendland2(dist_sqr);
                }
            }
        }
    }

    densities[idx] = density;
    pressures[idx] = fmaxf(rest_density * sound_speed * sound_speed * CONSTANTS.POLYTROPIC_INV * (powf(density * CONSTANTS.REST_DENSITY_INV, polytropic) - 1.0f), 0.0f);

}