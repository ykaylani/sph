#include <cuda_runtime.h>
#include <cstdint>

#include "../hash/hash.cuh"

__global__ void viscosityAcceleration(
    float* densities,
    float4* particle_accelerations,
    float4* particle_positions,
    float4* particle_positions_preceding,
    uint32_t particle_count,
    float dt_inv,

    int32_t* cell_starts,
    int32_t* cell_ends,
    uint32_t* indices,
    uint32_t hashmap_size,

    float smoothing_radius,
    float inverse_smoothing,
    float inverse_smoothing_sqr,
    float smoothing_norm,

    float viscosity_regularization,
    float viscosity) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count) return;

    float4 selected_position = particle_positions[idx];
    float4 selected_position_preceding = particle_positions_preceding[idx];
    float selected_density = densities[idx];

    float3 selected_dx = {selected_position.x - selected_position_preceding.x, selected_position.y - selected_position_preceding.y, selected_position.z - selected_position_preceding.z};
    float3 selected_velocity = {selected_dx.x * dt_inv, selected_dx.y * dt_inv, selected_dx.z * dt_inv};

    float4 acceleration = {0, 0, 0, 0};

    int32_t posx = floorf(selected_position.x * inverse_smoothing);
    int32_t posy = floorf(selected_position.y * inverse_smoothing);
    int32_t posz = floorf(selected_position.z * inverse_smoothing);

    for (int32_t dz = -1; dz <= 1; ++dz) {
        for (int32_t dy = -1; dy <= 1; ++dy) {
            for (int32_t dx = -1; dx <= 1; ++dx) {
                uint32_t bucket = hashCell(posx + dx, posy + dy, posz + dz, hashmap_size);

                int32_t start = cell_starts[bucket];
                if (start == -1) continue;

                int32_t end = cell_ends[bucket];

                for (int slot = start; slot < end; ++slot)
                {
                    uint32_t neighbor_idx = indices[slot];
                    if (neighbor_idx == idx) continue;

                    float4 neighbor_position = particle_positions[neighbor_idx];
                    float4 neighbor_position_preceding = particle_positions_preceding[neighbor_idx];

                    int32_t neighbor_x = floorf(neighbor_position.x * inverse_smoothing);
                    int32_t neighbor_y = floorf(neighbor_position.y * inverse_smoothing);
                    int32_t neighbor_z = floorf(neighbor_position.z * inverse_smoothing);
                    if (neighbor_x != posx + dx || neighbor_y != posy + dy || neighbor_z != posz + dz) continue;

                    float neighbor_density = densities[neighbor_idx];

                    float3 neighbor_dx = {neighbor_position.x - neighbor_position_preceding.x, neighbor_position.y - neighbor_position_preceding.y, neighbor_position.z - neighbor_position_preceding.z};
                    float3 neighbor_velocity = {neighbor_dx.x * dt_inv, neighbor_dx.y * dt_inv, neighbor_dx.z * dt_inv};

                    float3 distance = {selected_position.x - neighbor_position.x, selected_position.y - neighbor_position.y, selected_position.z - neighbor_position.z};
                    float distance_sqr =
                        (neighbor_position.x - selected_position.x) * (neighbor_position.x - selected_position.x) +
                        (neighbor_position.y - selected_position.y) * (neighbor_position.y - selected_position.y) +
                        (neighbor_position.z - selected_position.z) * (neighbor_position.z - selected_position.z);

                    float op = 1.0f - distance_sqr * inverse_smoothing_sqr;
                    if (op <= 0.0f) continue;

                    float gradient_scale = -6.0f * smoothing_norm * inverse_smoothing_sqr * op * op;

                    float3 gradW = {
                        gradient_scale * (selected_position.x - neighbor_position.x),
                        gradient_scale * (selected_position.y - neighbor_position.y),
                        gradient_scale * (selected_position.z - neighbor_position.z)};

                    float gradW_dot_distance = distance.x * gradW.x + distance.y * gradW.y + distance.z * gradW.z;
                    float weight_denominator = smoothing_radius * smoothing_radius * viscosity_regularization + distance_sqr;

                    float kernel_weight = -gradW_dot_distance / weight_denominator;
                    float prefactor = (2 * viscosity * neighbor_position.w) / (selected_density * neighbor_density);
                    float scalars = kernel_weight * prefactor;

                    float3 dvel = {neighbor_velocity.x - selected_velocity.x, neighbor_velocity.y - selected_velocity.y, neighbor_velocity.z - selected_velocity.z};

                    acceleration.x += dvel.x * scalars;
                    acceleration.y += dvel.y * scalars;
                    acceleration.z += dvel.z * scalars;
                }

            }
        }
    }

    float4 accelAcc = particle_accelerations[idx];
    accelAcc.x += acceleration.x;
    accelAcc.y += acceleration.y;
    accelAcc.z += acceleration.z;
    accelAcc.w += acceleration.w;

    particle_accelerations[idx] = accelAcc;
}