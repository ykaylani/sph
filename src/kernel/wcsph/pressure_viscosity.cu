#include <cuda_runtime.h>
#include <cstdint>

#include "../smoothing_kernels.cuh"
#include "../hash/hash.cuh"

__global__ void wcsphAcceleration(
    float* densities,
    float* pressures,

    float viscosity,
    float viscosity_regularization,

    int32_t* cell_starts,
    int32_t* cell_ends,
    uint32_t* indices,
    uint32_t hashmap_size,

    float4* particle_accelerations,
    float4* particle_velocities,
    float4* particle_positions,
    uint32_t particle_count) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count) return;

    float4 selected_position = particle_positions[idx];
    float inverse_smoothing = CONSTANTS.SMOOTHING_INV;

    int32_t posx = floorf(selected_position.x * inverse_smoothing);
    int32_t posy = floorf(selected_position.y * inverse_smoothing);
    int32_t posz = floorf(selected_position.z * inverse_smoothing);

    float selected_density = densities[idx];
    float4 selected_velocity = particle_velocities[idx];
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

                    // Pressure

                    float3 displacement = { selected_position.x - neighbor_position.x, selected_position.y - neighbor_position.y, selected_position.z - neighbor_position.z };
                    float distance_sqr = displacement.x * displacement.x + displacement.y * displacement.y + displacement.z * displacement.z;
                    float3 kernel_gradient = SmoothingKernels::wendland2Gradient(displacement, distance_sqr);

                    float neighbor_density = densities[neighbor_idx];
                    float weight = -neighbor_mass * (pressures[idx] / (selected_density * selected_density) + pressures[neighbor_idx] / (neighbor_density * neighbor_density));

                    float3 pressure_acceleration = {weight * kernel_gradient.x, weight * kernel_gradient.y, weight * kernel_gradient.z};

                    // Viscosity

                    float displacement_dot_gradient = displacement.x * kernel_gradient.x + displacement.y * kernel_gradient.y + displacement.z * kernel_gradient.z;
                    float weight_denominator = CONSTANTS.SMOOTHING_SQR * viscosity_regularization + distance_sqr;

                    float viscosity_laplacian = -displacement_dot_gradient / weight_denominator;
                    float prefactor = (2 * viscosity * neighbor_position.w) / (selected_density * neighbor_density);
                    float scalars = viscosity_laplacian * prefactor;

                    float4 neighbor_velocity = particle_velocities[neighbor_idx];
                    float3 velocity_difference = {neighbor_velocity.x - selected_velocity.x, neighbor_velocity.y - selected_velocity.y, neighbor_velocity.z - selected_velocity.z};
                    float3 viscosity_acceleration = {velocity_difference.x * scalars, velocity_difference.y * scalars, velocity_difference.z * scalars};

                    acceleration.x += pressure_acceleration.x + viscosity_acceleration.x;
                    acceleration.y += pressure_acceleration.y + viscosity_acceleration.y;
                    acceleration.z += pressure_acceleration.z + viscosity_acceleration.z;
                }
            }
        }
    }

    particle_accelerations[idx] = acceleration;
}