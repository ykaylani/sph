#ifndef SPH_PRESSURE_CUH
#define SPH_PRESSURE_CUH
#include <cstdint>

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
    float polytropic);

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
    uint32_t particle_count);

#endif //SPH_PRESSURE_CUH
