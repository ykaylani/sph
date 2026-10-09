#ifndef SPH_PRESSURE_CUH
#define SPH_PRESSURE_CUH

__global__ void density(
    int32_t* cell_starts,
    int32_t* cell_ends,
    uint32_t* indices,
    uint32_t hashmap_size,
    float4* particle_positions,
    uint32_t particle_count,
    float inverse_smoothing,
    float inverse_smoothing_sqr,
    float smoothing_norm,
    float* densities);

__global__ void pressure(
    float* pressure,
    float* density,
    uint32_t particle_count,
    float polytropic,
    float polytropic_inv,
    float rest_density,
    float rest_density_inv,
    float sound_speed);

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
    uint32_t particle_count);

#endif //SPH_PRESSURE_CUH
