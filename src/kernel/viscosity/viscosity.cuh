#ifndef SPH_VISCOSITY_CUH
#define SPH_VISCOSITY_CUH

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
    float viscosity);

#endif //SPH_VISCOSITY_CUH
