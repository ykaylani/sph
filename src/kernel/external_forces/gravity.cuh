#ifndef SPH_GRAVITY_CUH
#define SPH_GRAVITY_CUH

__global__ void gravity(float4* acceleration, uint32_t particle_count);

#endif //SPH_GRAVITY_CUH
