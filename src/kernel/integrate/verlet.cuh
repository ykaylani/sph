#ifndef SPH_VERLET_CUH
#define SPH_VERLET_CUH

__global__ void verlet(float4* positions_one, float4* positions_two, float4* accelerations, uint32_t particle_count, float dt, uint32_t step);

#endif //SPH_VERLET_CUH
