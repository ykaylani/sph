#ifndef SPH_INTEGRATORS_CUH
#define SPH_INTEGRATORS_CUH
#include <cstdint>

namespace Integrators {

    __global__ inline void siEuler(float4* particle_positions, float4* particle_positions_two, float4* particle_accelerations, float4* particle_velocities, uint32_t particle_count, float dt) {
        uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= particle_count) return;

        float4 acceleration = particle_accelerations[idx];
        float4 velocity = particle_velocities[idx];
        float4 position = particle_positions[idx];

        float4 dta = {acceleration.x * dt, acceleration.y * dt, acceleration.z * dt, 0};
        float4 velocity_new = {velocity.x + dta.x, velocity.y + dta.y, velocity.z + dta.z, 0};
        float4 dtv = {velocity_new.x * dt, velocity_new.y * dt, velocity_new.z * dt, 0};

        particle_velocities[idx] = velocity_new;
        particle_positions_two[idx] = {position.x + dtv.x, position.y + dtv.y, position.z + dtv.z, position.w};
    }
}

#endif //SPH_INTEGRATORS_CUH
