#include <cuda_runtime.h>
#include <cstdint>

__device__ float4 eulerStart(float4 position, float4 acceleration, float dt) {
    float3 vel = {acceleration.x * dt, acceleration.y * dt, acceleration.z * dt};
    return {position.x + vel.x * dt, position.y + vel.y * dt, position.z + vel.z * dt, position.w};
}

__global__ void verlet(float4* positions_one, float4* positions_two, float4* accelerations, uint32_t particle_count, float dt, uint32_t step) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count) return;

    float4 position_one = positions_one[idx];
    float4 acceleration = accelerations[idx];
    if (step == 0) {positions_one[idx] = eulerStart(position_one, acceleration, dt); return; }
    float4 position_two = positions_two[idx];

    float dt_sqr = dt * dt;

    float4 new_position = {
        2.0f * position_one.x - position_two.x + acceleration.x * dt_sqr,
        2.0f * position_one.y - position_two.y + acceleration.y * dt_sqr,
        2.0f * position_one.z - position_two.z + acceleration.z * dt_sqr,
        position_one.w
    };

    positions_two[idx] = new_position;
}