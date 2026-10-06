#ifndef SPH_PARTICLE_DATA_H
#define SPH_PARTICLE_DATA_H

#include <cuda_runtime.h>
#include <cstdint>

struct ParticleData {
    float4* positions_one = nullptr; // w is mass
    float4* positions_two = nullptr;
    float4* accelerations = nullptr;
    float4* forces = nullptr;

    ParticleData(uint32_t particle_count) {
        cudaMallocManaged(&positions_one, sizeof(float4) * particle_count);
        cudaMallocManaged(&positions_two, sizeof(float4) * particle_count);

        cudaMallocManaged(&accelerations, sizeof(float4) * particle_count);
        cudaMallocManaged(&forces, sizeof(float4) * particle_count);
    }

    ~ParticleData() {
        cudaFree(positions_one);
        cudaFree(positions_two);

        cudaFree(accelerations);
        cudaFree(forces);
    }

    ParticleData(const ParticleData&) = delete;
    ParticleData& operator=(const ParticleData&) = delete;
};

#endif //SPH_PARTICLE_DATA_H
