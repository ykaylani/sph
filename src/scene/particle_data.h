#ifndef SPH_PARTICLE_DATA_H
#define SPH_PARTICLE_DATA_H

#include <cuda_runtime.h>
#include <cstdint>

struct ParticleData {
    float4* positions_one = nullptr; // w is mass
    float4* positions_two = nullptr;
    float4* accelerations = nullptr;

    uint32_t statics_start = 0; // index in particle array where particles will not be integrated (only add to density)
    uint32_t dynamic_particle_count = 0;
    uint32_t static_particle_count = 0;
    uint32_t total_particles = 0;

    ParticleData(uint32_t dynamicparticle_count, uint32_t staticparticle_count) {
        statics_start = dynamicparticle_count;
        dynamic_particle_count = dynamicparticle_count;
        static_particle_count = staticparticle_count;
        total_particles = dynamicparticle_count + staticparticle_count;

        cudaMallocManaged(&positions_one, sizeof(float4) * total_particles);
        cudaMallocManaged(&positions_two, sizeof(float4) * total_particles);
        cudaMallocManaged(&accelerations, sizeof(float4) * total_particles);
    }

    ~ParticleData() {
        cudaFree(positions_one);
        cudaFree(positions_two);
        cudaFree(accelerations);
    }

    ParticleData(const ParticleData&) = delete;
    ParticleData& operator=(const ParticleData&) = delete;
};

#endif //SPH_PARTICLE_DATA_H
