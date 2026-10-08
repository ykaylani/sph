#ifndef SPH_EXEC_H
#define SPH_EXEC_H

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <cstdint>

#include "../scene/scene_settings.h"
#include "../scene/particle_data.h"
#include "../scene/internal/hashmap_data.h"

#include "../hash/hash.cuh"

constexpr uint16_t block_threads = 256;

struct Propagator {
    SceneSettings* settings = nullptr;
    ParticleData* particles = nullptr;

    Propagator(SceneSettings* settings, ParticleData* particles) : settings(settings), particles(particles) {}
    ~Propagator() = default;
    virtual void propagate(uint32_t step);
};

namespace Propagators {

    struct WCSPH : Propagator {
        HashmapData* hashmap_data = nullptr;

        WCSPH(SceneSettings* settings, ParticleData* particles, HashmapData* hashmap) : Propagator(settings, particles), hashmap_data(hashmap) {}

        void propagate(uint32_t step) override {
            float dt = settings->dt;
            float time = step * dt;

            uint32_t blocks_grid = (settings->particle_count + block_threads - 1) / block_threads;

            hashPositions<<<block_threads, blocks_grid>>>(
                hashmap_data->keys,
                hashmap_data->indices,
                hashmap_data->hashmap_size,
                particles->positions_one,
                settings->particle_count,
                settings->smoothing_radius);

            thrust::sort_by_key(thrust::device, hashmap_data->keys, hashmap_data->keys + settings->particle_count, hashmap_data->indices);
            thrust::fill(thrust::device, hashmap_data->cell_starts, hashmap_data->cell_starts + hashmap_data->hashmap_size, -1);
            thrust::fill(thrust::device, hashmap_data->cell_ends, hashmap_data->cell_ends + hashmap_data->hashmap_size, -1);

            hashRanges<<<block_threads, blocks_grid>>>(hashmap_data->keys, hashmap_data->cell_starts, hashmap_data->cell_ends, settings->particle_count);

            std::swap(particles->positions_one, particles->positions_two);
        }
    };
}

#endif //SPH_EXEC_H
