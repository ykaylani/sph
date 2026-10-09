#ifndef SPH_EXEC_H
#define SPH_EXEC_H

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/fill.h>

#include "../scene/scene_settings.h"
#include "../scene/particle_data.h"
#include "../scene/internal/hashmap_data.h"
#include "../scene/internal/intermediate_data.h"

#include "../kernel/hash/hash.cuh"
#include "../kernel/pressure/pressure.cuh"
#include "../kernel/integrate/verlet.cuh"

constexpr uint16_t block_threads = 256;

struct Propagator {
    SceneSettings* settings = nullptr;
    ParticleData* particles = nullptr;

    Propagator(SceneSettings* settings, ParticleData* particles) : settings(settings), particles(particles) {}
    virtual ~Propagator() = default;
    virtual void propagate(uint32_t step);
};

namespace Propagators {

    struct WCSPH : Propagator {
        HashmapData* hashmap_data = nullptr;
        IntermediateData* intermediate_data = nullptr;

        WCSPH(SceneSettings* settings, ParticleData* particles, HashmapData* hashmap, IntermediateData* intermediate_data) : Propagator(settings, particles), hashmap_data(hashmap), intermediate_data(intermediate_data) {}

        void propagate(uint32_t step) override {
            float dt = settings->dt;
            float time = step * dt;

            uint32_t blocks_grid = (settings->particle_count + block_threads - 1) / block_threads;

            hashPositions<<<blocks_grid, block_threads>>>(
                hashmap_data->keys,
                hashmap_data->indices,
                hashmap_data->hashmap_size,
                particles->positions_one,
                settings->particle_count,
                settings->inverse_smoothing);

            thrust::sort_by_key(thrust::device, hashmap_data->keys, hashmap_data->keys + settings->particle_count, hashmap_data->indices);
            thrust::fill(thrust::device, hashmap_data->cell_starts, hashmap_data->cell_starts + hashmap_data->hashmap_size, -1);
            thrust::fill(thrust::device, hashmap_data->cell_ends, hashmap_data->cell_ends + hashmap_data->hashmap_size, -1);

            hashRanges<<<blocks_grid, block_threads>>>(hashmap_data->keys, hashmap_data->cell_starts, hashmap_data->cell_ends, settings->particle_count);

            density<<<blocks_grid, block_threads>>>(
                hashmap_data->cell_starts,
                hashmap_data->cell_ends,
                hashmap_data->indices,
                hashmap_data->hashmap_size,
                particles->positions_one,
                settings->particle_count,
                settings->inverse_smoothing,
                settings->smoothing_sqr_inv,
                settings->smoothing_norm,
                intermediate_data->densities);

            pressure<<<blocks_grid, block_threads>>>(
                intermediate_data->pressures,
                intermediate_data->densities,
                settings->particle_count,
                settings->polytropic,
                settings->polytropic_inv,
                settings->rest_density,
                settings->rest_density_inv,
                settings->sound_speed);

            pressureAcceleration<<<blocks_grid, block_threads>>>(
                intermediate_data->densities,
                intermediate_data->pressures,
                hashmap_data->cell_starts,
                hashmap_data->cell_ends,
                hashmap_data->indices,
                hashmap_data->hashmap_size,
                settings->inverse_smoothing,
                settings->smoothing_sqr_inv,
                settings->smoothing_norm,
                particles->accelerations,
                particles->positions_one,
                settings->particle_count);

            verlet(particles->positions_one, particles->positions_two, particles->accelerations, settings->particle_count, settings->dt, step);
            std::swap(particles->positions_one, particles->positions_two);
        }
    };
}

#endif //SPH_EXEC_H
