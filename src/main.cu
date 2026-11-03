#include <cassert>
#include <cstdint>
#include <filesystem>

#include <cuda_runtime.h>

#include "loop/propagators.h"
#include "loop/exporters.h"
#include "scene/particle_data.h"
#include "scene/scene_settings.h"

constexpr uint32_t side = 10;
constexpr float spacing = 0.025f;

int main() {
    SceneSettings settings = {
        .steps = 3000,
        .dt = 0.0002f,
        .smoothing_radius = 0.05f,
        .rest_density = 1000.0f,
        .sound_speed = 30.0f,
        .polytropic = 7.0f
    };

    constexpr uint32_t dynamic_count = 3000;
    constexpr uint32_t floor_width = 20;
    constexpr uint32_t floor_depth = 25;
    constexpr uint32_t thickness = 2;

    constexpr uint32_t fluid_height = (dynamic_count + side * side - 1) / (side * side);

    constexpr uint32_t wall_height = fluid_height + 2;
    constexpr uint32_t floor_area = floor_width * floor_depth;
    constexpr uint32_t interior_area = (floor_width - 2 * thickness) * (floor_depth - 2 * thickness);

    constexpr uint32_t static_count = floor_area * thickness + (floor_area - interior_area) * wall_height;

    ParticleData particles(dynamic_count, static_count);

    const float mass = settings.rest_density * spacing * spacing * spacing;
    const float fluid_offset = 0.5f * static_cast<float>(side - 1);
    const float fluid_min_z = -fluid_offset * spacing;

    for (uint32_t i = 0; i < particles.dynamic_particle_count; ++i) {
        const uint32_t x = i % side;
        const uint32_t y = (i / side) % side;
        const uint32_t z = i / (side * side);

        const float4 position = {
            (static_cast<float>(x) - fluid_offset) * spacing,
            (static_cast<float>(y) - fluid_offset) * spacing,
            (static_cast<float>(z) - fluid_offset) * spacing,
            mass
        };

        particles.positions_one[i] = position;
        particles.positions_two[i] = position;
        particles.accelerations[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    uint32_t next_static = particles.dynamic_particle_count;

    for (uint32_t z = 0; z < thickness + wall_height; ++z) {
        for (uint32_t y = 0; y < floor_depth; ++y) {
            for (uint32_t x = 0; x < floor_width; ++x) {
                const bool is_floor = z < thickness;
                const bool is_wall = x < thickness || x >= floor_width - thickness || y < thickness || y >= floor_depth - thickness;

                if (!is_floor && !is_wall) { continue; }

                const float4 position = {
                    (static_cast<float>(x) - 0.5f * static_cast<float>(floor_width - 1)) * spacing,
                    (static_cast<float>(y) - 0.5f * static_cast<float>(floor_depth - 1)) * spacing,
                    fluid_min_z + (static_cast<float>(z) - static_cast<float>(thickness)) * spacing,
                    mass
                };

                particles.positions_one[next_static] = position;
                particles.positions_two[next_static] = position;
                particles.accelerations[next_static] = {0.0f, 0.0f, 0.0f, 0.0f};
                ++next_static;
            }
        }
    }

    HashmapData hashmap(particles.total_particles, 5000);
    IntermediateData inter_data(particles.total_particles);

    Propagators::WCSPH propagator(&settings, &particles, &hashmap, &inter_data);

    std::filesystem::path outpath = ".";
    std::filesystem::path filename_xmf = "sphdata";
    std::filesystem::path filename_bin = "sphdatabin";
    //Exporters::XDMF exporter(settings, outpath, filename_bin, filename_xmf);
    //exporter.Initialize(particles.total_particles);

    std::cout << "Initialized" << std::endl;

    for (uint32_t step = 0; step < settings.steps; ++step) {
        propagator.propagate(step);
        cudaDeviceSynchronize();

        //exporter.Export(particles.positions_one, particles.accelerations, inter_data.pressures, inter_data.densities, step, settings.dt, particles.total_particles);
    }

    //exporter.Finalize();
    return 0;
}