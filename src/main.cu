#include <cstdint>

#include "loop/propagators.h"
#include "loop/exporters.h"
#include "scene/particle_data.h"
#include "scene/scene_settings.h"

constexpr uint32_t side = 10;
constexpr float spacing = 0.025f;

int main() {
    SceneSettings settings = {
        .particle_count = 1000,
        .steps = 30000,
        .dt = 0.0002f,
        .smoothing_radius = 0.05f,
        .rest_density = 1000.0f,
        .sound_speed = 30.0f,
        .polytropic = 7.0f
    };

    ParticleData particles(settings.particle_count);

    const float mass = settings.rest_density * spacing * spacing * spacing;

    for (uint32_t i = 0; i < settings.particle_count; ++i) {
        const uint32_t x = i % side;
        const uint32_t y = (i / side) % side;
        const uint32_t z = i / (side * side);

        const float4 position = {
            (static_cast<float>(x) - 4.5f) * spacing,
            (static_cast<float>(y) - 4.5f) * spacing,
            (static_cast<float>(z) - 4.5f) * spacing,
            mass
        };

        particles.positions_one[i] = position;
        particles.positions_two[i] = position;
        particles.accelerations[i] = {0, 0, 0, 0};
    }

    HashmapData hashmap(settings.particle_count, 5000);
    IntermediateData inter_data(settings.particle_count);

    Propagators::WCSPH propagator(&settings, &particles, &hashmap, &inter_data);
    std::filesystem::path outpath = ".";
    std::filesystem::path filename = "sphdata";
    Exporters::CSV exporter(settings, outpath, filename);
    exporter.Initialize();

    for (int i = 0; i < settings.steps; i++) {
        propagator.propagate(i);
        cudaDeviceSynchronize();

        exporter.Export(particles.positions_one, particles.accelerations, inter_data.pressures, inter_data.densities, i, settings.dt);
    }

    exporter.Finalize();

    return 0;
}