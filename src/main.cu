#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <filesystem>

#include "scene/constants.cuh"
#include "scene/scene_settings.h"
#include "scene/particle_data.h"
#include "scene/internal/hashmap_data.h"
#include "scene/internal/intermediate_data.h"
#include "loop/propagators.cuh"
#include "loop/exporters.h"

namespace
{
    constexpr float spacing = 0.025f;
    constexpr int tank_x = 28;
    constexpr int tank_z = 12;
    constexpr int fluid_x = 10;
    constexpr int fluid_y = 10;
    constexpr int fluid_z = tank_z;
    constexpr int wall_height = 16;

    constexpr uint32_t export_every = 100;
    constexpr double pi = 3.14159265358979323846;
    static_assert(fluid_x < tank_x && fluid_z <= tank_z && fluid_y < wall_height);


    float wendlandLatticeMass(float rho0, float h) {
        const double radius = h;
        const double normalization = 21.0 / (2.0 * pi * radius * radius * radius);
        const int reach = static_cast<int>(std::ceil(radius / spacing));
        double sum = 0.0;

        for (int z = -reach; z <= reach; ++z) {
            for (int y = -reach; y <= reach; ++y) {
                for (int x = -reach; x <= reach; ++x) {
                    const double q = std::sqrt(double(x*x + y*y + z*z)) * spacing / radius;
                    if (q >= 1.0) continue;
                    const double t = 1.0 - q;
                    sum += normalization * t*t*t*t * (1.0 + 4.0*q);
                }
            }
        }

        return static_cast<float>(rho0 / sum);
    }
}

int main() {
    SceneSettings settings = {
        .steps = 100000,
        .dt = 0.00005f,
        .smoothing_radius = 2.5f * spacing,
        .rest_density = 1000.0f,
        .sound_speed = 30.0f,
        .polytropic = 7.0f,
        .viscosity_regularization = 0.01f,
        .viscosity = 0.001f
    };

    uloadConstants(settings.smoothing_radius, settings.polytropic, settings.rest_density, settings.dt);

    Constants uploaded{};
    cudaMemcpyFromSymbol(&uploaded, CONSTANTS, sizeof(uploaded));

    const double h = settings.smoothing_radius;
    const double expected_wendland = 21.0 / (2.0 * pi * h*h*h);

    const int thickness = static_cast<int>(std::ceil(h / spacing));
    const int outer_x = tank_x + 2 * thickness;
    const int outer_z = tank_z + 2 * thickness;

    const uint32_t dynamic_count = fluid_x * fluid_y * fluid_z;
    const uint32_t static_count = outer_x * outer_z * thickness + (outer_x * outer_z - tank_x * tank_z) * wall_height;

    ParticleData particles(dynamic_count, static_count);
    HashmapData hashmap(particles.total_particles, 8192);
    IntermediateData inter_data(particles.total_particles);

    const float mass = wendlandLatticeMass(settings.rest_density, settings.smoothing_radius);
    uint32_t next = 0;

    auto addParticle = [&](int x, int y, int z) {
        const float4 position = { (x - 0.5f * (tank_x - 1)) * spacing, (y + 0.5f) * spacing, (z - 0.5f * (tank_z - 1)) * spacing, mass };
        particles.positions_one[next] = position;
        particles.positions_two[next] = position;
        particles.velocities[next] = {0, 0, 0, 0};
        particles.accelerations[next] = {0, 0, 0, 0};
        inter_data.densities[next] = 0.0f;
        inter_data.pressures[next] = 0.0f;
        ++next;
    };

    for (int y = 0; y < fluid_y; ++y)
        for (int z = 0; z < fluid_z; ++z)
            for (int x = 0; x < fluid_x; ++x)
                addParticle(x, y, z);

    for (int y = -thickness; y < 0; ++y)
        for (int z = -thickness; z < tank_z + thickness; ++z)
            for (int x = -thickness; x < tank_x + thickness; ++x)
                addParticle(x, y, z);

    for (int y = 0; y < wall_height; ++y)
        for (int z = -thickness; z < tank_z + thickness; ++z)
            for (int x = -thickness; x < tank_x + thickness; ++x) {
                const bool outside = x < 0 || x >= tank_x || z < 0 || z >= tank_z;
                if (outside) addParticle(x, y, z);
            }

    Propagators::WCSPH propagator(&settings, &particles, &hashmap, &inter_data);
    const std::filesystem::path output = "output/";
    Exporters::XDMF exporter(settings, output, "binary_data.bin", "xdmf_organization.xmf");
    exporter.Initialize(particles.dynamic_particle_count);

    for (uint32_t step = 0; step < settings.steps; ++step) {
        propagator.propagate(step);

        const uint32_t completed = step + 1;
        if (completed != 1 && completed % export_every != 0 && completed != settings.steps) continue;
        exporter.Export(particles.positions_one, particles.velocities, inter_data.pressures, inter_data.densities, completed, settings.dt, particles.dynamic_particle_count);
    }

    exporter.Finalize();

    return 0;

}