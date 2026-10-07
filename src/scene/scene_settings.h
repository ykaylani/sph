#ifndef SPH_SCENE_SETTINGS_H
#define SPH_SCENE_SETTINGS_H

#include <cstdint>

struct SceneSettings {
    uint32_t particle_count;

    float smoothing_radius;

    float rest_density; // tait equation
    float stiffness;

    float dt; // general
    uint32_t steps;
};

#endif //SPH_SCENE_SETTINGS_H
