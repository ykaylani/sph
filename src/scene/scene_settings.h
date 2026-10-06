#ifndef SPH_SCENE_SETTINGS_H
#define SPH_SCENE_SETTINGS_H

#include <cstdint>

struct SceneSettings {
    uint32_t particle_count;

    float smoothing_radius;

    float rest_density; // tait equation
    float stiffness;
};

#endif //SPH_SCENE_SETTINGS_H
