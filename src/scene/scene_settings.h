#ifndef SPH_SCENE_SETTINGS_H
#define SPH_SCENE_SETTINGS_H

#include <cstdint>

struct SceneSettings {
    uint32_t steps;
    float dt;

    float smoothing_radius;

    float rest_density;
    float sound_speed;
    float polytropic = 7;

    float viscosity_regularization = 0.001f;
    float viscosity = 0.05f;
};

#endif //SPH_SCENE_SETTINGS_H
