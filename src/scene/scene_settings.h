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

    float viscosity_regularization = 0.01f;
    float viscosity = 0.05f;

    // useful precomputation

    float inverse_smoothing = (smoothing_radius != 0.0f) ? (1.0f / smoothing_radius) : 0.0f;
    float smoothing_sqr_inv = inverse_smoothing * inverse_smoothing;
    float smoothing_norm = 1.56668147106f * smoothing_sqr_inv * inverse_smoothing;

    float polytropic_inv = 1.0f / polytropic;
    float rest_density_inv = 1.0f / rest_density;

    float dt_inv = 1.0f / dt;
};

#endif //SPH_SCENE_SETTINGS_H
