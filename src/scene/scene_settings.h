#ifndef SPH_SCENE_SETTINGS_H
#define SPH_SCENE_SETTINGS_H

#include <cstdint>

struct SceneSettings {
    uint32_t particle_count; // general
    uint32_t steps;
    float dt;

    float smoothing_radius;

    float rest_density; // tait equation
    float sound_speed;
    float polytropic = 7;

    float inverse_smoothing = (smoothing_radius != 0.0f) ? (1.0f / smoothing_radius) : 0.0f;
    float smoothing_sqr_inv = inverse_smoothing * inverse_smoothing;
    float smoothing_norm = 1.56668147106f * smoothing_sqr_inv * inverse_smoothing;

    float polytropic_inv = 1.0f / polytropic;
    float rest_density_inv = 1.0f / rest_density;
};

#endif //SPH_SCENE_SETTINGS_H
