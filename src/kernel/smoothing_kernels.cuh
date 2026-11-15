#ifndef SPH_SMOOTHING_KERNELS_CUH
#define SPH_SMOOTHING_KERNELS_CUH
#include "../scene/constants.cuh"

namespace SmoothingKernels {

    __device__ __forceinline__ float poly6(float distance_sqr) {
        float smoothing_sqr = CONSTANTS.SMOOTHING_SQR;
        return CONSTANTS.POLY6 * fmaxf(smoothing_sqr - distance_sqr, 0.0f) * (smoothing_sqr - distance_sqr) * (smoothing_sqr - distance_sqr);
    }

    __device__ __forceinline__ float3 poly6Gradient(float3 displacement, float distance_sqr) {
        float smoothing_sqr = CONSTANTS.SMOOTHING_SQR;
        float kernel_gradient_scale = -6.0f * CONSTANTS.POLY6 * fmaxf(smoothing_sqr - distance_sqr, 0) * (smoothing_sqr - distance_sqr);
        return {displacement.x * kernel_gradient_scale, displacement.y * kernel_gradient_scale, displacement.z * kernel_gradient_scale};
    }

    __device__ __forceinline__ float wendland2(float distance_sqr) {
        float distance = sqrtf(distance_sqr);

        return CONSTANTS.WENDLAND2
            * fmaxf(1 - distance * CONSTANTS.SMOOTHING_INV, 0.0f)
            * (1 - distance * CONSTANTS.SMOOTHING_INV)
            * (1 - distance * CONSTANTS.SMOOTHING_INV)
            * (1 - distance * CONSTANTS.SMOOTHING_INV)
            * (1 + 4 * distance * CONSTANTS.SMOOTHING_INV);
    }

    __device__ __forceinline__ float3 wendland2Gradient(float3 displacement, float distance_sqr) {
        float distance = sqrtf(distance_sqr);
        float smoothing_inv = CONSTANTS.SMOOTHING_INV;

        float kernel_gradient_scale = -20.0f * CONSTANTS.WENDLAND2 * CONSTANTS.SMOOTHING_INV_SQR
            * fmaxf(1 - distance * smoothing_inv, 0)
            * (1 - distance * smoothing_inv)
            * (1 - distance * smoothing_inv);

        return {displacement.x * kernel_gradient_scale, displacement.y * kernel_gradient_scale, displacement.z * kernel_gradient_scale};
    }
}

#endif //SPH_SMOOTHING_KERNELS_CUH
