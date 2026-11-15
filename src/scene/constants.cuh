#ifndef SPH_CONSTANTS_CUH
#define SPH_CONSTANTS_CUH

struct Constants {
    float SMOOTHING_INV;
    float SMOOTHING_SQR;
    float SMOOTHING_INV_SQR;

    float POLYTROPIC_INV;
    float REST_DENSITY_INV;

    float HZ;
    float HALF_DT;

    float POLY6;
    float WENDLAND2;
};

inline __constant__ Constants CONSTANTS;

inline void uloadConstants(float smoothing_radius, float polytropic, float rest_density, float dt) {

    Constants computed {

        .SMOOTHING_INV = (smoothing_radius != 0.0f) ? (1.0f / smoothing_radius) : 0.0f,
        .SMOOTHING_SQR = smoothing_radius * smoothing_radius,
        .SMOOTHING_INV_SQR = computed.SMOOTHING_INV * computed.SMOOTHING_INV,

        .POLYTROPIC_INV = 1.0f / polytropic,
        .REST_DENSITY_INV = 1.0f / rest_density,

        .HZ = 1.0f / dt,
        .HALF_DT = dt / 2.0f,

        .POLY6 = 1.5666815f
            * computed.SMOOTHING_INV
            * computed.SMOOTHING_INV
            * computed.SMOOTHING_INV
            * computed.SMOOTHING_INV
            * computed.SMOOTHING_INV
            * computed.SMOOTHING_INV
            * computed.SMOOTHING_INV
            * computed.SMOOTHING_INV
            * computed.SMOOTHING_INV,

        .WENDLAND2 = 3.3422538f
            * computed.SMOOTHING_INV
            * computed.SMOOTHING_INV
            * computed.SMOOTHING_INV
    };

    cudaMemcpyToSymbol(CONSTANTS, &computed, sizeof(computed));
}

#endif //SPH_CONSTANTS_CUH
