#ifndef SPH_INTERMEDIATE_DATA_H
#define SPH_INTERMEDIATE_DATA_H

struct IntermediateData {
    float* densities = nullptr;
    float* pressures = nullptr;

    IntermediateData(uint32_t total_particle_count) {
        cudaMallocManaged(&densities, total_particle_count * sizeof(float));
        cudaMallocManaged(&pressures, total_particle_count * sizeof(float));
    }

    ~IntermediateData() {
        cudaFree(densities);
        cudaFree(pressures);
    }

    IntermediateData(const IntermediateData&) = delete;
    IntermediateData& operator=(const IntermediateData&) = delete;
};

#endif //SPH_INTERMEDIATE_DATA_H
