#ifndef SPH_HASHMAP_DATA_H
#define SPH_HASHMAP_DATA_H

struct HashmapData {
    uint32_t hashmap_size;

    int64_t* keys;
    uint32_t* indices;

    int32_t* cell_starts;
    int32_t* cell_ends;

    HashmapData(uint32_t total_particle_count, uint32_t hashmap_size) : hashmap_size(hashmap_size) {
        cudaMallocManaged(&keys, total_particle_count * sizeof(int64_t));
        cudaMallocManaged(&indices, total_particle_count * sizeof(uint32_t));

        cudaMallocManaged(&cell_starts, hashmap_size * sizeof(uint32_t));
        cudaMallocManaged(&cell_ends, hashmap_size * sizeof(uint32_t));
    }

    ~HashmapData() {
        cudaFree(keys);
        cudaFree(indices);

        cudaFree(cell_starts);
        cudaFree(cell_ends);
    }

    HashmapData(const HashmapData&) = delete;
    HashmapData& operator=(const HashmapData&) = delete;
};

#endif //SPH_HASHMAP_DATA_H
