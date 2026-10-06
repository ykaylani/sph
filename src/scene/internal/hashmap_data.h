#ifndef SPH_HASHMAP_DATA_H
#define SPH_HASHMAP_DATA_H

struct HashmapData {
    uint32_t hashmap_size;

    int64_t* keys;
    uint32_t* indices;

    HashmapData(uint32_t body_count, uint32_t hashmap_size) : hashmap_size(hashmap_size) {
        cudaMallocManaged(&keys, body_count * sizeof(int64_t));
        cudaMallocManaged(&indices, body_count * sizeof(uint32_t));
    }

    ~HashmapData() {
        cudaFree(keys);
        cudaFree(indices);
    }

    HashmapData(const HashmapData&) = delete;
    HashmapData& operator=(const HashmapData&) = delete;
};

#endif //SPH_HASHMAP_DATA_H
