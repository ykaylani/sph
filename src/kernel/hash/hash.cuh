#ifndef SPH_HASH_CUH
#define SPH_HASH_POSITION_CUH

__global__ void hashPositions(int64_t* keys_out, uint32_t* indices_out, uint32_t hashmap_size, float4* positions, uint32_t particle_count, float smoothing_radius);
__global__ void hashRanges(int64_t* keys, int32_t* cell_starts, int32_t* cell_ends, uint32_t particle_count);
__device__ uint32_t hashCell(int32_t x, int32_t y, int32_t z, uint32_t hashmap_size);

#endif //SPH_HASH_CUH
