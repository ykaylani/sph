#ifndef SPH_HASH_CUH
#define SPH_HASH_POSITION_CUH

__global__ void hashPositions(int64_t* keys_out, uint32_t* indices_out, uint32_t hashmap_size, float4* positions, uint32_t body_count, float smoothing_radius);
__global__ void hashRanges(int64_t* keys, int32_t* cell_starts, int32_t* cell_ends, uint32_t body_count);

#endif //SPH_HASH_CUH
