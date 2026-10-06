#ifndef SPH_HASH_CUH
#define SPH_HASH_POSITION_CUH

__global__ void hashPositions(int64_t* keys_out, uint32_t* indices_out, uint32_t hashmap_size, float4* positions, uint32_t body_count, float smoothing_radius);


#endif //SPH_HASH_CUH
