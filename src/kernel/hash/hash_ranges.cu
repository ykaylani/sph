#include <cuda_runtime.h>
#include <cstdint>

__global__ void hashRanges(int64_t* keys, int32_t* cell_starts, int32_t* cell_ends, uint32_t particle_count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_count) return;

    int64_t key = keys[idx];
    if (idx == 0 || key != keys[idx - 1]) cell_starts[key] = idx;
    if (idx == particle_count - 1 || key != keys[idx + 1]) cell_ends[key] = idx + 1;
}