// ple_gpu_gather — GPU-side row gather from the mmap'd PLE table via ATS.
//
// On GB10 / Grace-Blackwell (Addressing Mode: ATS) a CUDA kernel can
// dereference pageable, file-backed host memory directly: loads walk the CPU
// page tables over the coherent link, the page cache services the reads, and
// nothing is pinned or copied. Probe-verified on-box: byte-identical gathers,
// 0.195 ms warm for 4096 x 160 B, and capturable inside CUDA graphs (0.216 ms),
// which is the point — with the gather on-device the whole PLE lookup can stay
// inside piecewise CUDA graph segments instead of splitting them.
//
// Compiled at image build (Dockerfile step 8) into ple_gpu_gather.so and loaded
// by vllm_ple_mmap.py when VLLM_PLE_GPU_GATHER=1. No host-side prep: the .so
// ships in the image.
//
// The shard table is passed as a device array of base pointers (one per shard,
// each the mmap base of that shard's row region) so one launch covers rows from
// any mix of shards.

#include <cstdint>
#include <cuda_runtime.h>

extern "C" {

__global__ void ple_gather_kernel(const uint8_t* const* shard_bases,
                                  int64_t shard_size,
                                  const int64_t* ids,
                                  uint8_t* out,
                                  int64_t row_bytes,
                                  int64_t n_rows) {
    int64_t row = blockIdx.x;
    if (row >= n_rows) return;
    int64_t id = ids[row];
    const uint8_t* src =
        shard_bases[id / shard_size] + (id % shard_size) * row_bytes;
    uint8_t* dst = out + row * row_bytes;
    // 128 threads stride a 160-byte row; wider rows also covered.
    for (int64_t j = threadIdx.x; j < row_bytes; j += blockDim.x) dst[j] = src[j];
}

// Launch wrapper: everything is passed as plain integers/pointers so Python can
// call through ctypes with no torch C++ extension build dependency.
int ple_gpu_gather(const void* shard_bases_dev,  // const uint8_t* const* (device)
                   long long shard_size,
                   const void* ids_dev,          // const int64_t* (device)
                   void* out_dev,                // uint8_t* (device)
                   long long row_bytes,
                   long long n_rows,
                   void* stream) {
    if (n_rows <= 0) return 0;
    ple_gather_kernel<<<(unsigned)n_rows, 128, 0, (cudaStream_t)stream>>>(
        (const uint8_t* const*)shard_bases_dev, shard_size,
        (const int64_t*)ids_dev, (uint8_t*)out_dev, row_bytes, n_rows);
    return (int)cudaGetLastError();
}

}  // extern "C"
