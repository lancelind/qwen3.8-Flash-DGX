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
// Cold pages are handled OUTSIDE this kernel: the CPU warm path (madvise +
// touch threads at disk queue depth ~23) populates the page cache ahead of
// the launch — prefill synchronously, decode via the execute_model hook. A
// row the warmer missed demand-faults here (queue-depth-~1, slow but
// correct).
//
// Compiled at image build (Dockerfile step 8) into ple_gpu_gather.so and loaded
// by vllm_ple_mmap.py when VLLM_PLE_GPU_GATHER=1. No host-side prep: the .so
// ships in the image.
//
// The shard table is passed as a device array of base pointers (one per shard,
// each the mmap base of that shard's row region) so one launch covers rows from
// any mix of shards.
//
// Safety: every id is checked against [0, vocab) and against a missing
// (null-base) shard before any dereference. An invalid id zero-fills its output
// row and bumps *oob_count instead of reading an arbitrary host address —
// checked in-kernel so it is capture-safe and costs one compare per row.
// The in-kernel check relies on a load-time invariant: every present shard
// holds exactly its expected row count (enforced by _setup_table), so no
// in-vocab id can land past a short middle shard's mapping.

#include <cstdint>
#include <cuda_runtime.h>

extern "C" {

__global__ void ple_gather_kernel(const uint8_t* const* shard_bases,
                                  int64_t shard_size,
                                  int64_t vocab,
                                  const int64_t* ids,
                                  uint8_t* out,
                                  int64_t row_bytes,
                                  int64_t n_rows,
                                  unsigned long long* oob_count) {
    int64_t row = blockIdx.x;
    int64_t id = ids[row];
    uint8_t* dst = out + row * row_bytes;
    const uint8_t* base =
        (id >= 0 && id < vocab) ? shard_bases[id / shard_size] : nullptr;
    if (base == nullptr) {
        for (int64_t j = threadIdx.x; j < row_bytes; j += blockDim.x) dst[j] = 0;
        // System scope: the counter is pinned host memory read by the CPU;
        // device-scope atomics on host-visible memory are not guaranteed
        // atomic or host-visible by the CUDA memory model.
        if (oob_count != nullptr && threadIdx.x == 0)
            atomicAdd_system(oob_count, 1ULL);
        return;
    }
    const uint8_t* src = base + (id % shard_size) * row_bytes;
    // 128 threads stride a 160-byte row; wider rows also covered.
    for (int64_t j = threadIdx.x; j < row_bytes; j += blockDim.x) dst[j] = src[j];
}

// Launch wrapper: everything is passed as plain integers/pointers so Python can
// call through ctypes with no torch C++ extension build dependency.
int ple_gpu_gather(const void* shard_bases_dev,  // const uint8_t* const* (device)
                   long long shard_size,
                   long long vocab,
                   const void* ids_dev,          // const int64_t* (device)
                   void* out_dev,                // uint8_t* (device)
                   long long row_bytes,
                   long long n_rows,
                   void* stream,
                   void* oob_count) {            // 1 uint64, nullptr = off
    if (n_rows <= 0) return 0;
    if (n_rows > 0x7fffffffLL) return (int)cudaErrorInvalidValue;
    // Drain any pending non-sticky error left by unrelated prior work so the
    // post-launch check below is attributable to THIS launch. (Sticky errors
    // — illegal address etc. — survive this and poison the context anyway.)
    (void)cudaGetLastError();
    ple_gather_kernel<<<(unsigned)n_rows, 128, 0, (cudaStream_t)stream>>>(
        (const uint8_t* const*)shard_bases_dev, shard_size, vocab,
        (const int64_t*)ids_dev, (uint8_t*)out_dev, row_bytes, n_rows,
        (unsigned long long*)oob_count);
    return (int)cudaGetLastError();
}

}  // extern "C"
