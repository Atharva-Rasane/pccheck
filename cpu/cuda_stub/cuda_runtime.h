#pragma once

// CPU-only compatibility header used exclusively by cpu/build_pccheck_writer.sh.
// The original PCcheck DRAMAlloc.h uses CUDA only for cudaMallocHost() and
// cudaSuccess. For functional GPU emulation on CPU, preserve the allocation
// boundary with a distinct real CPU allocation without requiring a CUDA SDK.

#include <cstddef>
#include <cstdlib>

using cudaError_t = int;

static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorMemoryAllocation = 2;

inline cudaError_t cudaMallocHost(void **ptr, std::size_t bytes)
{
    if (ptr == nullptr)
        return cudaErrorMemoryAllocation;

    // Use page-aligned host memory. This is a real independent CPU allocation;
    // it does not claim to reproduce CUDA page-locking/pinned-memory hardware.
    void *p = nullptr;
    const std::size_t alignment = 4096;
    const int rc = posix_memalign(&p, alignment, bytes == 0 ? alignment : bytes);
    if (rc != 0 || p == nullptr)
    {
        *ptr = nullptr;
        return cudaErrorMemoryAllocation;
    }

    *ptr = p;
    return cudaSuccess;
}

inline cudaError_t cudaFreeHost(void *ptr)
{
    std::free(ptr);
    return cudaSuccess;
}
