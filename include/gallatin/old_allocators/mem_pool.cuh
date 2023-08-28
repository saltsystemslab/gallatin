#ifndef BETA_MEM_POOL
#define BETA_MEM_POOL

// dummy mem pool for benchmarking.
// exposes the same functions as the allocator.

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <poggers/allocators/alloc_memory_table.cuh>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/allocators/block_storage.cuh>
#include <poggers/allocators/ext_veb_nosize.cuh>
#include <poggers/allocators/offset_slab.cuh>
#include <poggers/allocators/one_size_allocator.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

namespace poggers {

namespace allocators {

#define REQUEST_BLOCK_MAX_ATTEMPTS 1

// alloc table associates chunks of memory with trees

// using uint16_t as there shouldn't be that many trees.

// register atomically inserst tree num, or registers memory from segment_tree.

using namespace poggers::utils;

struct mem_pool {
  using my_type = mem_pool;

  uint64_t counter;
  uint64_t max_allocs;
  uint64_t size;

  char *memory;

  static __host__ my_type *generate_on_device(uint64_t max_bytes,
                                              uint64_t alloc_size) {
    my_type *host_version = get_host_version<my_type>();

    // plug in to get max chunks

    uint64_t max_allocs = max_bytes / alloc_size;

    uint64_t bytes_to_allocate = alloc_size * max_allocs;

    // uint64_t max_chunks = get_max_chunks<bytes_per_segment>(max_bytes);

    // host_version->segment_tree = veb_tree::generate_on_device(max_chunks,
    // seed);

    // one_size_allocator::generate_on_device(max_chunks, bytes_per_segment,
    // seed);

    char *extra_memory;

    cudaMalloc((void **)&extra_memory, bytes_to_allocate);

    host_version->counter = 0;
    host_version->max_allocs = max_allocs;
    host_version->memory = extra_memory;
    host_version->size = alloc_size;

    return move_to_device(host_version);
  }

  static __host__ void free_on_device(my_type *dev_version) {
    // this frees dev version.
    my_type *host_version = move_to_host<my_type>(dev_version);

    cudaFree(host_version->memory);

    cudaFreeHost(host_version);

    return;
  }

  __device__ void *malloc() {
    uint64_t my_count = atomicAdd((unsigned long long int *)&counter, 1ULL);

    if (my_count >= max_allocs) {
      return nullptr;
    }

    return (void *)(memory + my_count * size);
  }

  __device__ void free(void *alloc) { return; }

  uint64_t get_offset_from_ptr(void *ext_ptr) {
    char *pointer = (char *)ext_ptr;

    uint64_t raw_offset = (pointer - memory);

    return raw_offset / size;
  }
};

}  // namespace allocators

}  // namespace poggers

#endif  // End of VEB guard