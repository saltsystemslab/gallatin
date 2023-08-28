#ifndef EXT_VEB_TREE
#define EXT_VEB_TREE
// A CUDA implementation of the Extending Van Emde Boas tree, made by Hunter
// McCoy (hunter@cs.utah.edu) Copyright (C) 2023 by Hunter McCoy

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without l> imitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so,
//  subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial
//  portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY,
//  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
//  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

// The Extending Van Emde Boas Tree, or EVEB, is a data structure that supports
// efficient data grouping and allocation/deallocation based on use. given a
// target size and a memory chunk size, the tree dynamically pulls/pushes chunks
// to the free list based on usage. the metadata supports up to the maximum size
// passed in, and persists so that the true structure does not mutate over the
// runtime.

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/allocators/sub_veb.cuh>
#include <poggers/allocators/veb.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

#define EXT_VEB_RESTART_CUTOFF 30

#define EXT_VEB_GLOBAL_LOAD 1
#define EXT_VEB_MAX_ATTEMPTS 15

namespace poggers {

namespace allocators {

#define FILL_NEW .8

// planning
// each tree will use bytes bytes for the main allocation.

// universe_size = bytes_per_chunk/alloc_size...

template <typename tree_type>
__global__ void init_subtree_kernel(tree_type *tree) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= tree->max_segments) return;

  tree->counters[tid] = 0;

  tree->memory_segments[tid] = nullptr;

  // no trees start valid.
  tree->active_trees->remove(tid);
}

template <typename tree_type, typename bit_allocator, typename memory_allocator>
__global__ void dev_version_cleanup_kernel(tree_type *tree,
                                           bit_allocator *balloc,
                                           memory_allocator *mem_alloc) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= tree->max_chunks) return;

  if (tree->memory_segments[tid] != nullptr) {
    mem_alloc->free(tree->memory_segments[tid]);

    tree->counters[tid];

    balloc->free(tree->trees[tid].get_mem());

    tree->trees[tid].reset_mem();
  }
}

template <uint64_t bytes_per_chunk, uint64_t alloc_size>
struct extending_veb_allocator {
  using my_type = extending_veb_allocator<bytes_per_chunk, alloc_size>;

  uint64_t max_segments;

  veb_tree *active_trees;

  uint64_t *counters;

  sub_veb_tree *trees;

  void **memory_segments;

  static_assert(bytes_per_chunk % alloc_size == 0);

  static __host__ uint64_t get_max_veb_chunks() {
    return poggers::utils::get_max_chunks<bytes_per_chunk>();
  }

  // boot the tree.
  static __host__ my_type *generate_on_device(uint64_t ext_seed) {
    uint64_t num_allocs_per_segment = bytes_per_chunk / alloc_size;

    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    uint64_t max_chunks = get_max_veb_chunks();

    host_version->max_segments = max_chunks;

    uint64_t size_of_sub_tree =
        sub_veb_tree::get_size_bytes_noarray(num_allocs_per_segment);

    sub_veb_tree *dev_trees;

    cudaMalloc((void **)&dev_trees, max_chunks * size_of_sub_tree);

    uint64_t *dev_counters;
    cudaMalloc((void **)&dev_counters, sizeof(uint64_t) * max_chunks);

    void **ext_memory_segments;

    cudaMalloc((void **)&ext_memory_segments, sizeof(void *) * max_chunks);

    host_version->active_trees =
        veb_tree::generate_on_device(max_chunks, ext_seed);
    host_version->trees = dev_trees;
    host_version->counters = dev_counters;
    host_version->memory_segments = ext_memory_segments;

    printf(
        "Tree with %llu bytes per chunk and %llu alloc size has %llu total "
        "chunks and %llu num_allocs_per_segment\n",
        bytes_per_chunk, alloc_size, max_chunks, num_allocs_per_segment);

    my_type *dev_version;

    cudaMalloc((void **)&dev_version, sizeof(my_type));

    cudaMemcpy(dev_version, host_version, sizeof(my_type),
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    init_subtree_kernel<my_type>
        <<<(max_chunks - 1) / 512 + 1, 512>>>(dev_version);

    cudaFreeHost(host_version);

    cudaDeviceSynchronize();

    return dev_version;
  }

  // free up all memory used - included blocks handed to you by the allocators
  template <typename bit_allocator, typename memory_segment_allocator>
  static __host__ void free_on_device(my_type *dev_version,
                                      bit_allocator *balloc,
                                      memory_segment_allocator *memalloc) {
    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    cudaMemcpy(host_version, dev_version, sizeof(my_type),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    dev_version_cleanup_kernel<my_type, bit_allocator, memory_segment_allocator>
        <<<(host_version->max_segments - 1) / 512 + 1, 512>>>(dev_version,
                                                              balloc, memalloc);

    cudaDeviceSynchronize();

    veb_tree::free_on_device(host_version->active_trees);

    cudaFree(host_version->trees);

    cudaFree(host_version->counters);

    cudaFree(host_version->memory_segments);

    cudaFree(dev_version);
  }

  __device__ void malloc() {
    // pick a first active tree.
  }

  __device__ void free(uint64_t relative_offset) {
    // Free process

    // get tree offset and internal offset
    // free internal offset.

    // if counter is now 0 - free
    // else if counter < threshold, reactivate, tree
  }
};

}  // namespace allocators

}  // namespace poggers

#endif  // End of VEB guard