#ifndef BETA_BLOCK_STORAGE
#define BETA_BLOCK_STORAGE
// Betta, the block-based extending-tree thread allocaotor, made by Hunter McCoy
// (hunter@cs.utah.edu) Copyright (C) 2023 by Hunter McCoy

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

// Block Storage are localized storage for each

// inlcudes
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/allocators/uint64_bitarray.cuh>
#include <poggers/counter_blocks/block.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/representations/representation_helpers.cuh>
#include <vector>

#include "assert.h"
#include "stdio.h"

// These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#define BETA_BLOCK_STORAGE_DEBUG 0

namespace beta {

namespace allocators {

// V2 of the block
// blocks now hand out relative offsets in the range of 0-4095.
// the global offset is determined entirely by the local offset.

// states
// 11--alloced and ready to go
// 01 - alloced
template <int num_backups>
struct block_storage {
  static_assert(num_backups < 64);

  // one bit buffer?
  uint64_t_bitarr slab_markers;

  Block *slab_ptrs[num_backups + 1];

  __device__ void init() {
    slab_markers = 0ULL;

    for (int i = 0; i < num_backups + 1; i++) {
      slab_ptrs[i] = nullptr;
    }
  }

  // ISSUE
  // non primary may be passed in here and non be the primary... - thats bad
  // mkay to rectify need to detect non match and swap out?

  // reserve that you are exclusive manager - unset manager bit
  // todo - only swap with known value of primary - prevent unneccesary swap
  // outs

  __device__ bool pivot_non_primary(int index, Block *old_item) {
    printf("Function pivot_non_primary shouldn't be used - block storage\n");

    if (!(slab_markers.unset_index(index) & SET_BIT_MASK(index))) {
      // claimed by someone else
      return false;
    }
  }

  __device__ int pivot_primary(Block *old_primary) {
    // printf("Thread %d entering pivot\n", threadIdx.x);

    if (!(slab_markers.unset_index(0) & SET_BIT_MASK(0))) {
      return -1;
    }

    __threadfence();

    while (true) {
      int index = slab_markers.get_random_active_bit_nonzero();

      if (index == -1) {
        return -1;
      }

      if (slab_markers.unset_index(index) & SET_BIT_MASK(index)) {
        // legal and very cool
        // other threads must check that they do not receive a nullptr in this
        // rather unstable state.
        uint64_t old =
            atomicExch((unsigned long long int *)&slab_ptrs[index], 0ULL);

        if (atomicCAS((unsigned long long int *)&slab_ptrs[0],
                      (unsigned long long int)old_primary,
                      (unsigned long long int)old) !=
            (unsigned long long int)old_primary) {
          // printf("This is the fail case\n");

          slab_markers.set_index(0);
          attach_new_buffer(index, (Block *)old);
          return -1;
        }

        slab_markers.set_index(0);

        // printf("Index %d successfully swapped to primary\n", index);
        return index;

      } else {
        // someone else has grabbed an index for alloc - this should be
        // impossible?
        printf("This may be undefined behavior\n");
      }
    }
  }

  // potential bug - pivot allows for multiple to succeed
  // accidentally swapping valud blocks
  // adding this bit check appears to solve - check extensively
  __device__ Block *get_primary() {
    Block *ret = slab_ptrs[0];

    // if (ret == nullptr || !(slab_markers.bits & SET_BIT_MASK(0))){
    // 	return get_non_primary();
    // }

    if ((uint64_t)ret == 0x1) {
      printf("Bug inside get primary\n");
    }

    return ret;

    // int valid_index = poggers::utils::get_smid();
  }

  __device__ Block *get_non_primary() {
    // multiple threads in the same warp should maintain coalescing.

    int index = slab_markers.get_random_active_bit_warp();

    if (index == -1) return nullptr;

    // if ( (uint64_t) slab_ptrs[index] == 0x1){
    // 	printf("Bug in get_non_primary\n");
    // }

    return slab_ptrs[index];
  }

  __device__ bool attach_new_buffer(int index, Block *new_buffer) {
    uint64_t old = atomicCAS((unsigned long long int *)&slab_ptrs[index], 0ULL,
                             (unsigned long long int)new_buffer);

    if (old != 0ULL) {
      printf("%d Exchanged for an already set buffer: %llx exchanged\n", index,
             old);

      return false;
    }

    if (slab_markers.set_index(index) & SET_BIT_MASK(index)) {
      printf("Error attaching: index %d already set\n", index);

      return false;
    }

    return true;
  }

  template <typename block_allocator>
  __device__ void init_with_allocators_memory(block_allocator *balloc) {
    // boot myself to clear memory
    init();

    for (int i = 0; i < num_backups + 1; i++) {
      uint64_t slab_offset = balloc->get_offset();

      Block *slab = (Block *)balloc->get_mem_from_offset(slab_offset);

      if (slab == nullptr) {
        printf("Failed to load slab from allocator\n");
        return;
      }

      // uint64_t offset = slab_offset*ext_offset;

      // if (offset == memory_allocator::fail()){
      // 	balloc->free(slab);
      // 	printf("Fail to claim memory for slab\n");

      // }

      // don't forget to actually boot memory lol
      slab->init();

      // slab->attach_allocation(offset);

      attach_new_buffer(i, slab);
    }
  }
};

template <int num_blocks, typename block_allocator>
__global__ void smid_pinned_block_init_storage_char(
    block_allocator *balloc, block_storage<num_blocks> *storages,
    int num_storages) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= num_storages) return;

  storages[tid].init_with_allocators_memory(balloc);
}

template <int num_blocks>
struct pinned_block_storages {
  using my_type = pinned_block_storages<num_blocks>;

  using pinned_type = block_storage<num_blocks>;

  pinned_type *storages;

  template <typename block_allocator>
  static __host__ my_type *generate_on_device(int device,
                                              block_allocator *balloc) {
    my_type *host_storage;

    cudaMallocHost((void **)&host_storage, sizeof(my_type));

    pinned_type *dev_storages;

    int num_storages =
        poggers::utils::get_num_streaming_multiprocessors(device);

    printf("Booting up %d storages, %llu bytes\n", num_storages,
           sizeof(pinned_type) * num_storages);
    cudaMalloc((void **)&dev_storages, sizeof(pinned_type) * num_storages);

    smid_pinned_block_init_storage_char<num_blocks, block_allocator>
        <<<(num_storages - 1) / 256 + 1, 256>>>(balloc, dev_storages,
                                                num_storages);

    cudaDeviceSynchronize();

    host_storage->storages = dev_storages;

    my_type *dev_ptr;

    cudaMalloc((void **)&dev_ptr, sizeof(my_type));

    cudaMemcpy(dev_ptr, host_storage, sizeof(my_type), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaFreeHost(host_storage);

    return dev_ptr;
  }

  // if you don't specify we go on device 0.
  template <typename block_allocator>
  static __host__ my_type *generate_on_device(block_allocator *balloc) {
    return my_type::generate_on_device(0, balloc);
  }

  static __host__ void free_on_device(my_type *dev_storage) {
    my_type *host_storage;

    cudaMallocHost((void **)&host_storage, sizeof(my_type));

    cudaMemcpy(host_storage, dev_storage, sizeof(my_type),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(dev_storage);

    cudaFree(host_storage->storages);

    cudaFreeHost(host_storage);

    return;
  }

  __device__ pinned_type *get_pinned_blocks() {
    return &storages[poggers::utils::get_smid()];
  }

  static __host__ my_type *generate_on_device() {
    my_type *host_storage;

    cudaMallocHost((void **)&host_storage, sizeof(my_type));

    pinned_type *dev_storages;

    int num_storages = poggers::utils::get_num_streaming_multiprocessors(0);

    printf("Booting up %d storages, %llu bytes\n", num_storages,
           sizeof(pinned_type) * num_storages);
    cudaMalloc((void **)&dev_storages, sizeof(pinned_type) * num_storages);

    // smid_pinned_block_init_storage_char<num_blocks,
    // block_allocator><<<(num_storages-1)/256+1,256>>>(balloc, ext_offset,
    // dev_storages, num_storages);

    cudaDeviceSynchronize();

    host_storage->storages = dev_storages;

    my_type *dev_ptr;

    cudaMalloc((void **)&dev_ptr, sizeof(my_type));

    cudaMemcpy(dev_ptr, host_storage, sizeof(my_type), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaFreeHost(host_storage);

    return dev_ptr;
  }
};

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard