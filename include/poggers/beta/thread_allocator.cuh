#ifndef BETA_THREAD_ALLOCATOR
#define BETA_THREAD_ALLOCATOR
// Betta, the block-based extending-tree thread allocaotor, made by Hunter
// McCoy (hunter@cs.utah.edu) Copyright (C) 2023 by Hunter McCoy

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without l> imitation the
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
//  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
//  OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

// The alloc table is an array of uint64_t, uint64_t pairs that store

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
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/representations/representation_helpers.cuh>
#include <vector>

#include "assert.h"
#include "stdio.h"

// These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#define BETA_BLOCK_DEBUG 0

namespace beta {

namespace allocators {

// V2 of the block
// blocks now hand out relative offsets in the range of 0-4095.
// the global offset is determined entirely by the local offset.
struct block {
  uint64_t internal_offset;
  uint64_t_bitarr manager_bits;
  uint64_t_bitarr alloc_bits[64];

  __device__ void init() {
    manager_bits.bits = ~(0ULL);
    for (int i = 0; i < 64; i++) {
      alloc_bits[i].bits = ~(0ULL);
    }
    // at some point work on this
    internal_offset = ~0ULL;
  }

  // helper functions

  // frees must succeed - precondition - fail on double free but print error.
  // uint64_t must be clipped ahead of time. 0 - 4096
  __device__ bool block_free(uint64_t offset) {
    int upper_bit = (offset) / 64;

    int lower_bit = (offset) % 64;

    if (upper_bit > 63 || lower_bit > 63) {
      printf("Free bug - upper %d lower %d\n", upper_bit, lower_bit);
    }

    // uint64_t my_mask = (1ULL << lower_bit);

    uint64_t old_lower_bits = alloc_bits[upper_bit].set_index(lower_bit);

    if (old_lower_bits & SET_BIT_MASK(lower_bit)) {
#if BETA_BLOCK_DEBUG
      printf("Double free bug\n");
#endif
      return false;
    }

    // if we set a bit and 63 were set before, we have finished the row
    // need to set the upper metadata bit.
    if (__popcll(old_lower_bits) == 63) {
      old_lower_bits =
          atomicExch((unsigned long long int *)&alloc_bits[upper_bit], ~0ULL);

#if BETA_BLOCK_DEBUG
      if (old_lower_bits != ~0ULL) {
        printf("Bug in conversion\n");
      }
#endif

      uint64_t old_upper_bits = manager_bits.set_index(upper_bit);

      if (old_upper_bits & SET_BIT_MASK(upper_bit)) {
#if BETA_BLOCK_DEBUG
        printf("failed to reclaim bit %d\n", upper_bit);
#endif
        return false;
      } else {
#if BETA_BLOCK_DEBUG
        printf("Returned %d\n", upper_bit);
#endif
        return __popcll(old_upper_bits) == 63;
      }
    }

    return false;
  }

  __device__ bool is_full_atomic() {
    return __popcll(atomicOr((unsigned long long int *)&manager_bits, 0ULL)) ==
           64;
  }

  // should this loop?
  // nah
  // if failure, retry at higher level.
  __device__ uint64_t block_malloc(cg::coalesced_group &active_threads,
                                   uint64_t &remainder) {
    // if (active_threads.thread_rank() == 0){
    // 	printf("Warp %d has entered the selection process!\n", threadIdx.x/32);
    // }

#if SLAB_GLOBAL_LOADING

    manager_bits.global_load_this();

#endif

    int upper_index;

    while (active_threads.thread_rank() == 0) {
      upper_index = manager_bits.get_random_active_bit();

      if (upper_index == -1) break;

      if (manager_bits.unset_index(upper_index) & SET_BIT_MASK(upper_index))
        break;

      // if (active_threads.thread_rank() == 0){
      // 	printf("Warp %d has failed to claim index %d!\n",
      // threadIdx.x/32, upper_index);
      // }

      // printf("Stuck in main bit_malloc_v2\n");
    }

    upper_index = active_threads.shfl(upper_index, 0);

    if (upper_index == -1) return false;

#if BETA_BLOCK_DEBUG
    if (active_threads.thread_rank() == 0) {
      printf("Warp %d has claimed index %d!\n", threadIdx.x / 32, upper_index);
    }
#endif

    // stored bits, we are going to use atomicExch to swap those out of the
    // block.
    uint64_t_bitarr bits;

    // leader performs full swap
    if (active_threads.thread_rank() == 0) {
      bits = alloc_bits[upper_index].swap_to_empty();
    }

    bits = active_threads.shfl(bits, 0);

#if BETA_BLOCK_DEBUG
    if (active_threads.thread_rank() == 0) {
      if (__popcll(bits.bits) < active_threads.size()) {
        printf("Warp %d: Not enough allocations: %d for %d threads\n",
               threadIdx.x / 32, __popcll(bits.bits), active_threads.size());
      }
    }
#endif

    int my_index = select_unique_bit(active_threads, bits);

#if BETA_BLOCK_DEBUG
    if (!check_indices(active_threads, my_index)) {
      printf(
          "Team %d with %d threads, Bug in select unique main alloc "
          "index %d\n",
          threadIdx.x / 32, active_threads.size(), my_index);
    }
#endif

    remainder = bits;

    // offset = (internal_offset & ~1ULL)+upper_index*64+my_index;

    // this doesn't occur
    if (remainder != 0ULL & my_index == -1) {
      printf("Bug in selecting bits\n");
    }

    if (my_index == -1) {
      return ~0ULL;
    }

    return my_index + upper_index * 64;
  }

  __device__ void mark_pinned() {
    atomicOr((unsigned long long int *)&internal_offset, 1ULL);
  }

  __device__ void mark_unpinned() {
    atomicAnd((unsigned long long int *)&internal_offset, ~1ULL);
  }

  // returns true when unpinned
  __device__ bool atomic_check_unpinned() {
    return ((poggers::utils::ldca(&internal_offset) & 1ULL) == 0);
  }

  __device__ uint64_t get_active_bits() {
    uint64_t counter = 0;

    for (int i = 0; i < 64; i++) {
      counter += __popcll(alloc_bits[i]);
    }

    return counter;
  }
};

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard