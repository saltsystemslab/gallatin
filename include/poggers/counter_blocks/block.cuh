#ifndef BETA_BLOCK
#define BETA_BLOCK
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

// The alloc table is an array of uint64_t, uint64_t pairs that store

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

#include <poggers/allocators/alloc_utils.cuh>


#include <cooperative_groups.h>

#include <vector>

#include "assert.h"
#include "stdio.h"

// These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#define BETA_BLOCK_DEBUG 1

namespace beta {

namespace allocators {

struct Block {
  uint malloc_counter;
  uint free_counter;

  __device__ void init() {
    malloc_counter = 0ULL;
    free_counter = 0ULL;
  }

  // helper functions

  // frees must succeed - precondition - fail on double free but print error.
  // uint64_t must be clipped ahead of time. 0 - 4096
  __device__ bool block_free() {
    uint old = atomicAdd((unsigned int *)&free_counter, 1ULL);


    #if BETA_BLOCK_DEBUG

    if (old > 4095) printf("Double free to block: %u frees\n", old+1);

    #endif

    return (old == 4095);
  }

  __device__ uint64_t block_malloc(cg::coalesced_group &active_threads) {
    uint old_count;

    if (active_threads.thread_rank() == 0) {
      old_count =
          atomicAdd((unsigned int *)&malloc_counter, active_threads.size());
    }

    old_count = active_threads.shfl(old_count, 0);

    uint my_value = old_count + active_threads.thread_rank();

    if (my_value < 4096) {
      return my_value;
    }

    return ~0ULL;
  }

  __device__ void reset_block() {
    uint old = atomicExch((unsigned int *)&free_counter, 0ULL);

#if BETA_BLOCK_DEBUG

    if (old != 4096) {
      printf("Double free issue %u != 4096\n", old);
    }

#endif

    atomicExch((unsigned int *)&malloc_counter, 0ULL);
  }
};

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard