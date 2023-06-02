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

//doesn't hurt to have on  ¯\_(ツ)_/¯
#define BETA_BLOCK_DEBUG 1

#define BETA_BLOCK_PIN_BIT 16

namespace beta {

namespace allocators {

struct Block {

  uint malloc_counter;
  uint free_counter;

  __device__ void init() {
    malloc_counter = 0ULL;
    free_counter = 0ULL;
  }


  //pinning occurs before attachment, this should always succeed
  __device__ bool pin(){

    uint old = atomicCAS((unsigned int *)&free_counter, 0U, SET_BIT_MASK(BETA_BLOCK_PIN_BIT));

    if (old != 0U){
      printf("Failed to pin: old is %u\n", old);
    }

    return (old == 0U);

  }


  __device__ uint unpin(){

    return atomicSub((unsigned int *)&free_counter, SET_BIT_MASK(BETA_BLOCK_PIN_BIT));

  }

  //returns true if pinned
  __device__ bool atomic_check_pinned(){

    uint old = atomicCAS((unsigned int *)&free_counter, 0U, 0U);

    return (old & SET_BIT_MASK(BETA_BLOCK_PIN_BIT));


  }

  //unset the pin
  //returns true if the pin 
  __device__ bool unpin_and_check_free(){


    uint old = unpin();

    if (!(old & SET_BIT_MASK(BETA_BLOCK_PIN_BIT))){
      #if BETA_BLOCK_DEBUG

        printf("Failed to unpin: %u\n", old);
      #endif

    }

    uint count = old & ~SET_BIT_MASK(BETA_BLOCK_PIN_BIT);

    //old should never be more than 4096
    //unless pin is set.
    #if BETA_BLOCK_DEBUG
      if (count > 4096){
        printf("Count in block exceeded max possible: %u\n", count);
      }
    #endif

    return (count == 4096);

  }


  __device__ bool free_and_check_unpinned(){

    uint old = block_free();

    if (old & SET_BIT_MASK(BETA_BLOCK_PIN_BIT)) return false;

    //count is 1 less as we have added to it
    uint count = old & ~SET_BIT_MASK(BETA_BLOCK_PIN_BIT);

    #if BETA_BLOCK_DEBUG

    if (count >= 4096){
      printf("Double free in block: true count %u\n", count+1);
    }

    #endif

    return (count == 4095);


  }

  // helper functions

  // frees must succeed - precondition - fail on double free but print error.
  // uint64_t must be clipped ahead of time. 0 - 4096
  __device__ uint block_free() {
    uint old = atomicAdd((unsigned int *)&free_counter, 1ULL);

    return old;
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


  //debug check - has anyone tampered with the block before setup?
  //if true, we know that the blocks are being handled improperly - resetting
  //keeps block static until new segment allocation.
  __device__ bool atomic_check_block(){

    uint old_free = atomicExch((unsigned int *)&free_counter, 0ULL);
    uint old_malloc = atomicExch((unsigned int *)&malloc_counter, 0ULL);

    if (old_free != 0 || old_malloc != 0){
      printf("Block was tampered with %u malloc %u free\n", old_malloc, old_free);
      return true;
    }


    return false;

  }


};

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard