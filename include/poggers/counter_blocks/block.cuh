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
#define BETA_BLOCK_DEBUG 0

#define BETA_BLOCK_TREE_OFFSET 20

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


  __device__ void reset_free(){

    uint old = atomicExch((unsigned int *)&free_counter, 0ULL);

    #if BETA_BLOCK_DEBUG

    if (old != 4096) {
      printf("Double free issue %u != 4096\n", old);
    }

    #endif

  }

  //setting
  __device__ void init_malloc(uint16_t tree_size){


    //uint big_tree_size = tree_size;

    uint shifted_tree_size = tree_size << BETA_BLOCK_TREE_OFFSET;

    atomicExch((unsigned int *)&malloc_counter, shifted_tree_size);

  }


  //atomically increment the counter and add the old value
  __device__ uint block_malloc_tree(cg::coalesced_group &active_threads){

    uint old_count;

    if (active_threads.thread_rank() == 0) {
      old_count =
          atomicAdd((unsigned int *)&malloc_counter, active_threads.size());
    }

    old_count = active_threads.shfl(old_count, 0);

    return old_count;

  }

  //set the malloc bits in the block to 4096
  //this guarantees that no other threads can allocate
  //does a sanity check that the block is not already in use, which may occur.
  //block comes initialized, so we just need to set malloc counter. 
  //if it fails throw a fit - this shouldn't occur but ya never know.
  __device__ uint malloc_fill_block(uint16_t tree_size){

    // uint old = atomicAdd((unsigned int *)&free_counter, 4095);

    // #if BETA_BLOCK_DEBUG
    // if (old != 0){

    //   printf("Old in fill block is %u\n", old);

    // }

    // #endif

    uint old_merged = atomicAdd((unsigned int *)&malloc_counter, 4096);

    uint old_count = clip_count(old_merged);

    if (!check_valid(old_merged, tree_size)){


      #if BETA_BLOCK_DEBUG
      printf("Catrastrophic failure! Block for contiguous section is in use\n");
      #endif

      asm("trap;");

    }

    if (old_count != 0){

      //we fucked up, but it's ok! just need to add to the free counter so the block cycles

      #if BETA_BLOCK_DEBUG

      printf("Block for full segment already malloced. Not an error but concerning.\n")

      #endif

      atomicAdd(&free_counter, (4095-old_count));

    }

    return old_count;

  }

  //return true if the 
  __device__ bool check_valid(uint old_count, uint16_t tree_size){

    uint block_tree_size = (old_count >> BETA_BLOCK_TREE_OFFSET);

    #if BETA_BLOCK_DEBUG

    if (block_tree_size != tree_size){
      printf("Block has different block tree size: %u != %u", block_tree_size, tree_size);
    }

    #endif

    return (block_tree_size == tree_size);

  }

  __device__ uint64_t extract_count(cg::coalesced_group &active_threads, uint old_count){

    uint true_count = (old_count & BITMASK(BETA_BLOCK_TREE_OFFSET));

    uint my_value = true_count + active_threads.thread_rank();

    if (my_value < 4096) {
      return my_value;
    }

    return ~0ULL;

  }


  __device__ uint64_t clip_count(uint old_count){

    return (old_count & BITMASK(BETA_BLOCK_TREE_OFFSET));

  }


};

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard