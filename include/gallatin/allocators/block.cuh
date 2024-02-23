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

// #include <cassert>
// #include <cmath>
// #include <cstdio>
// #include <iostream>

#include <gallatin/allocators/alloc_utils.cuh>


#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

// #include "assert.h"
// #include "stdio.h"

// These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#define GALLATIN_BLOCK_DEBUG 0

#define GALLATIN_BLOCK_TREE_OFFSET 20

namespace gallatin {

namespace allocators {

struct Block {
  
  uint malloc_counter;
  uint free_counter;

  __device__ void init() {
    //f u its gotta be big.
    malloc_counter = 4097UL;
    free_counter = 0UL;
  }

  // helper functions

  // frees must succeed - precondition - fail on double free but print error.
  // uint64_t must be clipped ahead of time. 0 - 4096
  __device__ bool block_free() {
    uint old = atomicAdd((unsigned int *)&free_counter, 1ULL);


    #if GALLATIN_BLOCK_DEBUG

    if (old > 4095) printf("Double free to block: %u frees\n", old+1);

    #endif

    return (old == 4095);
  }

  __device__ bool block_free_multiple(uint num_frees) {
    uint old = atomicAdd((unsigned int *)&free_counter, num_frees);


    #if GALLATIN_BLOCK_DEBUG

    if (old > 4096-num_frees) printf("Double free to block: %u frees\n", old+num_frees);

    #endif

    return (old+num_frees == 4096);
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

  //allow threads to procure multiple allocations simultaneously
  __device__ uint64_t block_malloc_multi_size(cg::coalesced_group &active_threads, uint copies_needed){

    //calculate exclusive sum - if value is less than that, valid

    uint my_group_sum = cg::exclusive_scan(active_threads, copies_needed, cg::plus<uint>());

    //last thread in group has total size and controls atomic

    uint old_count;

    if (active_threads.thread_rank() == active_threads.size()-1){

      old_count = atomicAdd((unsigned int *)&malloc_counter, my_group_sum+copies_needed);

    }

    old_count = active_threads.shfl(old_count, 0);

    uint my_value = old_count + my_group_sum;

    if (my_value + copies_needed <= 4096){

      //example here
      //one thread - malloc set to 4095
      //group sum = 0
      //sum_copies = 1
      //old count = 4095
      //value + copies needed = 4096 - valid!

      //two threads - 2 and 1
      // malloc set to 4094
      // group sum = 2
      // sum + copies = 3
      //thread 1 - 4094 + 2 = 4096

      //thread 3 - 4094+2+1 = 4097 - fail.

      //recoalesce

      cg::coalesced_group successful_threads = cg::coalesced_threads();

      uint excess_allocs = cg::reduce(successful_threads, copies_needed-1, cg::plus<uint>());

      //only reduce excess if necessary
      if (excess_allocs > 0 && successful_threads.thread_rank() == 0){

        //don't need to check free logic, as at least one allocation must be active!
        atomicAdd((unsigned int *)&free_counter, excess_allocs);

      }

      return my_value;



    }


    return ~0ULL;


  }

//   __device__ void reset_block() {
//     uint old = atomicExch((unsigned int *)&free_counter, 0ULL);

// #if GALLATIN_BLOCK_DEBUG

//     if (old != 4096) {
//       printf("Double free issue %u != 4096\n", old);
//     }

// #endif

//     atomicExch((unsigned int *)&malloc_counter, 4097ULL);


//   }


  __device__ void reset_free(){

    uint old = atomicExch((unsigned int *)&free_counter, 0ULL);

    #if GALLATIN_BLOCK_DEBUG

    if (old != 4096) {
      printf("Double free issue %u != 4096\n", old);
    }

    #endif

  }

  //setting
  __device__ void init_malloc(uint16_t tree_size){


    //uint big_tree_size = tree_size;

    uint shifted_tree_size = tree_size << GALLATIN_BLOCK_TREE_OFFSET;

    atomicExch((unsigned int *)&malloc_counter, shifted_tree_size);

  }


  //atomically increment the counter and add the old value
  //this version accounts for the tree size.
  __device__ uint block_malloc_tree(cg::coalesced_group &active_threads){

    uint old_count;

    if (active_threads.thread_rank() == 0) {
      old_count =
          atomicAdd((unsigned int *)&malloc_counter, active_threads.size());
    }

    old_count = active_threads.shfl(old_count, 0);

    return old_count;

  }

   //allow threads to procure multiple allocations simultaneously
  __device__ uint64_t block_malloc_tree_multi_size(cg::coalesced_group &active_threads, uint group_sum){

    //calculate exclusive sum - if value is less than that, valid

    //last thread in group has total size and controls atomic

    uint old_count;

    if (active_threads.thread_rank() == active_threads.size()-1){

      old_count = atomicAdd((unsigned int *)&malloc_counter, group_sum);

    }

    old_count = active_threads.shfl(old_count, active_threads.size()-1);


    return old_count;

  }

  //secondary correction
  __device__ void block_correct_frees(cg::coalesced_group &active_threads, uint copies_needed){


    uint excess_allocs = cg::reduce(active_threads, copies_needed, cg::plus<uint>());

    //only reduce excess if necessary
    if (excess_allocs > 0 && active_threads.thread_rank() == 0){

      //don't need to check free logic, as at least one allocation must be active!
      atomicAdd((unsigned int *)&free_counter, excess_allocs);

    }


  }

  //set the malloc bits in the block to 4096
  //this guarantees that no other threads can allocate
  //does a sanity check that the block is not already in use, which may occur.
  //block comes initialized, so we just need to set malloc counter. 
  //if it fails throw a fit - this shouldn't occur but ya never know.
  __device__ uint malloc_fill_block(uint16_t tree_size){

    // uint old = atomicAdd((unsigned int *)&free_counter, 4095);

    // #if GALLATIN_BLOCK_DEBUG
    // if (old != 0){

    //   printf("Old in fill block is %u\n", old);

    // }

    // #endif

    uint old_merged = atomicAdd((unsigned int *)&malloc_counter, 4096);

    uint old_count = clip_count(old_merged);

    if (!check_valid(old_merged, tree_size)){


      #if GALLATIN_BLOCK_DEBUG
      printf("Catrastrophic failure! Block for contiguous section is in use\n");
      #endif

      asm("trap;");

    }

    //uint64_t amount_to_free = (4095 - (old_count + (old_count == 0)));

    atomicAdd(&free_counter, 4095 - old_count);

    #if GALLATIN_BLOCK_DEBUG

    if (old_count != 0){

      //we fucked up, but it's ok! just need to add to the free counter so the block cycles

      

      printf("Block for full segment already malloced. Not an error but concerning.\n");

     

      //atomicAdd(&free_counter, (4095-old_count));

    }

    #endif

    return old_count;

  }

  //return true if the 
  __device__ bool check_valid(uint old_count, uint16_t tree_size){

    uint block_tree_size = (old_count >> GALLATIN_BLOCK_TREE_OFFSET);

    #if GALLATIN_BLOCK_DEBUG

    if (block_tree_size != tree_size){
      printf("Block has different block tree size: %u != %u", block_tree_size, tree_size);
    }

    #endif

    return (block_tree_size == tree_size);

  }

  __device__ uint64_t extract_count(cg::coalesced_group &active_threads, uint old_count){

    uint true_count = (old_count & BITMASK(GALLATIN_BLOCK_TREE_OFFSET));

    uint my_value = true_count + active_threads.thread_rank();

    if (my_value < 4096) {
      return my_value;
    }

    return ~0ULL;

  }

  __device__ uint64_t extract_count_multi_size(cg::coalesced_group &active_threads, uint old_count, uint group_sum, uint my_size){

    uint true_count = (old_count & BITMASK(GALLATIN_BLOCK_TREE_OFFSET));

    uint my_value = true_count + group_sum;

    return my_value;

    //push check condition outside.
    // if (my_value + my_size <= 4096) {
    //   return my_value;
    // }

    // return ~0ULL;

  }


  __device__ uint64_t clip_count(uint old_count){

    return (old_count & BITMASK(GALLATIN_BLOCK_TREE_OFFSET));

  }


};

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard