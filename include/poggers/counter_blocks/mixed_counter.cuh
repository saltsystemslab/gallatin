#ifndef GALLATIN_MIXED_COUNTER
#define GALLATIN_MIXED_COUNTER
// The Gallatin Allocator, made by Hunter McCoy
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

// The mixed counter includes logic for performing a specialized simultaneous (add+subtract) 
// on a pair of uint counters. This is needed for the Gallatin intra-segment queueing system
// Basically, precondition that valid upper+lower is always increasing - this ensures unique offsets are handed out.

//operations look like this:

// Malloc:
// object is upper_counter + lower counter
// 1. To acquire the semaphore, atomicAdd(upper_counter) - perform coalesced read to acq lower counter.
// 2. If < cutoff, done - Queue value = upper_counter + lower_counter.
// 2.1 - else, solo decrement of the counter.


//Frees
//1. increment ext counter to determine return slot
//2. atomicCAS back in
//3. simultaneous dec/add to return counter.

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/alloc_utils.cuh>

namespace gallatin {

namespace utils {


#define MIXED_LOWER_BIT

#define GAL_MIXED_LOWER_MASK (1ULL << 32)-1

#define GAL_MIXED_UPPER_MASK (GAL_MIXED_LOWER_MASK) << 32


//atomicCAS version: maintain max cap + swap - must increment swap + 1 AND
// increment + 1

struct mixed_counter {

  union {

    uint64_t full_counter;
    uint32_t individual_counters [2];

  };

  __host__ __device__ void init(uint live_cap){

    individual_counters[0] = 0;
    individual_counters[1] = live_cap;

  }

  __device__ void atomicInit(uint live_cap){

    mixed_counter copy_counter(0, live_cap);

    atomicExch((unsigned long long int *)&this->full_counter, copy_counter.full_counter);

  }

  __host__ __device__ explicit mixed_counter(uint64_t x) {

    full_counter = x;

  }

  __host__ __device__ explicit mixed_counter(uint live_count, uint counter){


    individual_counters[0] = counter;
    individual_counters[1] = live_count;


  }

  __device__ uint get_counter(){
    return individual_counters[0];
  }

  __device__ uint get_live_count(){
    return individual_counters[1];
  }

  __device__ void reset(){

    //swap both counters
    atomicExch((unsigned long long int *) &full_counter, 0ULL);

  }


  //return an index for this thread or ~0ULL if not available.
  // live cap is the current cap of the system, defined based on tree size in Gallatin.
  __device__ uint64_t count_and_increment(){


    //global read the current value and determine if the current system can suppor
    // a pull
    mixed_counter read_value = mixed_counter(poggers::utils::ldca((uint64_t * ) this));


    uint current_index = read_value.get_counter();
    uint current_live_count = read_value.get_live_count();


    if (current_live_count == 0){

      return ~0ULL;

    }


    mixed_counter replace_count(current_live_count-1, current_index+1);
  

    mixed_counter old_status(atomicCAS((unsigned long long int *)this, read_value.full_counter, replace_count.full_counter));


    if (old_status.full_counter == read_value.full_counter){

      //success!
      return current_index;

    }


    return ~0ULL;

  }

  //return an index for this thread or ~0ULL if not available.
  // live cap is the current cap of the system, defined based on tree size in Gallatin.
  __device__ uint64_t count_and_increment_check_last(bool & last){

    last = false;

    //global read the current value and determine if the current system can suppor
    // a pull
    mixed_counter read_value = mixed_counter(poggers::utils::ldca((uint64_t * ) this));


    uint current_index = read_value.get_counter();
    uint current_live_count = read_value.get_live_count();


    if (current_live_count == 0){

      return ~0ULL;

    }


    mixed_counter replace_count(current_live_count-1, current_index+1);
  

    mixed_counter old_status(atomicCAS((unsigned long long int *)this, read_value.full_counter, replace_count.full_counter));


    if (old_status.full_counter == read_value.full_counter){

      last = (current_live_count == 1);

      //success!
      return current_index;

    }


    return ~0ULL;

  }


  __device__ uint64_t release(){

    //range from 0-num_blocks-1
    return atomicAdd((unsigned int *)&individual_counters[1], 1U);

  }

  // __device__ uint decrement_upper(){

  //   return atomicSub(&lower, 1U);

  // }


  // __device__ uint increment_lower(){

  //   return atomicAdd(&lower, 1U);

  // }

  // __device__ uint decrement_lower(){

  //   return atomicSub(&lower, 1U);

  // }

  // __device__ uint64_t increment_both(){

  //   return atomicAdd((unsigned long long int *)this, (1ULL << 32) + 1);

  // }

  // __device__ mixed_counter inc_lower_read_both(){

  //   uint64_t result = atomicAdd((unsigned long long int *)this, 1ULL);

  //   return ((mixed_counter *) &result)[0];
  // }

  // __device__ uint64_t dec_lower_inc_upper(){

    
  //   return atomicAdd((unsigned long long int *)this, (1ULL << 32)-1); 
  // }


  // __device__ uint decode_upper(uint64_t mixed){
  //   return mixed & GAL_MIXED_UPPER_MASK;
  // }


  // __device__ uint decode_lower(uint64_t mixed){
  //   return mixed & GAL_MIXED_LOWER_MASK;
  // }

};

  //if CUDA does something weird with padding this isn't going to work.
  //we need the casting black magic
  static_assert(sizeof(mixed_counter) == 8);

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard