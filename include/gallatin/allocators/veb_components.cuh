#ifndef GALLATIN_VEB_SEGMENTS
#define GALLATIN_VEB_SEGMENTS
// A CUDA implementation of the Van Emde Boas tree, made by Hunter McCoy
// (hunter@cs.utah.edu)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without l> imitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so,
// subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial
// portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

#include <gallatin/allocators/config.cuh>
//#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/allocators/murmurhash.cuh>




namespace gallatin {

namespace internals {


//templatized segment that does different sizes for internal layers
//this controls fanout.
//all segments use atomics and ld_acq semantics to ensure coherent operations.

//template specialization determines this.
template <uint size_in_bytes>
struct internal_bitarray{

  static_assert(size_in_bytes == 4 || size_in_bytes == 8 || size_in_bytes == 16);

};

template<>
struct internal_bitarray<4> {

  using my_type = internal_bitarray<4>;

  static const uint n_bits = 32;

  uint32_t data;

  __device__ int ffs(){

    return __ffs(data);

  }

  __device__ int ffs(uint starting_index){

    return __ffs(data & (BITMASK(32) << starting_index));

  }

  __device__ uint popc(){

    return __popc(data);

  }

  //writes to index index, and returns true if successful
  // this is performed atomically
  __device__ bool set_loc_atomic(int index){


    #if GALLATIN_DEBUG

    if (index < 0){

      printf("Index is less than 0\n");
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    return !(atomicOr((unsigned int *)&data, SET_BIT_MASK(index)) & SET_BIT_MASK(index));

  }

  __device__ bool set_loc_atomic(int index, my_type & local_copy){


    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    uint32_t result = atomicOr((unsigned int *)&data, SET_BIT_MASK(index));

    bool was_set = result & SET_BIT_MASK(index);

    local_copy.data = result;

    return !was_set;

  }

  __device__ bool unset_loc_atomic(int index){


    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    return atomicAnd((unsigned int *)&data, ~SET_BIT_MASK(index)) & SET_BIT_MASK(index);

  }

  //variant for internal use that calls unset to update a mutable copy.
  __device__ bool unset_loc_atomic(int index, my_type & local_copy){


    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    uint result = atomicAnd((unsigned int *)&data, ~SET_BIT_MASK(index));

    bool unset_bit = result & SET_BIT_MASK(index);

    
    local_copy.data = result;
  

    return unset_bit;

  }


  __device__ inline internal_bitarray<4> ld_acq(){

    internal_bitarray<4> copy_to_load;
    asm volatile("ld.gpu.acquire.u32 %0, [%1];" : "=r"(copy_to_load.data) : "l"(this));
    
    return copy_to_load;

  }

  //atomically set n_contiguous bits starting from starting_index
  //on failure this rolls back and returns false.
  __device__ bool set_contiguous(uint n_contiguous, uint starting_index){

    #if GALLATIN_DEBUG

    if (n_contiguous + starting_index > n_bits){
      write_global_log(6, n_contiguous, starting_index, n_bits);
    }

    #endif

    uint set_bitmask = BITMASK(n_contiguous) << starting_index;

    uint previous_masked_bits = atomicOr((unsigned int *)&data, set_bitmask) & set_bitmask;

    if (previous_masked_bits == 0) return true;

    //otherwise rollback.

    //unset claimed bits.
    set_bitmask = ~previous_masked_bits;

    atomicAnd((unsigned int *)&data, set_bitmask);

    return false;


  }

  __device__ bool unset_contiguous(uint n_contiguous, uint starting_index){

        #if GALLATIN_DEBUG

    if (n_contiguous + starting_index > n_bits){
      write_global_log(6, n_contiguous, starting_index, n_bits);
    }

    #endif

    uint set_bitmask = ((BITMASK(n_contiguous) << starting_index));

    uint previous_masked_bits = atomicAnd((unsigned int *)&data, ~set_bitmask) & set_bitmask;

    if (__popc(previous_masked_bits) == n_contiguous) return true;

    //otherwise rollback.

    //unset claimed bits.

    //reset the bits I claimed.
    atomicOr((unsigned int *)&data, ~previous_masked_bits);

    return false;


  }

  //claim the first available bit
  //returns -1 if fail.
  __device__ int claim_first(my_type & local_copy, uint starting_index=0){


    local_copy = ld_acq();

    int first_available = local_copy.ffs(starting_index)-1;

    while (first_available != -1){

      if (unset_loc_atomic(first_available, local_copy)) return first_available;

      //always increment first_available.
      first_available = local_copy.ffs(first_available)-1;


    }

    return -1;

  }


  __device__ bool query(uint query_index){


    return data & SET_BIT_MASK(query_index);

  }


  __device__ void print(){

    printf("Segment data %x\n", data);

  }


};

template <>
struct internal_bitarray<8> {

  using my_type = internal_bitarray<8>;

  static const uint n_bits = 64;

  uint64_t data;

  __device__ int ffs(){

    return __ffsll(data);

  }


  __device__ int ffs(uint starting_index){

    return __ffsll(data & (BITMASK(64) << starting_index));

  }

  __device__ uint popc(){

    return __popcll(data);

  }


  __device__ bool set_loc_atomic(int index){

    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    return !(atomicOr((unsigned long long int *)&data, SET_BIT_MASK(index)) & SET_BIT_MASK(index));

  }


  __device__ bool set_loc_atomic(int index, my_type & local_copy){


    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    uint64_t result = atomicOr((unsigned long long int *)&data, SET_BIT_MASK(index));

    bool was_set = result & SET_BIT_MASK(index);

    local_copy.data = result;

    return !was_set;

  }

  __device__ bool unset_loc_atomic(int index){

    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    return atomicAnd((unsigned long long int *)&data, ~SET_BIT_MASK(index)) & SET_BIT_MASK(index);

  }


    //variant for internal use that calls unset to update a mutable copy.
  __device__ bool unset_loc_atomic(int index, my_type & local_copy){


    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    uint64_t result = atomicAnd((unsigned long long int *)&data, ~SET_BIT_MASK(index));

    bool unset_bit = result & SET_BIT_MASK(index);

    
    local_copy.data = result;
  

    return unset_bit;

  }



  __device__ inline internal_bitarray<8> ld_acq(){

    internal_bitarray<8> copy_to_load;
    asm volatile("ld.gpu.acquire.u64 %0, [%1];" : "=l"(copy_to_load.data) : "l"(this));
    
    return copy_to_load;

  }


  //atomically set n_contiguous bits starting from starting_index
  //on failure this rolls back and returns false.
  __device__ bool set_contiguous(uint n_contiguous, uint starting_index){

    #if GALLATIN_DEBUG

    if (n_contiguous + starting_index > n_bits){
      write_global_log(6, n_contiguous, starting_index, n_bits);
    }

    #endif

    uint64_t set_bitmask = BITMASK(n_contiguous) << starting_index;

    uint64_t previous_masked_bits = atomicOr((unsigned long long int *)&data, set_bitmask) & set_bitmask;

    if (previous_masked_bits == 0) return true;

    //otherwise rollback.

    //unset claimed bits.
    set_bitmask = ~previous_masked_bits;

    atomicAnd((unsigned int *)&data, set_bitmask);

    return false;


  }

  __device__ bool unset_contiguous(uint n_contiguous, uint starting_index){

    #if GALLATIN_DEBUG

    if (n_contiguous + starting_index > n_bits){
      write_global_log(6, n_contiguous, starting_index, n_bits);
    }

    #endif


    uint64_t set_bitmask = ((BITMASK(n_contiguous) << starting_index));

    
    uint64_t previous_masked_bits = atomicAnd((unsigned long long int *)&data, ~set_bitmask) & set_bitmask;

    if (__popcll(previous_masked_bits) == n_contiguous) return true;

    //otherwise rollback.

    //unset claimed bits.

    //reset the bits I claimed.
    atomicOr((unsigned long long int *)&data, ~previous_masked_bits);

    return false;


  }

  //claim the first available bit
  //returns -1 if fail.
  __device__ int claim_first(my_type & local_copy, uint starting_index = 0U){


    local_copy = ld_acq();


    int first_available = local_copy.ffs(starting_index)-1;

    while (first_available != -1){


      if (unset_loc_atomic(first_available, local_copy)) return first_available;

      //always increment first_available.
      first_available = local_copy.ffs(first_available)-1;


    }

    return -1;

  }

  __device__ bool query(uint query_index){

    return data & SET_BIT_MASK(query_index);

  }

  __device__ void print(){

    printf("Segment data %lx\n", data);

  }




};

template <>
struct internal_bitarray<16> {

  using my_type = internal_bitarray<16>;

  static const uint n_bits = 128;

  uint64_t first;
  uint64_t second;


  __device__ int ffs(){

    int result_first = __ffsll(first);

    if (result_first) return result_first;

    int result_second = __ffsll(second);

    if (result_second) return result_second+64;

    return 0;

  }

  __device__ int ffs(uint starting_index){

    int result_first =  __ffsll(first & (BITMASK(64) << starting_index));

    if (result_first) return result_first;

    if (starting_index < 64) starting_index = 64;

    int result_second = __ffsll(second & (BITMASK(64) << (starting_index-64)));

    if (result_second){
       return result_second+64;
    }

    return 0;
   

  }

  __device__ uint popc(){

    return __popcll(first) + __popcll(second);

  }

  __device__ bool set_loc_atomic(int index){

    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    if (index < 64){
      return !(atomicOr((unsigned long long int *)&first, SET_BIT_MASK(index)) & SET_BIT_MASK(index));

    }

    return !(atomicOr((unsigned long long int *)&second, SET_BIT_MASK(index-64)) & SET_BIT_MASK(index-64));

  }


  __device__ bool set_loc_atomic(int index, my_type & local_copy){

    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    if (index < 64){

      uint64_t result = atomicOr((unsigned long long int *)&first, SET_BIT_MASK(index));

      bool was_set = result & SET_BIT_MASK(index);

      local_copy.first = result;

      return !was_set;
     

    }

    uint64_t result = atomicOr((unsigned long long int *)&second, SET_BIT_MASK(index-64));

    bool was_set = result & SET_BIT_MASK(index-64);


    local_copy.second = result;
    
    
    return !was_set;

  }




  __device__ bool unset_loc_atomic(int index){

    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    if (index < 64){
      return atomicAnd((unsigned long long int *)&first, ~SET_BIT_MASK(index)) & SET_BIT_MASK(index);

    }

    return atomicAnd((unsigned long long int *)&second, ~SET_BIT_MASK(index-64)) & SET_BIT_MASK(index-64);

  }


  __device__ bool unset_loc_atomic(int index, my_type & local_copy){

    #if GALLATIN_DEBUG

    if (index < 0){
      write_global_log(5, (uint64_t) index);
      return;

    } 

    if (index >= n_bits){

      write_global_log(3, index, n_bits);
      return;

    } 

    #endif

    if (index < 64){

      uint64_t result = atomicAnd((unsigned long long int *)&first, ~SET_BIT_MASK(index));


      bool was_unset = result & SET_BIT_MASK(index);

      local_copy.first = result;

      return was_unset;
     

    }

    uint64_t result = atomicAnd((unsigned long long int *)&second, ~SET_BIT_MASK(index-64));

    bool was_unset = result & SET_BIT_MASK(index-64);

    local_copy.second = result;
    
    
    return was_unset;

  }


  __device__ inline internal_bitarray<16> ld_acq(){

    internal_bitarray<16> copy_to_load;

    asm volatile("ld.gpu.acquire.v2.u64 {%0,%1}, [%2];" : "=l"(copy_to_load.first), "=l"(copy_to_load.second) : "l"(this));
         
    return copy_to_load;

  }



    //atomically set n_contiguous bits starting from starting_index
  //on failure this rolls back and returns false.
  __device__ bool set_contiguous_internal(uint n_contiguous, uint starting_index, const uint64_t * data){


    uint64_t set_bitmask = BITMASK(n_contiguous) << starting_index;

    uint64_t previous_masked_bits = atomicOr((unsigned long long int *)data, set_bitmask) & set_bitmask;

    if (previous_masked_bits == 0) return true;

    //otherwise rollback.

    //unset claimed bits.
    set_bitmask = ~previous_masked_bits;

    atomicAnd((unsigned int *)data, set_bitmask);

    return false;


  }

  __device__ bool unset_contiguous_internal(uint n_contiguous, uint starting_index, const uint64_t * data){


    uint64_t set_bitmask = ((BITMASK(n_contiguous) << starting_index));

    
    uint64_t previous_masked_bits = atomicAnd((unsigned long long int *)data, ~set_bitmask) & set_bitmask;

    if (__popcll(previous_masked_bits) == n_contiguous) return true;

    //otherwise rollback.

    //unset claimed bits.

    //reset the bits I claimed.
    atomicOr((unsigned long long int *)data, ~previous_masked_bits);

    return false;


  }


  __device__ bool set_contiguous(uint n_contiguous, uint starting_index){

    #if GALLATIN_DEBUG

    if (n_contiguous + starting_index > n_bits){
      write_global_log(6, n_contiguous, starting_index, n_bits);
    }

    #endif

    if (starting_index < 64){

      if (starting_index+n_contiguous < 64){

        //one step.
        return set_contiguous_internal(n_contiguous, starting_index, &first);

      } else {

        uint64_t first_n_contig = 64-starting_index;

        if (!set_contiguous_internal(first_n_contig, starting_index,&first)) return false;

        //first_n_contig claimed.

        if (!set_contiguous_internal(n_contiguous-first_n_contig, 0, &second)){
          //roll back.
          unset_contiguous_internal(first_n_contig, starting_index, &first);
          return false;
        }

        return true;

      }

    } else {


      return set_contiguous_internal(n_contiguous, starting_index-64, &second);

    }


  }


  __device__ bool unset_contiguous(uint n_contiguous, uint starting_index){

    #if GALLATIN_DEBUG

    if (n_contiguous + starting_index > n_bits){
      write_global_log(6, n_contiguous, starting_index, n_bits);
    }

    #endif

    if (starting_index < 64){

      if (starting_index+n_contiguous < 64){

        //one step.
        return unset_contiguous_internal(n_contiguous, starting_index, &first);

      } else {

        uint64_t first_n_contig = 64-starting_index;

        if (!unset_contiguous_internal(first_n_contig, starting_index,&first)) return false;

        //first_n_contig claimed.

        if (!unset_contiguous_internal(n_contiguous-first_n_contig, 0, &second)){
          //roll back.
          set_contiguous_internal(first_n_contig, starting_index, &first);
          return false;
        }

        return true;

      }

    } else {


      return unset_contiguous_internal(n_contiguous, starting_index-64, &second);

    }


  }

  //claim the first available bit
  //returns -1 if fail.
  __device__ int claim_first(my_type & local_copy, uint starting_index = 0U){


    local_copy = ld_acq();

    int first_available = local_copy.ffs(starting_index)-1;

  

    while (first_available != -1){

      if (unset_loc_atomic(first_available, local_copy)){

        return first_available;
      }

      //always increment first_available.
      first_available = local_copy.ffs(first_available)-1;


    }

    return -1;

  }

  __device__ bool query(uint query_index){


    if (query_index < 64){
      return first & SET_BIT_MASK(query_index);
    }

    return second & SET_BIT_MASK(query_index-64);

  }


  __device__ void print(){

    printf("Segment data %lx %lx\n", first, second);

  }




};

}  // namespace allocators

}  // namespace gallatin

#endif  // End of VEB guard