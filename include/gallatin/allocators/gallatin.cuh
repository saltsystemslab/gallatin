#ifndef GALLATIN_ALLOCATOR
#define GALLATIN_ALLOCATOR
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


/*** ABOUT
  Gallatin is a generic vEB-based GPU allocator that allows for individual
threads on the GPU to perform allocations.

When building the type, the template arguments are as follows:

* bytes_per_segment: Bytes per segment. Defualt 16 megabytes
  - This is the granularity that GPU memory is sliced into.
* uint64_t smallest: Number of bytes in the smallest slice size. Defualt 16 bytes
* uint64_t biggest: Number of bytes in the largest slice size. Defualt 4 kilobytes

Based on these template parameters, the number of trees
 and intermediate slice sizes are determined at compile-time.


Usage: 
  Gallatin must be constructed and destructed by host.
To do so, call Gallatin<template_args>::generate_on_device()
and supply the # of bytes to be made allocable, along with a random seed.
This function returns a handle to the allocator that can be used in device kernels.

To free device memory at the end of execution, call

 Gallatin<template_args>::free_on_device(your_pointer);

This will free the associated device memory, including all memory that has been handed out.
THIS WILL NOT WIPE DEVICE POINTERS. 
Using memory allocated by Gallatin after this call is undefined behavior.

Inside of a kernel, you must pass a pointer to the allocator.
You can then allocate new memory with the malloc method:

  void * alloc_ptr->malloc(uint64_t num_bytes)

This returns a void * type of at least num_bytes(), or nullptr if no allocation is available.

Once the memory is no longer needed, it can be returned via

  void alloc_ptr->free(void * memory_ptr);

The pointer returned must be the same address that was returned - 
  trying to free a different address can result in undefined behavior.

*/




// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <gallatin/allocators/config.cuh>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/allocators/block.cuh>
#include <gallatin/allocators/block_storage.cuh>
#include <gallatin/allocators/segment.cuh>
#include <gallatin/allocators/murmurhash.cuh>
#include <gallatin/allocators/veb.cuh>


namespace gallatin {

namespace allocators {

using namespace gallatin::internals;


template <typename allocator>
__global__ void init_blocks(allocator * gallatin, uint n_trees){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_trees) return;

  uint my_max_blocks = gallatin->tree_storage[tid]->n_blocks;

  for (uint i = 0; i < my_max_blocks; i++){

    //init in fail state.
    uint16_t block = allocator::block_fail();
    uint32_t segment = allocator::segment_fail();
    if (!gallatin->get_new_block(tid,block,segment)){

      #if GALLATIN_DEBUG
      write_global_log(16, i, my_max_blocks, tid);
      #endif

    } else {

      uint32_t packed = gallatin->tree_storage[tid]->pack_segment_block(segment, block);
      gallatin->tree_storage[tid]->set(i, packed);

    }



  }

}


__host__ constexpr uint host_get_first_bit_bigger(uint64_t counter){
  return 63 - __builtin_clzll(counter) + (__builtin_popcountll(counter) != 1);
}


__host__ constexpr uint calculate_max_size(uint biggest, uint smallest){



  return host_get_first_bit_bigger(biggest)-host_get_first_bit_bigger(smallest)+1;
}

// main allocator structure
// template arguments are
//  - size of each segment in bytes
//  - size of smallest segment allocatable
//  - size of largest segment allocatable
template <uint32_t bytes_per_segment, uint32_t smallest, uint32_t biggest, uint segmentVebSize, uint blockVebSize>
struct Gallatin {


  using my_type = Gallatin<bytes_per_segment, smallest, biggest, segmentVebSize, blockVebSize>;
  using segment_tree_type = veb<segmentVebSize>;
  using block_tree_type = veb<blockVebSize>;

  //all nice powers of two please
  static_assert(__builtin_popcount(smallest) == 1 && __builtin_popcount(biggest) == 1 && __builtin_popcount(bytes_per_segment) == 1);

  //and objects should fit into segments.
  static_assert(bytes_per_segment >= biggest*4096);


  static constexpr uint n_trees = calculate_max_size(biggest, smallest);

  static constexpr uint max_blocks_per_segment = bytes_per_segment/(4096*smallest);

  static constexpr uint min_size = host_get_first_bit_bigger(smallest);

  using segment_type = segment<max_blocks_per_segment, smallest, bytes_per_segment>;

  uint n_segments;
  uint64_t total_mem;

  uint32_t tree_locks;

  segment_tree_type * segment_tree;

  block_tree_type * block_trees[n_trees];

  block_storage * tree_storage[n_trees];

  segment_type * segments;

  char * memory_base;


  //get size of an object in the form 2^x.
  // returns the x from above, so 2 returns 1
  // promotes object to next power of 2 if not matched.
  __host__ __device__ static uint get_size_p2(uint size){

    return gallatin::utils::get_first_bit_bigger(size);

  }

  __device__ static uint get_size_p2(uint64_t size){
    return gallatin::utils::get_first_bit_bigger(size);
  }


  // generate the allocator on device
  // this takes in the number of bytes owned by the allocator (does not include
  // the space of the allocator itself.)
  static __host__ my_type *generate_on_device(uint64_t max_bytes,
                                              uint64_t seed, bool print_info=true, bool running_calloc=false) {
    



    my_type * host_version = gallatin::utils::get_host_version<my_type>();


    host_version->n_segments = gallatin::utils::get_max_chunks<bytes_per_segment>(max_bytes);

    uint max_chunks = host_version->n_segments;

    host_version->total_mem = max_bytes;

    host_version->segment_tree = segment_tree_type::generate_on_device(max_chunks);

    host_version->tree_locks = 0;

    for (uint i = 0; i < n_trees; i++){
      host_version->block_trees[i] = block_tree_type::generate_on_device_cleared(max_chunks);
    }


    //for now just define these as tuned constants.
    uint block_segment_sizes[] = {128, 64, 32, 16, 8,6,6,6,6,6};

    for (uint i = 0; i < n_trees; i++){
      host_version->tree_storage[i]= block_storage::generate_on_device(block_segment_sizes[i]);
    }

    host_version->segments = segment_type::generate_on_device(max_chunks);

    host_version->memory_base = gallatin::utils::get_device_version<char>(max_bytes);

    cudaMemset(host_version->memory_base, 0, max_bytes);

    my_type * device_version = gallatin::utils::move_to_device<my_type>(host_version);
    

    init_blocks<my_type><<<(n_trees-1)/256+1,256>>>(device_version, n_trees);

    GPUErrorCheck(cudaDeviceSynchronize());

    return device_version;

  }



  // return the index of the largest bit set
  static __host__ __device__ int get_first_bit_bigger(uint64_t counter) {
    return gallatin::utils::get_first_bit_bigger(counter);
  }

  // get number of sub trees live
  constexpr static __host__ __device__ int get_num_trees() {
    return get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest) + 1;
  }

  // return memory_base used to device
  static __host__ void free_on_device(my_type *dev_version) {
    // this frees dev version.
    my_type *host_version = gallatin::utils::move_to_host<my_type>(dev_version);

    //free segment_tree
    segment_tree_type::free_on_device(host_version->segment_tree);

    for (uint i = 0; i < n_trees; i++){
      block_tree_type::free_on_device(host_version->block_trees[i]);
    }

    for (uint i = 0; i < n_trees; i++){
      block_storage::free_on_device(host_version->tree_storage[i]);
    }

    segment_type::free_on_device(host_version->segments);

    cudaFree(host_version->memory_base);



    cudaFreeHost(host_version);
  }


  //alloc functions
  //segment comes initialized.
  //attempts to pull from the same segment as the last block if possible
  __device__ bool get_new_block(const uint16_t size, uint16_t & block, uint32_t & segment){


  int n_attempts = 0;

  while(n_attempts < GALLATIN_MAX_NEW_BLOCK_ATTEMPTS){

    if (segment != segment_fail()){

      //attempt requisition of a new block from the segment.

      bool last = false; 

      block = segments[segment].reserve_block(size, last);

      if (block != segment_type::fail_size()){

        if (last){
          block_trees[size]->remove(segment);
        }

        //segments[segment].setup_block(block, size);

        __threadfence();

        return true;

      } else {

        //occurs when none available || bad size.
        //can always safely remove. 
        block_trees[size]->remove(segment);
      }

    }

    //dead state - no segment available.
    if (!find_valid_segment(size, segment)) return false;

    //new segment found, retry.
    n_attempts++;

  }

  //timed out.
  return false;

  }

  __device__ bool acq_tree_lock(uint16_t size){


    uint32_t bitmask = (uint32_t) SET_BIT_MASK(size);

    return ((atomicOr(&tree_locks, bitmask) & bitmask) == 0);

  }

  __device__ void release_tree_lock(uint16_t size){

    uint32_t unset_bitmask = ~((uint32_t) SET_BIT_MASK(size));

    atomicAnd(&tree_locks, unset_bitmask);

  }

  //find new segment at size to claim
  __device__ bool find_valid_segment(const uint16_t size, uint32_t & segment){


    int n_attempts = 0;

    while (n_attempts < GALLATIN_MAX_NEW_SEGMENT_ATTEMPTS){


      segment = block_trees[size]->find_random();

      if (segment != segment_fail()){

        return true;
      }


      //acquire lock.
      if (!acq_tree_lock(size)){
        n_attempts++;
        __nanosleep(GALLATIN_SEGMENT_SLEEP_DURATION);
        __threadfence();
        continue;
      }

      //claim first

      segment = segment_tree->claim_first(0);

      if (segment != segment_fail()){

        //printf("Claimed segment %u for size %u\n", segment, size);

        uint capacity = max_blocks_per_segment >> size;

        segments[segment].set_size_capacity(size, capacity);

        __threadfence();

        block_trees[size]->insert(segment);

        release_tree_lock(size);

        return true;

      }

      release_tree_lock(size);
      __threadfence();

      n_attempts++;

    }

    //timeout

    return false;


}

constexpr static __device__ uint16_t block_fail(){
  return ~((uint16_t) 0);
}

constexpr static __device__ uint32_t segment_fail(){
  return segment_tree_type::fail();
}

__device__ void * malloc(uint64_t bytes_needed){

  uint size = get_first_bit_bigger(bytes_needed)-min_size;

  //printf("Size is %u\n", size);

  if (size < n_trees){
    uint allocs_needed = 1;

    return malloc_slice(size, allocs_needed);

  }

}

__device__ void * malloc_slice(const uint & size, const uint & allocs_needed){


  int n_attempts = 0;

  
  uint index = ~0;
  uint16_t block = block_fail();

  uint32_t segment = segment_fail();

  while (n_attempts < GALLATIN_MAX_NEW_SLICE_ATTEMPTS){

    bool success = false;

    auto coalesced_team = cg::coalesced_threads();

    auto active_threads = cg::labeled_partition(coalesced_team, size);


    if (active_threads.thread_rank() == 0){

      //retreive and fetch

      success = tree_storage[size]->get_valid_block(segment, block, index);

      //could this trigger bad index?
      if (!success){

        //replace and retry.
        replace_block_or_mark_dead(size, block, segment, index);
        __threadfence();

      }


    }

    //what we know
    //no thread ins9ide of the "active threads == 0"
    // mark detects the issues

    #if GALLATIN_DEBUG
    uint n_success = __popc(active_threads.ballot(success));


    if (n_success != 1 && n_success != 0){
      write_global_log(17, n_success);
    }

    #endif

    success = active_threads.ballot(success);



    //totally dead, system has no blocks.
    if (!success){

      n_attempts++;
      continue;
    }


    //broadcast

    index = active_threads.shfl(index, 0);
    block = active_threads.shfl(block, 0);
    segment = active_threads.shfl(segment, 0);

    #if GALLATIN_DEBUG

    if (index >= tree_storage[size]->n_blocks){
      write_global_log(18, index, tree_storage[size]->n_blocks);
    }

    if (index >= tree_storage[size]->n_blocks && active_threads.thread_rank() ==0){
      write_global_log(19, index, tree_storage[size]->n_blocks);
    }


    #endif



    bool reset_segment = false;

    uint64_t offset = segments[segment].allocate_offset_from_block(active_threads, allocs_needed, block, segment, size, reset_segment);
    
    //if segment is removable re-insert into segment tree
    if (reset_segment){

      if (segments[segment].set_invalid(size, segment_type::get_n_blocks_at_size(size))){

        //reset
        block_trees[size]->remove(segment);
        segment_tree->insert(segment);
        __threadfence();

      }

    }

    success = (offset != ~0ULL);

    if (active_threads.ballot(!success) && active_threads.thread_rank() == 0){


      replace_block_or_mark_dead(size, block, segment, index);

    }

    if (offset != ~0ULL){

      //correct - assert that the memory is not outside of bounds.

      #if GALLATIN_DEBUG

        if (offset >= total_mem){
          write_global_log(23, offset, total_mem);
        }

        if (offset + segment_type::get_allocation_size(size) > total_mem){

          write_global_log(24, offset, total_mem, segment_type::get_allocation_size(size));

        }

      #endif


      return (void *) (offset + (uint64_t) memory_base);
    }


    n_attempts++;

    //block occupied.
    __threadfence();



  }

  return nullptr;

  }


  __device__ void replace_block_or_mark_dead(const uint16_t & size, uint16_t & block, uint32_t & segment, uint32_t & index){


    //claim unset
    //if this fails someone else is in charge.
    if (!tree_storage[size]->claim_to_set_exact(index, block, segment)) return;

    if (!get_new_block(size, block, segment)){
      tree_storage[size]->mark_dead(index);
      return;
    }

    //set.
    tree_storage[size]->set(index, tree_storage[size]->pack_segment_block(segment, block));

    return;

  }


  __device__ uint64_t get_allocation_as_offset(void * allocation){

    return ((uint64_t) allocation) - (uint64_t) memory_base;
  }


  __device__ void free(void * allocation){

    uint64_t allocation_as_offset = get_allocation_as_offset(allocation);

    uint64_t segment_num = allocation_as_offset/bytes_per_segment;

    uint16_t read_size = segments[segment_num].read_size();

    #if GALLATIN_DEBUG
    if (read_size == segment_type::fail_size()){ 
      write_global_log(28, segment_num, read_size);
      return;
    }
    #endif

    free_slice_offset(allocation_as_offset % bytes_per_segment, segment_num, read_size);

  }

  __host__ __device__ static constexpr uint64_t get_bytes_per_segment(){
    return bytes_per_segment;
  }

  __device__ void free_slice_offset(uint64_t offset, uint64_t segment_num, uint16_t size){

    // uint64_t segment_num = offset/bytes_per_segment;

    // offset = offset % bytes_per_segment;

    //potential issue - if return for reset is too fast it can occur before free.
    //need to add back before done?
    //segments[segment_num]->return_offset

    bool first_freed = false;
    uint16_t read_size;

    bool free_to_reset = segments[segment_num].return_offset(offset, read_size, first_freed);


    #if GALLATIN_DEBUG

    if (read_size != size){
      write_global_log(29, segment_num, read_size, size);
    }
    #endif

    if (first_freed && !free_to_reset){
      block_trees[read_size]->insert(segment_num);
    }


    if (free_to_reset){

      //attempt to reset.

      if (segments[segment_num].set_invalid(read_size, segment_type::get_n_blocks_at_size(read_size))){

        //unclaimed.
        block_trees[read_size]->remove(segment_num);
        __threadfence();
        segment_tree->insert(segment_num);
        __threadfence();

      }

      
    }

  }


  __host__ void print_info(){


    GALLATIN_TODO(make this do things.)
    return;
  }



};

}  // namespace allocators

}  // namespace Gallatin

#endif  // End of VEB guard