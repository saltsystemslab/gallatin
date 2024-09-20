#ifndef GALLATIN_SEGMENT
#define GALLATIN_SEGMENT
// A CUDA implementation of the segment queue, made by Hunter McCoy
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

// The segment queue fetches and releases blocks depending on the max cap set.

// inlcudes


#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <gallatin/allocators/config.cuh>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/allocators/block.cuh>
#include <gallatin/allocators/veb.cuh>
#include <gallatin/allocators/murmurhash.cuh>


namespace gallatin {

namespace internals {

  //initialize all segments for later acquisition
  //the default state of a segment is to have all blocks unavailable
  // an available size of "-1" (none can be reserved)
  // and a size set to fail state (~0 uint16_t)
  template <typename segment>
  __global__ void init_segments_kernel(segment * segments, uint64_t n_segments){


    uint64_t tid = gallatin::utils::get_tid();

    if (tid >= n_segments) return;

    segments[tid].first_time_init();

  }


  //segments are naturally aligned on 8 bytes
  //can force 16 but all operations should be aligned.
  #pragma pack(8)
  template <uint blocks_per_segment, uint min_allocation_size, uint bytes_per_segment>
  struct segment{


    static_assert(blocks_per_segment*4096*min_allocation_size == bytes_per_segment);

    using my_type = segment<blocks_per_segment, min_allocation_size, bytes_per_segment>;

    using block_type = gallatin::internals::block;

    //8 bytes
    //Likely can't be 4 - 16 bit backing needed.
    uint64_t blocks_available_and_size;

    //8 bytes

    uint enqueue_counter;

    uint dequeue_counter;

    uint enqueue_finalize_counter;

    uint padding;

    //blocks - 8 bytes each.
    block_type blocks [blocks_per_segment];
    uint16_t queue[blocks_per_segment];


    //can static and constexpr/const stack?
    //or does static imply const as it is stateless?
    //constexpr 
    __device__ constexpr static uint16_t fail_size(){
      return ~ ((uint16_t) 0);
    }

    __device__ uint16_t read_size(){

      const uint16_t * upper_bits = &((uint16_t *) &blocks_available_and_size)[3];

      return gallatin::utils::ld_acq(upper_bits);

    }


    __device__ uint16_t extract_size(uint64_t packed_count_and_size){

      return (uint16_t) (packed_count_and_size >> SEGMENT_PACKED_COUNTER_OFFSET);

    }

    __device__ int extract_count(uint64_t packed_count_and_size){

      uint64_t packed_no_size = packed_count_and_size & BITMASK(SEGMENT_PACKED_COUNTER_OFFSET);

      return ((int) packed_no_size) - SEGMENT_SIZE_OFFSET;

    }

    //function signature casts
    //pack both variables together into a format reversible
    // with extract_size and extract_count
    __device__ static uint64_t pack_size_available(uint64_t size, int max_available){

      return (size << SEGMENT_PACKED_COUNTER_OFFSET) + (max_available+SEGMENT_SIZE_OFFSET);

    }

    //each size up halves the number of blocks.
    static __device__ inline uint get_n_blocks_at_size(uint16_t size){
      return blocks_per_segment >> size;
    }

    //set size to the failure size, and count to -1
    //swap ONLY occurs when all blocks are released - otherwise what's the point?
    __device__ bool set_invalid(uint16_t expected_size, int max_available){

      //let the compiler handle this.
      const uint64_t packed_val = pack_size_available(fail_size(), -1);

      uint64_t necessary_swap_value = pack_size_available(expected_size, max_available);



      //to prevent races this must be an atomic
      // - explicitly swap with expected "ALL_AVAILABLE" marker.
      //can return bool - if it fails
      return (atomicCAS((unsigned long long int *)&blocks_available_and_size, necessary_swap_value, packed_val) == necessary_swap_value);

    }

    //godbolt this and see if an instruction can be saved on cast.
    __device__ bool set_size_capacity(uint16_t size, int capacity){


      //need to think about this more - this requires some form
      // of CAS to ensure correctness.
      // maybe 16 bit CAS first, then 64?
      // seems reasonable - if you swap from invalid than you are correct
      //doing so does not affect lower bits so other atomic BS goes through

      const uint16_t * upper_bits = &((uint16_t *) &blocks_available_and_size)[3];

      //can this race with other threads?
      //gallatin free is going to swap to unse
      if (atomicCAS((unsigned short int *)upper_bits, (unsigned short int) fail_size(), (unsigned short int) size) == fail_size()){

        //success! set values in earnest
        uint64_t packed_val = pack_size_available(size, capacity);

        gallatin::utils::st_rel(&enqueue_counter, 0);
        gallatin::utils::st_rel(&enqueue_finalize_counter, 0);
        gallatin::utils::st_rel(&dequeue_counter, 0);
        gallatin::utils::st_rel(&blocks_available_and_size, packed_val);

        return true;
      
      }

      return false;


    }

    __device__ void first_time_init(){


      const uint64_t invalid_val = pack_size_available(fail_size(), -1);

      gallatin::utils::st_rel(&blocks_available_and_size, invalid_val);

      for (int i = 0; i < blocks_per_segment; i++){

        blocks[i].init();
        queue[i] = fail_size();


      }


    }

    __device__ uint16_t reserve_block(uint16_t size, bool & last){


      //credit to https://stackoverflow.com/questions/7221409/is-unsigned-integer-subtraction-defined-behavior
      // for help diagnosing this issue.
      // overflow is defined behavior on atomics, the result is wrapped % MAX_UINT+1
      // this means to subtract one you must add ~0.
      // Adding ~1 actually makes it subtraction by 2.  
      uint64_t size_and_count = atomicAdd((unsigned long long int *)&blocks_available_and_size, ~0ULL);

      //atomicAdd((unsigned long long int *)&blocks_available_and_size, 1ULL);

      uint16_t stored_size = extract_size(size_and_count);

      int count = extract_count(size_and_count);


      if (stored_size != size || count <= 0){

        //count failed, rollback
        atomicAdd((unsigned long long int *)&blocks_available_and_size, 1ULL);

        return fail_size();
      }

      last = (count == 1);

      uint queue_position = atomicAdd((unsigned int *)&dequeue_counter, 1ULL);

      if (queue_position < get_n_blocks_at_size(size)){

        //valid queue position for direct steal.

        setup_block(queue_position, size);

        return queue_position;


      }

      //to claim queue position, must be valid in queue.
      //queue allowed to grow above the max size
      uint16_t queue_val = gallatin::utils::ld_acq(&queue[queue_position % blocks_per_segment]);

      #if GALLATIN_DEBUG

      if (queue_val > get_n_blocks_at_size(size)){
        write_global_log(11, get_n_blocks_at_size(size), queue_val);
      }

      #endif

      setup_block(queue_val, size);

      return queue_val;

    }

    __device__ void setup_block(uint16_t block_id, uint16_t size){


      //it is an error to request a block that does not exist.
      #if GALLATIN_DEBUG

      if (block_id >= get_n_blocks_at_size(size)){
        write_global_log(11, block_id, get_n_blocks_at_size(size));
      }

      #endif

      blocks[block_id].reset(size);

    }

    __device__ block * acquire_block(uint16_t block_id, uint16_t size){


      //it is an error to request a block that does not exist.
      #if GALLATIN_DEBUG

      if (block_id >= get_n_blocks_at_size(size)){
        write_global_log(11, block_id, get_n_blocks_at_size(size));
      }

      #endif

      blocks[block_id].reset(size);

      return &blocks[block_id];

    }

    //release a block back to the system
    //returns true if this is the last block returned - may be valid to swap out IFF no other threads interrupt.  
    __device__ bool free_block(uint16_t block_id, bool & first_freed){


      //increment free counter;

      uint enqueue_address = atomicAdd((unsigned int *)&enqueue_counter, 1ULL);


      gallatin::utils::st_rel(&queue[enqueue_address % blocks_per_segment], block_id);

      __threadfence();

      while (atomicCAS((unsigned int *)&enqueue_finalize_counter, enqueue_address, enqueue_address+1) != enqueue_address);


      //increment return counter;

      uint64_t prev_stored = atomicAdd((unsigned long long int *)&blocks_available_and_size, 1ULL);

      uint16_t expected_size = extract_size(prev_stored);

      uint max_allocs = get_n_blocks_at_size(expected_size);

      int extracted_count = extract_count(prev_stored);

      #if GALLATIN_DEBUG


      

      first_freed = (extracted_count == 0);


      //if you try to return more blocks than exist than I'mma get ya
      if (extracted_count+1 > max_allocs){

        write_global_log(12, extracted_count+1, max_allocs);

      }

      #endif

      return (extracted_count+1 == max_allocs);



    }

    __device__ static uint64_t get_allocation_size(uint16_t size){
      return min_allocation_size << size;
    }

    //given a claimed block and segment, return allocation # or ~0ULL on failure.
    //reset_segment may return true if the request for num_allocs trips and resets the block.
    //resets are handled by this code, but may trigger a reset request for the segment
    //which is passed back by the reset_segment bool passed by reference.
    // the reset_segment is only set true IFF return value is not ~0ULL
    // as any valid allocation from the block implies block free_counter != 4096 so not reset.

    //return value is an offset into memory controlled by gallatin.
    // this offset can be directly cast against a memory base as an allocation.
    __device__ uint64_t allocate_offset_from_block(cg::coalesced_group active_threads, uint num_allocs, uint16_t block_id, uint16_t segment_id, uint16_t size, bool & reset_segment){

      //calculate size

      //sanity check

      #if GALLATIN_DEBUG

      if (block_id >= get_n_blocks_at_size(size)){
        write_global_log(20, block_id, get_n_blocks_at_size(size), size);
      }

      #endif

      bool block_free = false;

      const uint64_t allocation_size = get_allocation_size(size);

      uint64_t block_offset = blocks[block_id].block_malloc(active_threads, num_allocs, size, block_free);

      if (block_offset == ~0ULL){
        //fail

        //only one thread should receive this.
        if (block_free){
          reset_segment = free_block(block_id, block_free);
        }

        return ~0ULL;
      }

      //else success!
      //calculate global offset.


      uint64_t segment_offset = 1ULL*bytes_per_segment*segment_id;

      GALLATIN_TODO(check if offset calculation is FMA on godbolt.)



      uint64_t final_offset = segment_offset + allocation_size*(block_id*4096+block_offset);

      #if GALLATIN_DEBUG

      uint64_t total_block_offset = allocation_size*(block_id*4096+block_offset);



      if (total_block_offset + allocation_size > bytes_per_segment){
        write_global_log(21, total_block_offset, allocation_size, bytes_per_segment);
      }

      //sanity assertion that segment allocation doesn't bleed
      if (final_offset + allocation_size > ((uint64_t) (segment_id+1)) * bytes_per_segment){
        write_global_log(13, final_offset, allocation_size,  bytes_per_segment);
      }

      //sanity check - the allocation given out associates 1-1 with the segment

      uint64_t segment = final_offset/bytes_per_segment;

      if (segment != segment_id){
        write_global_log(22, final_offset, segment, segment_id);
      }

      #endif


      //check that final offset does translate?

      return final_offset;



    }

    //return allocation to the system.
    //This assumes that the allocation hash been run through "determine_segment_num" function below.
    // this casts back to "count space", returns handle to appropriate block, and returns
    // if last block in system was freed
    // add on first_freed returns true if first block in system is freed
    __device__ bool return_offset(uint64_t allocation, uint16_t & read_size, bool & first_freed){

      GALLATIN_TODO(Check efficiency of different free strategies)
      GALLATIN_TODO(Only let one thread call ld_acq)

      const uint16_t * upper_bits = &((uint16_t *) &blocks_available_and_size)[3];

      //threadfence before to ensure morally strong
      __threadfence();
      read_size = gallatin::utils::ld_acq(upper_bits);

      #if GALLATIN_DEBUG

      if (read_size == fail_size()){
        write_global_log(14);
      }

      #endif

      const uint64_t allocation_size = get_allocation_size(read_size);

      #if GALLATIN_DEBUG

      if (allocation % allocation_size != 0){
        write_global_log(15, allocation, allocation_size);
      }

      if (allocation+allocation_size > bytes_per_segment){
        write_global_log(25,allocation, allocation_size, bytes_per_segment);
      }

      if (read_size == fail_size()){
        write_global_log(30, read_size);
      }

      #endif

      //uint64_t allocation_offset = allocation/allocation_size;

      uint16_t block_id = allocation/(4096*allocation_size);

      auto full_team = cg::coalesced_threads();


      #if GALLATIN_DEBUG

      //make sure all threads agree on size
      auto debug_active_threads = labeled_partition(full_team, block_id);

      uint16_t first_size = debug_active_threads.shfl(read_size, 0);

      if (first_size != read_size){
        write_global_log(31, first_size, read_size);
      }

      uint lead_block_id = debug_active_threads.shfl(block_id, 0);

      if (lead_block_id != block_id){
        write_global_log(32, lead_block_id, block_id);
      }


      #endif
      //separate by segment.
      auto first_active_threads = labeled_partition(full_team, (uint64_t) this);
      auto active_threads = labeled_partition(first_active_threads, block_id);
      //auto active_threads = labeled_partition(full_team, full_team.thread_rank());

      bool block_freed = blocks[block_id].block_free(active_threads);

      if (block_freed){
        return free_block(block_id, first_freed);
        //return false;
      }

      return false;



    }

    //determine which segment an allocation belongs to. 
    // the segment must be "unbased", i.e. it has been converted from void * space
    // to gallatin's internal tracker space.
    //this also clips the allocation for return to the segment, as clipping is only used for return.
    __device__ static uint64_t determine_segment_num(uint64_t  & unbased_allocation){

      uint64_t segment = unbased_allocation / bytes_per_segment;

      GALLATIN_TODO(check that godbolt says this gets converted to bitmask. If not convert to BITMASK and static assert of property.)
      unbased_allocation = unbased_allocation % bytes_per_segment;

    }

    __host__ static my_type * generate_on_device(uint32_t n_segments){

      my_type * device_version = gallatin::utils::get_device_version<my_type>(n_segments);

      init_segments_kernel<my_type><<<(n_segments-1)/512+1,512>>>(device_version, n_segments);

      return device_version;

    }

    //segments as a whole can be released
    //no unset of memory required and no lower components.
    __host__ static void free_on_device(my_type * dev_version){
      cudaFree(dev_version);
    }



  };

}  // namespace internals

}  // namespace gallatin

#endif  // End of VEB guard