#ifndef GALLATIN_BLOCK
#define GALLATIN_BLOCK

#include <gallatin/allocators/config.cuh>


namespace gallatin {


namespace internals {


  //A block represents 4096 allocations of a specific size 
  //each block is represented by 2 counters, the malloc counter and free counter
  //the upper 6 bits of a malloc counter represent the tree size, while the lower 26 bits represent malloc count
  //rollover must not occur

  //Addition of pack gaurantees that uint64_t swap is valid, as it is aligned.
  #pragma pack(8)
  struct block {

    uint malloc_counter;
    uint free_counter;


    __device__ void init(){
      malloc_counter = 4097;
      free_counter = 4096;
    }


    __device__ void reset(uint16_t tree_size){

      uint64_t shifted_tree_size = ((uint64_t) tree_size) << (GALLATIN_BLOCK_TREE_OFFSET);

      //uint64_t merged = ((uint64_t) shifted_tree_size) << 32;

      uint64_t leftover = atomicExch((unsigned long long int *)this, (unsigned long long int) shifted_tree_size);

      #if GALLATIN_DEBUG

      //only fully allocated, fully released blocks should be freed.
      uint32_t malloc_leftover = leftover & BITMASK(32);
      uint32_t free_leftover = leftover >> 32;


      uint count = malloc_leftover & BITMASK(GALLATIN_BLOCK_TREE_OFFSET);

      if (count < 4096 || free_leftover != 4096){
        write_global_log(26, count, free_leftover);
      }

      #endif


    }


    // //called before block is returned to the system.
    // __device__ void reset_frees(){

    //   gallatin::utils::st_rel(&free_counter, 0U);

    // }


    //helpers for extracting
  __device__ uint64_t extract_count(cg::coalesced_group &active_threads, uint old_count, uint group_sum){

    uint true_count = (old_count & BITMASK(GALLATIN_BLOCK_TREE_OFFSET));

    uint my_value = true_count + group_sum;

    return my_value;

  }


  __device__ bool check_valid(uint old_count, uint16_t tree_size){


    uint block_tree_size = (old_count >> GALLATIN_BLOCK_TREE_OFFSET);

    return (block_tree_size == tree_size);

  }


    //pull multiple allocations per thread
    //the threads operate as a team to perform this together.
    __device__ uint64_t block_malloc(cg::coalesced_group &active_threads, uint copies_needed, uint16_t tree_id, bool & reset_free){

    //calculate exclusive sum - if value is less than that, valid

    uint my_group_sum = cg::exclusive_scan(active_threads, copies_needed, cg::plus<uint>());

    //last thread in group has total size and controls atomic

    uint prev_count;

    if (active_threads.thread_rank() == active_threads.size()-1){

      prev_count = atomicAdd((unsigned int *)&malloc_counter, my_group_sum+copies_needed);

    }

    prev_count = active_threads.shfl(prev_count, active_threads.size()-1);

    bool valid = check_valid(prev_count, tree_id);

    //after this prev count is correct.
    //and marks the # of allocations given out beforehand
    // this is also the start index of this allocation.
    prev_count = extract_count(active_threads, prev_count, my_group_sum);


    #if GALLATIN_DEBUG

    if (prev_count > 30000000){ write_global_log(2, (uint64_t)this, prev_count); }

    #endif

    uint n_drop;

    //successful marks if allocation was correct.
    bool successful = (prev_count + copies_needed) <= 4096;

    if (successful){

      //if valid, should be copies_needed-1
      //if invalid, should be copies needed as all must be returned.
      n_drop = (copies_needed) + (valid)*-1;
    } else if (prev_count < 4096){

      //this is the case where the allocation bled over
      //need to correct for claimed allocs
      //this one alloc runs from prev_count > 4096, if it didn't it would be successful.
      n_drop = 4096-prev_count;

      //printf("Trigger: my_count %u, copies_needed %u, correcting %u\n", prev_count, copies_needed, n_drop);


    } else {
      //no allocations claimed.
      n_drop = 0;
    }


    uint excess_allocs = cg::reduce(active_threads, n_drop, cg::plus<uint>());

    if (excess_allocs > 0 && active_threads.thread_rank() == 0){

      //don't need to check free logic, as at least one allocation must be active!
      // this is only true IFF the stride is an even multiple.
      uint free_result = atomicAdd((unsigned int *)&free_counter, excess_allocs);


      reset_free = (free_result+excess_allocs) == 4096;

    }

    //must team wait? if any thread is successful thread 0 must be successful.
    //active_threads.sync();

    if (valid && successful){

      #if GALLATIN_DEBUG

      if (prev_count >= 4096 || prev_count + copies_needed > 4096){
        write_global_log(27, prev_count);
      }
      #endif

      return prev_count;
    }

    //threads should undo bad progress
    // if (prev_count+my_group_sum+copies_needed > 2000000 && active_threads.thread_rank() == active_threads.size()-1){


    //   // if (my_group_sum+copies_needed == 0){
    //   //   printf("Bad group sum\n");
    //   // }
    //   // printf("Undoing progress, count is %u, copies %u, free_counter %u\n", prev_count, my_group_sum+copies_needed, gallatin::utils::ld_acq(&free_counter));

    //   while(prev_count+my_group_sum+copies_needed > 2000000){


    //     uint prepped_val = (((uint) tree_id) << GALLATIN_BLOCK_TREE_OFFSET)+prev_count;
    //     prev_count = atomicCAS(&malloc_counter, prepped_val+my_group_sum+copies_needed, prepped_val);

    //     if (prev_count == prev_count+my_group_sum+copies_needed) return ~0ULL;
    //     prev_count = prev_count-(my_group_sum+copies_needed);

    //   }

    // }


    return ~0ULL;


  }


  __device__ bool block_free(cg::coalesced_group &active_threads) {

    #if GALLATIN_DEBUG

    uint64_t host_this = active_threads.shfl((uint64_t) this, 0);

    if (host_this != ((uint64_t) this)){
      write_global_log(33, host_this, (uint64_t) this);
    }
    #endif

    uint old;
    if (active_threads.thread_rank() == 0){

      old = atomicAdd((unsigned int *)&free_counter, active_threads.size());

    }


    old = active_threads.shfl(old, 0) + active_threads.thread_rank();

    #if GALLATIN_DEBUG


    if (old >= 4096){
      write_global_log(1, (uint64_t) this, gallatin::utils::ld_acq(&malloc_counter), old);
    } 

    #endif

    //return true if this is the last thread.
    return (old == 4095);
  }


};




} //namespace internals

} //namespace gallatin


#endif







