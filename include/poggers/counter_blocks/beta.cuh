#ifndef BETA_ALLOCATOR
#define BETA_ALLOCATOR
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


#define BETA_TRAP_ON_ERR 1

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/counter_blocks/one_size_allocator.cuh>
#include <poggers/counter_blocks/block.cuh>
#include <poggers/counter_blocks/memory_table.cuh>
#include <poggers/counter_blocks/shared_block_storage.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>

#ifndef BETA_DEBUG_PRINTS
#define BETA_DEBUG_PRINTS 1
#endif




namespace beta {

namespace allocators {

#define REQUEST_BLOCK_MAX_ATTEMPTS 10

#define BETA_MAX_ATTEMPTS 150
#define BETA_MALLOC_LOOP_ATTEMPTS 5

//minimum number of pinned blocks per tree
#define MIN_PINNED_CUTOFF 4


//maximum number of segments allowed per tree.
//this is to prevent exponential blowup.
#define BETA_TREE_SEGMENT_CAP 10

// alloc table associates chunks of memory with trees

// using uint16_t as there shouldn't be that many trees.

// register atomically inserst tree num, or registers memory from segment_tree.

using namespace poggers::utils;

__global__ void boot_segment_trees(veb_tree **segment_trees,
                                   uint64_t max_chunks, int num_trees) {
  uint64_t tid = poggers::utils::get_tid();

  if (tid >= max_chunks) return;

  for (int i = 0; i < num_trees; i++) {
    segment_trees[i]->remove(tid);
  }
}


//sanity check: are the VEB trees empty?
__global__ void assert_empty(veb_tree ** segment_trees, int num_trees){

  uint64_t tid = poggers::utils::get_tid();

  if (tid != 0) return;


  for (int i =0; i< num_trees; i++){

    uint64_t alloc = segment_trees[i]->malloc_first();

    if (alloc != veb_tree::fail()){
      printf("Failed to clean VEB tree %d: Index %llu live\n", i, alloc);
    }
  }


}

//boot the allocator memory blocks during initialization
//this loops through the blocks and initializes half as many at each iteration.
template <typename allocator>
__global__ void boot_shared_block_container(allocator * alloc, uint16_t max_tree_id, int max_smid, int cutoff){

	uint64_t tid = poggers::utils::get_tid();

	uint16_t tree_id = 0;

	while (tree_id < max_tree_id){

		if (tid >= max_smid) return;

		alloc->boot_block(tree_id, tid);

		max_smid = max_smid/2;

		if (max_smid < cutoff) max_smid = cutoff;

		tree_id+=1;

	}

}



template <typename allocator>
__global__ void boot_shared_block_container_one_thread(allocator * alloc, uint16_t max_tree_id, int max_smid, int cutoff){

  uint64_t tid = poggers::utils::get_tid();

  uint16_t tree_id = 0;

  if (tid != 0) return;


  while (tree_id < max_tree_id){

    for (int i = 0; i < max_smid; i++){

      alloc->boot_block(tree_id, i);

    }

    max_smid = max_smid/2;

    if (max_smid < cutoff) max_smid = cutoff;

    tree_id +=1;

  }

}

template <typename allocator>
__global__ void print_overhead_kernel(allocator * alloc){

  uint64_t tid = poggers::utils::get_tid();

  if (tid != 0) return;

  uint64_t overhead = alloc->calculate_overhead();

  printf("Allocator is using %llu bytes of overhead\n", overhead);

  return;

}

template <typename allocator>
__global__ void print_guided_fill_kernel(allocator * table, uint16_t id){

  uint64_t tid = poggers::utils::get_tid();

  if (tid != 0) return;

  table->print_guided_fill(id);

}



// main allocator structure
// template arguments are
//  - size of each segment in bytes
//  - size of smallest segment allocatable
//  - size of largest segment allocatable
template <uint64_t bytes_per_segment, uint64_t smallest, uint64_t biggest>
struct beta_allocator {
  using my_type = beta_allocator<bytes_per_segment, smallest, biggest>;
  using sub_tree_type = veb_tree;
  using pinned_block_type = pinned_shared_blocks<smallest, biggest>;

  // internal structures
  veb_tree *segment_tree;

  alloc_table<bytes_per_segment, smallest> *table;

  sub_tree_type **sub_trees;

  pinned_block_type *local_blocks;

  int num_trees;

  int smallest_bits;

  uint * locks;

  // generate the allocator on device.
  // this takes in the number of bytes owned by the allocator (does not include
  // the space of the allocator itself.)
  static __host__ my_type *generate_on_device(uint64_t max_bytes,
                                              uint64_t seed, bool print_info=true) {
    my_type *host_version = get_host_version<my_type>();

    // plug in to get max chunks
    uint64_t max_chunks = get_max_chunks<bytes_per_segment>(max_bytes);

    uint64_t total_mem = max_bytes;

    host_version->segment_tree = veb_tree::generate_on_device(max_chunks, seed);

    // estimate the max_bits
    uint64_t blocks_per_pinned_block = 128;
    uint64_t num_bits = bytes_per_segment / (4096 * smallest);

    host_version->local_blocks =
        pinned_block_type::generate_on_device(blocks_per_pinned_block, MIN_PINNED_CUTOFF);

    uint64_t num_bytes = 0;

    do {
      //printf("Bits is %llu, bytes is %llu\n", num_bits, num_bytes);

      num_bytes += ((num_bits - 1) / 64 + 1) * 8;

      num_bits = num_bits / 64;
    } while (num_bits > 64);

    num_bytes += 8 + num_bits * sizeof(Block);

    uint64_t num_trees =
        get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest) + 1;
    host_version->smallest_bits = get_first_bit_bigger(smallest);
    host_version->num_trees = num_trees;

    // init sub trees
    sub_tree_type **ext_sub_trees =
        get_host_version<sub_tree_type *>(num_trees);

    for (int i = 0; i < num_trees; i++) {
      sub_tree_type *temp_tree =
          sub_tree_type::generate_on_device(max_chunks, i + seed);
      ext_sub_trees[i] = temp_tree;
    }

    host_version->sub_trees =
        move_to_device<sub_tree_type *>(ext_sub_trees, num_trees);

    boot_segment_trees<<<(max_chunks - 1) / 512 + 1, 512>>>(
        host_version->sub_trees, max_chunks, num_trees);

   

    #if BETA_DEBUG_PRINTS

    cudaDeviceSynchronize();


    assert_empty<<<1,1>>>(host_version->sub_trees, num_trees);

    cudaDeviceSynchronize();

    #endif


    uint * host_locks = poggers::utils::get_host_version<uint>(num_trees);

    for (int i =0; i < num_trees; i++){
      host_locks[i] = 0;
    }

    host_version->locks = poggers::utils::move_to_device<uint>(host_locks, num_trees);

    host_version
        ->table = alloc_table<bytes_per_segment, smallest>::generate_on_device(
        max_bytes);  // host_version->segment_tree->get_allocator_memory_start());

    if (print_info){
      printf("Booted BETA with %llu trees in range %llu-%llu and %f GB of memory %llu segments\n", num_trees, smallest, biggest, 1.0*total_mem/1024/1024/1024, max_chunks);
    }
    


    auto device_version = move_to_device(host_version);

    boot_shared_block_container<my_type><<<(blocks_per_pinned_block-1)/128+1, 128>>>(device_version,num_trees, blocks_per_pinned_block, MIN_PINNED_CUTOFF);

    cudaDeviceSynchronize();

    return device_version;

  }

  // return the index of the largest bit set
  static __host__ __device__ int get_first_bit_bigger(uint64_t counter) {
    return poggers::utils::get_first_bit_bigger(counter);
  }

  // get number of sub trees live
  static __host__ __device__ int get_num_trees() {
    return get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest) + 1;
  }

  // return memory used to device
  static __host__ void free_on_device(my_type *dev_version) {
    // this frees dev version.
    my_type *host_version = move_to_host<my_type>(dev_version);

    uint64_t num_trees =
        get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest) + 1;

    sub_tree_type **host_subtrees =
        move_to_host<sub_tree_type *>(host_version->sub_trees, num_trees);

    for (int i = 0; i < num_trees; i++) {
      sub_tree_type::free_on_device(host_subtrees[i]);
    }

    alloc_table<bytes_per_segment, smallest>::free_on_device(
        host_version->table);


    cudaFree(host_version->locks);

    veb_tree::free_on_device(host_version->segment_tree);

    pinned_block_type::free_on_device(host_version->local_blocks);

    cudaFreeHost(host_subtrees);

    cudaFreeHost(host_version);
  }

  // given a pointer, return the segment it belongs to
  // __device__ inline uint64_t snap_pointer_to_block(void *ext_ptr) {


  //   char *memory_start = table->get_segment_memory_start(segment);

  //   uint64_t snapped_offset =
  //       ((uint64_t)(ext_ptr - memory_start)) / bytes_per_segment;

  //   return snapped_offset;
  // }

  // Cast an offset back into a memory pointer
  // this requires the offset and the tree_id so that we know how far to scale
  // the pointer
  __device__ void *alloc_offset_to_ptr(uint64_t offset, uint16_t tree_id) {
    uint64_t block_id = offset / 4096;

    uint64_t relative_offset = offset % 4096;

    uint64_t segment = block_id / table->blocks_per_segment;

    uint64_t alloc_size = table->get_tree_alloc_size(tree_id);

    char *memory_start = table->get_segment_memory_start(segment);

    // with start of segment and alloc size, we can set the pointer relative to
    // the segment
    return (void *)(memory_start + relative_offset * alloc_size);
  }



  //initialize a block for the first time.
  __device__ void boot_block(uint16_t tree_id, int smid){

    per_size_pinned_blocks * local_shared_block_storage =
          local_blocks->get_tree_local_blocks(tree_id);


    if (smid >= local_shared_block_storage->num_blocks){

      #if BETA_DEBUG_PRINTS
      printf("ERR %d >= %llu\n", smid, local_shared_block_storage->num_blocks);
      #endif

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

      return;

    }

  	Block * new_block = request_new_block_from_tree(tree_id);

  	if (new_block == nullptr){

      #if BETA_DEBUG_PRINTS
  		printf("Failed to initialize block %d from tree %u", smid, tree_id);
      #endif

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

  		return;
  	}


    #if BETA_DEBUG_PRINTS


    uint64_t alt_block_segment = table->get_segment_from_block_ptr(new_block);

    uint16_t alt_tree_id = table->read_tree_id(alt_block_segment);


    uint64_t block_id = table->get_global_block_offset(new_block);

    if (tree_id != alt_tree_id){
        //this triggers, yeet
        printf("Boot Block %llu: segment %llu not init for malloc: %u != %u\n", block_id, alt_block_segment, tree_id, alt_tree_id);

        #if BETA_TRAP_ON_ERR
        asm("trap;");
        #endif

      }
    #endif



    if(!local_shared_block_storage->swap_out_nullptr(smid, new_block)){

      #if BETA_DEBUG_PRINTS
    	printf("Error: Block in position %d for tree %d already initialized\n", smid, tree_id);
      #endif

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

    }

  }


  //replace block with a new one pulled from the system
  //gets called when a block is detected to be empty.
  __device__ bool replace_block(int tree_id, int smid, Block * my_block, per_size_pinned_blocks * my_pinned_blocks){

  	if (my_pinned_blocks->swap_out_block(smid, my_block)){

      __threadfence();

  		Block * new_block = request_new_block_from_tree(tree_id);

  		if (new_block == nullptr){

        #if BETA_DEBUG_PRINTS
        printf("Failed to acquire block\n");
        #endif

  			return false;
  		}

  		if (!my_pinned_blocks->swap_out_nullptr(smid, new_block)){

        #if BETA_DEBUG_PRINTS
  			printf("Incorrect behavior when swapping out block index %d for tree %d\n", smid, tree_id);
        #endif

  			free_block(new_block);
  			return false;
  		}

  	}

  	return true;


  }


  __device__ uint16_t get_tree_id_from_size(uint64_t size){

      if (size < smallest) return 0;

      return get_first_bit_bigger(size) - smallest_bits;

  }



  __device__ void * malloc(uint64_t size){

    uint16_t tree_id = get_tree_id_from_size(size);

    if (tree_id >= num_trees){

      int smallest_tree_bits = get_first_bit_bigger(smallest*4096);

      int block_tree = (int) get_first_bit_bigger(size) - smallest_tree_bits;

      if (block_tree < 0) block_tree = 0;

      // #if BETA_DEBUG_PRINTS
      // printf("Snapped tree id to size %d\n", block_tree);
      // #endif

      tree_id = (uint16_t) block_tree;

    }

    uint64_t attempt_counter = 0;

    uint64_t offset = malloc_offset(size);

    while (offset == ~0ULL && attempt_counter < BETA_MALLOC_LOOP_ATTEMPTS){

        offset = malloc_offset(size);
        attempt_counter+=1;
        
    }

    if (offset == ~0ULL) return nullptr;


    #if BETA_DEBUG_PRINTS

      uint64_t segment = table->get_segment_from_offset(offset);

      uint16_t alt_tree_id = table->read_tree_id(segment);

      uint64_t block_id = offset/4096;

      Block * my_block = table->get_block_from_global_block_id(block_id);

      uint64_t block_segment = table->get_segment_from_block_ptr(my_block);

      uint64_t relative_offset = table->get_relative_block_offset(my_block);

      uint64_t block_tree = table->read_tree_id(block_segment);

      if (alt_tree_id != tree_id){

        uint16_t next_segment_id = table->read_tree_id(block_segment+1);

        uint16_t prev_segment_id = table->read_tree_id(block_segment-1);

        //read the counters
        int malloc_status = atomicCAS((int *)&table->malloc_counters[segment], 0, 0);
        int free_status = atomicCAS((int *)&table->free_counters[segment], 0, 0);

        //test here verifies that segment is being reset...
        //It is not a misread of the segment≥
        #if BETA_DEBUG_PRINTS
        printf("Mismatch for offset: %llu in tree ids for alloc of size %llu: %u != %u...Block %llu segment %llu offset %llu tree %u... prev is %u Next is %u. Malloc %d, free %d.\n", offset, size, tree_id, alt_tree_id, block_id, block_segment, relative_offset, block_tree, prev_segment_id, next_segment_id, malloc_status, free_status);
        #endif

        #if BETA_TRAP_ON_ERR
        asm("trap;");
        #endif

      }

    #endif


    void * alloc = offset_to_allocation(offset, tree_id);

    #if BETA_DEBUG_PRINTS

    uint64_t alloc_segment = table->get_segment_from_ptr(alloc);

    if (alloc_segment != segment){

      printf("Malloc: Offset %llu mismatch in segment: %llu != %llu, tree %u\n", offset, segment, alloc_segment, tree_id);

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

    }

    uint64_t alt_offset = allocation_to_offset(alloc, tree_id);

    uint64_t alt_offset_segment = table->get_segment_from_offset(offset);

    if (alt_offset_segment != segment){
      printf("Malloc: mismatch in cast back: %llu != %llu\n", alt_offset_segment, segment);

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

    }

    #endif

    return alloc;

  }



  __device__ void free(void * allocation){



    //this logic is verifie allocation to offset
    uint64_t segment = table->get_segment_from_ptr(allocation);

    uint16_t tree_id = table->read_tree_id(segment);


    #if BETA_DEBUG_PRINTS
    if (tree_id > 10){
      printf("Segment %llu reports large tree %u\n", segment, tree_id);

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif
    }

    #endif

    uint64_t offset = allocation_to_offset(allocation, tree_id);

   

    #if BETA_DEBUG_PRINTS

      uint64_t raw_bytes = (char *) allocation - table->memory;
    
      uint64_t offset_segment = table->get_segment_from_offset(offset);

      if (segment != offset_segment){
        printf("pointer %llx - bytes: %llu, offset: %llu - Free segment Ids Mismatch: %llu != %llu, tree %u\n", (uint64_t) allocation, raw_bytes, offset, segment, offset_segment, tree_id);
      }

    #endif

    return free_offset(offset);


  }



  // malloc an individual allocation
  // returns an offset that can be cast into the associated void *
  __device__ uint64_t malloc_offset(uint64_t bytes_needed) {

    if (bytes_needed < smallest) bytes_needed = smallest;

    uint16_t tree_id = get_first_bit_bigger(bytes_needed) - smallest_bits;

    if (tree_id >= num_trees) {

      //first, determine if the allocation can be satisfied by a full block
      int smallest_tree_bits = get_first_bit_bigger(smallest*4096);

      int block_tree = get_first_bit_bigger(bytes_needed) - smallest_tree_bits;

      if (block_tree < num_trees){

        if (block_tree < 0) block_tree = 0;

        // #if BETA_DEBUG_PRINTS
        // printf("Alloc of %llu bytes pulling from block in tree %d\n", bytes_needed, block_tree);
        // #endif

        Block * my_block = request_new_block_from_tree((uint16_t ) block_tree);

        if (my_block == nullptr){
          return ~0ULL;
        }

        uint64_t global_block_id = table->get_global_block_offset(my_block);

        uint old = my_block->malloc_fill_block(block_tree);

        if (old != 0){

          #if BETA_DEBUG_PRINTS
          printf("Block was already set %u\n", old);
          #endif


          free_offset(global_block_id*4096);

          return ~0ULL;

        }


        return global_block_id*4096;


      } else {

        #if BETA_DEBUG_PRINTS
        printf("Segment-size allocations not supported\n");
        #endif

        return ~0ULL;

      }


      // get big allocation
      // this is currently unfinished, is a todo after ouroboros

      #if BETA_DEBUG_PRINTS
      printf("This code should not trigger\n");
      #endif

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

      return ~0ULL;

    } else {
      // get local block storage and thread storage
      per_size_pinned_blocks * local_shared_block_storage =
          local_blocks->get_tree_local_blocks(tree_id);

      int shared_block_storage_index;
      Block * my_block;

      int num_attempts = 0;

      // this cycles until we either receive an allocation or fail to request a
      // new block
      while (num_attempts < BETA_MAX_ATTEMPTS) {

      //reload memory at start of each loop
      __threadfence();

    	cg::coalesced_group full_warp_team = cg::coalesced_threads();

      cg::coalesced_group coalesced_team = labeled_partition(full_warp_team, tree_id);

    	if (coalesced_team.thread_rank() == 0){
    		shared_block_storage_index = local_shared_block_storage->get_valid_block_index();
    		my_block = local_shared_block_storage->get_my_block(shared_block_storage_index);
    	}

    	//recoalesce and share block.
    	shared_block_storage_index = coalesced_team.shfl(shared_block_storage_index, 0);
    	my_block = coalesced_team.shfl(my_block, 0);

      //cycle if we read an old block
    	if (my_block == nullptr){
    		num_attempts+=1;
    		continue;
    	}

      #if BETA_DEBUG_PRINTS


      uint64_t alt_block_segment = table->get_segment_from_block_ptr(my_block);

      uint16_t alt_tree_id = table->read_tree_id(alt_block_segment);

      uint64_t block_id = table->get_global_block_offset(my_block);

      uint64_t relative_block_id = table->get_relative_block_offset(my_block);

      if (tree_id != alt_tree_id){

        if (alt_tree_id > num_trees){

          //this error occurs when the tree id stored differs - this is an indicator of stale pointers

          //stale pointers are now detected and pruned
          printf("Reading from broken segment %llu, alt value %u != %u\n", alt_block_segment, alt_tree_id, tree_id);
          return ~0ULL;

        }



        bool main_tree_owns = sub_trees[tree_id]->query(alt_block_segment);

        bool alt_tree_owns = sub_trees[alt_tree_id]->query(alt_block_segment);

        if (!main_tree_owns){

          if (alt_tree_owns){
            printf("ERROR: Tree %u reading from segment %llu owned by %u\n", tree_id, alt_block_segment, alt_tree_id);
          } else {
            printf("ERROR: Neither %u or %u own segment %llu\n", tree_id, alt_tree_id, alt_block_segment);
          }

        } else {

          if (alt_tree_owns){
            printf("ERR: Trees %u and %u both own segment %llu\n", tree_id, alt_tree_id, alt_block_segment);
          }

        }

        if (!sub_trees[tree_id]->query(alt_block_segment)){
          printf("Sub tree %u does not own segment %llu\n", tree_id, alt_block_segment);
        }
        //this triggers, yeet
        printf("Block %llu: segment %llu relative %llu not init for malloc: %u != %u\n", block_id, alt_block_segment, relative_block_id, tree_id, alt_tree_id);

        __threadfence();

        continue;
      }

      #endif




        // select block to pull from and get global stats
      uint64_t global_block_id = table->get_global_block_offset(my_block);

    	//TODO: add check here that global block id does not exceed bounds

      uint merged_count = my_block->block_malloc_tree(coalesced_team);

    	uint64_t allocation = my_block->extract_count(coalesced_team, merged_count);

      if (allocation >= 4096 && allocation != ~0ULL){

        #if BETA_DEBUG_PRINTS
        printf("Allocation too big\n");
        #endif

        #if BETA_TRAP_ON_ERR
        asm("trap;");
        #endif

        //invalid alloc, move on.
        //should never occur but continue just in case.
        continue;

      } 

    	//bool should_replace = (allocation == 4095 || allocation == ~0ULL);

      bool should_replace = (allocation == 4095);


      should_replace = coalesced_team.ballot(should_replace);


    	if (should_replace){

    		if (coalesced_team.thread_rank() == 0){
    			replace_block(tree_id, shared_block_storage_index, my_block, local_shared_block_storage);
    		}

    	}

      //sync is necessary for block transistion - illegal to free block until detached.
      __threadfence();
      coalesced_team.sync();

    	if (allocation != ~0ULL){

        if (!my_block->check_valid(merged_count, tree_id)){

          #if BETA_DEBUG_PRINTS
          printf("Gave out wrong offset\n");
          #endif

          free_offset(allocation+global_block_id*4096);

        } else {
          return allocation + global_block_id*4096;
        }

    		

    	}


    	num_attempts+=1;

    }

    return ~0ULL;

  }

}

  // get a new segment for a given tree!
  __device__ bool gather_new_segment(uint16_t tree) {

    // request new segment
    uint64_t new_segment_id = segment_tree->malloc_first();

    if (new_segment_id == veb_tree::fail()) {
      // no segment available - this signals allocator full, return nullptr.
      return false;
    }

    // otherwise, both initialized
    // register segment
    if (!table->setup_segment(new_segment_id, tree)) {
      
      #if BETA_DEBUG_PRINTS
      printf("Failed to acquire updatable segment\n");
      #endif

      #if BETA_TRAP_ON_ERR

      asm("trap;");

      #endif

      //segment_tree->insert_force_update(new_segment_id);
      // abort, but not because no segments are available.
      // this is fine.
      return true;
    }

    __threadfence();

    // insertion with forced flush
    bool inserted = sub_trees[tree]->insert_force_update(new_segment_id);

    __threadfence();

    #if BETA_DEBUG_PRINTS

    bool found = sub_trees[tree]->query(new_segment_id);

    //printf("Sub tree %u owns segment %llu: inserted %d queried %d\n", tree, new_segment_id, inserted, found);

    #endif

    return true;
  }

  // lock given tree to prevent oversubscription
  __device__ bool acquire_tree_lock(uint16_t tree) {

    uint count = atomicAdd((unsigned int *)&locks[tree], 1U);

    if (count >= BETA_TREE_SEGMENT_CAP){

      atomicSub((unsigned int *)&locks[tree], 1U);

      return false;

    }

    return true;

  }

  //return space in tree.
  __device__ bool release_tree_lock(uint16_t tree) {
      
    atomicSub((unsigned int *)&locks[tree], 1U);

  }


  // gather a new block for a tree.
  // this attempts to pull from a memory segment.
  //  and will atteach a new segment if now
  __device__ Block *request_new_block_from_tree(uint16_t tree) {
    int attempts = 0;

    while (attempts < REQUEST_BLOCK_MAX_ATTEMPTS) {
      __threadfence();

      // find first segment available in my sub tree
      uint64_t segment = sub_trees[tree]->find_first_valid_index();

      if (segment == veb_tree::fail()) {

        if (acquire_tree_lock(tree)) {
          bool success = gather_new_segment(tree);

          

          __threadfence();

          // failure to acquire a tree segment means we are full.
          if (!success) {
            // timeouts should be rare...
            // if this failed its more probable that someone else added a
            // segment!

            release_tree_lock(tree);
            __threadfence();
            attempts++;
          }
        }

        __threadfence();

        // for the moment, failures due to not being full enough aren't
        // penalized.
        continue;
      }

      bool last_block = false;

      bool valid = true;

      // valid segment, get new block.
      Block * new_block = table->get_block(segment, tree, last_block, valid);


      #if BETA_DEBUG_PRINTS

      //verify segments match

      if (new_block != nullptr){

        uint64_t block_segment = table->get_segment_from_block_ptr(new_block);

        if (valid && block_segment != segment){

           printf("Segment misaligned when requesting block: %llu != %llu\n", block_segment, segment);
        }

       

      }

      #endif

      if (last_block) {
        // if the segment is fully allocated, it can be detached
        // and returned to the segment tree when empty
        

        bool removed = sub_trees[tree]->remove(segment);

        release_tree_lock(tree);

        #if BETA_DEBUG_PRINTS

        //only worth bringing up if it failed.
        if (!removed){
          printf("Removed segment %llu from tree %u: %d success?\n", segment, tree, removed);
        }

        #endif

        // if (acquire_tree_lock(tree)) {
        //   gather_new_segment(tree);
        //   //release_tree_lock(tree);
        // }

        __threadfence();
      }

      if (!valid){
        free_block(new_block);
        new_block = nullptr;
      }

      if (new_block != nullptr) {
        return new_block;
      }
    }

    // on attempt failures, allocator is full
    return nullptr;
  }

  // return a block to the system
  // this is called by a block once all allocations have been returned.
  __device__ void free_block(Block *block_to_free) {
    
    bool need_to_deregister =
        table->free_block(block_to_free);

    

    if (need_to_deregister) {

      uint64_t segment = table->get_segment_from_block_ptr(block_to_free);

      #if DEBUG_NO_FREE

      #if BETA_DEBUG_PRINTS
      printf("Segment %llu derregister. this is a bug\n", segment);
      #endif

      return;

      #endif



      // returning segment
      // don't need to reset anything, just pull from table and threadfence
      uint16_t tree = table->read_tree_id(segment);


      // pull from tree
      // should be fine, no one can update till this point
      //this should have happened earlier
      if (sub_trees[tree]->remove(segment)){

        release_tree_lock(tree);

        #if BETA_DEBUG_PRINTS
        printf("Failed to properly release segment %llu from tree %u\n", segment, tree);
        #endif

      }

      if (!table->reset_tree_id(segment, tree)){

        #if BETA_DEBUG_PRINTS
        printf("Failed to reset tree id for segment %llu, old ID %u\n", segment, tree);

        #endif
      }
      __threadfence();

      // insert with threadfence
      if (!segment_tree->insert_force_update(segment)){

        #if BETA_DEBUG_PRINTS

        printf("Failed to reinsert segment %llu into segment tree\n", segment);
        #endif

      }
    }
  }

  // return a uint64_t to the system
  __device__ void free_offset(uint64_t malloc) {

    // get block

    uint64_t block_id = malloc/4096;

    Block * my_block = table->get_block_from_global_block_id(block_id);

    if (my_block->block_free()){


      #if !DEBUG_NO_FREE
      my_block->reset_free();
      #endif

      free_block(my_block);

    }

    return;
  }


  //given a uint64_t allocation, return a void * corresponding to the desired memory
  __device__ void * offset_to_allocation(uint64_t offset, uint16_t tree_id){


      //to start, get the segment

      return table->offset_to_allocation(offset, tree_id);

  }


  //given a void * and the known size (expressed as tree id), translate to offset in global space.
  __device__ uint64_t allocation_to_offset(void * allocation, uint16_t tree_id){

    
      return table->allocation_to_offset(allocation, tree_id);
    
  }




  // print useful allocator info.
  // this returns the number of segments owned by each tree
  // and maybe more useful things later.
  __host__ void print_info() {
    my_type *host_version = copy_to_host<my_type>(this);

    uint64_t segments_available = host_version->segment_tree->report_fill();

    uint64_t max_segments = host_version->segment_tree->report_max();

    printf("Allocator sees %llu/%llu segments available\n", segments_available,
           max_segments);

    sub_tree_type **host_trees = copy_to_host<sub_tree_type *>(
        host_version->sub_trees, host_version->num_trees);

    for (int i = 0; i < host_version->num_trees; i++) {
      uint64_t sub_segments = host_trees[i]->report_fill();

      uint64_t sub_max = host_trees[i]->report_max();

      printf("Tree %d: size %lu, owns %llu/%llu\n", i, table->get_tree_alloc_size(i), sub_segments, sub_max);
    }

    uint64_t free_indices = host_version->table->report_free();

    printf("Table reports %llu indices have been freed\n", free_indices);

    uint64_t live_indices = host_version->table->report_live();

    printf("Table reports %llu indices have been used\n", live_indices);

    cudaFreeHost(host_trees);

    cudaFreeHost(host_version);

    this->print_overhead();


    this->print_usage();

  }

  static __host__ __device__ uint64_t get_blocks_per_segment(uint16_t tree) {
    return alloc_table<bytes_per_segment, smallest>::get_blocks_per_segment(
        tree);
  }

  //return maximum # of possible allocations per segment.
  static __host__ __device__ uint64_t get_max_allocations_per_segment(){

    return alloc_table<bytes_per_segment, smallest>::get_max_allocations_per_segment();

  }

  //launch a thread to calculate overhead
  __device__ uint64_t calculate_overhead(){

    uint64_t overhead = 0;

    overhead += sizeof(my_type);

    

    //segment tree
    overhead += segment_tree->calculate_overhead();

    //sub trees

    for (int i =0; i < num_trees; i++){
      overhead += sub_trees[i]->calculate_overhead();
    }

    //local blocks

    for (int i = 0; i < num_trees; i++){

      overhead += local_blocks->get_tree_local_blocks(i)->calculate_overhead();

    }

    //mem table

    overhead += table->calculate_overhead();

    return overhead;


  }

  __host__ void print_overhead(){


    print_overhead_kernel<my_type><<<1,1>>>(this);

    cudaDeviceSynchronize();



  }


  __host__ void print_usage(){

    my_type *host_version = copy_to_host<my_type>(this);


    for (uint16_t i = 0; i < host_version->num_trees; i++){

      print_guided_fill_host(i);

    }


    cudaFreeHost(host_version);


  }

  //generate average fill using the info from the segment tree
  __device__ void print_guided_fill(uint16_t id){


    uint64_t count = 0;

    int malloc_count = 0;
    int free_count = 0;

    uint64_t nblocks = table->get_blocks_per_segment(id);

    for (uint64_t i = 0; i < table->num_segments; i++){
    
      if (sub_trees[id]->query(i)){

        count += 1;
        malloc_count += table->malloc_counters[i];
        free_count += table->free_counters[i];

      }


  }


  printf("Tree %u: %lu live blocks | avg malloc %f / %llu | avg free %f / %llu\n", id, count, 1.0*malloc_count/count, nblocks, 1.0*free_count/count, nblocks);


  }


  __host__ void print_guided_fill_host(uint16_t id){

    print_guided_fill_kernel<my_type><<<1,1>>>(this, id);

  }


};

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard