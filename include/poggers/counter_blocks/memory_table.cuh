#ifndef BETA_MEMORY_TABLE
#define BETA_MEMORY_TABLE
// A CUDA implementation of the alloc table, made by Hunter McCoy
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
#include <poggers/counter_blocks/block.cuh>
#include <poggers/counter_blocks/veb.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/counter_blocks/mixed_counter.cuh>

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif


//This locks the ability of blocks to be returned to the system.
//so blocks accumulate as normal, but segments are not recycled.
//used to test consistency
#define DEBUG_NO_FREE 0

#define BETA_MEM_TABLE_DEBUG 0

#define BETA_TABLE_GLOBAL_READ 1

namespace beta {

namespace allocators {


using mixed_counter = gallatin::utils::mixed_counter;

//get the total # of allocs freed in the system.
//max # blocks - this says something about the current state
template <typename table>
__global__ void count_block_free_kernel(table * alloc_table, uint64_t num_blocks, uint64_t * counter){

  uint64_t tid = poggers::utils::get_tid();

  if (tid >= num_blocks) return;

  uint64_t fill = alloc_table->blocks[tid].free_counter;

  atomicAdd((unsigned long long int *)counter, fill);


}


template <typename table>
__global__ void count_block_live_kernel(table * alloc_table, uint64_t num_blocks, uint64_t * counter){

  uint64_t tid = poggers::utils::get_tid();

  if (tid >= num_blocks) return;

  uint64_t merged_fill = alloc_table->blocks[tid].malloc_counter;

  uint64_t fill = alloc_table->blocks[tid].clip_count(merged_fill);

  if (fill > 4096) fill = 4096;

  atomicAdd((unsigned long long int *)counter, fill);


}

// alloc table associates chunks of memory with trees
// using uint16_t as there shouldn't be that many trees.
// register atomically insert tree num, or registers memory from chunk_tree.

__global__ void betta_init_counters_kernel(int *malloc_counters,
                                           int *free_counters,
                                           int * active_counts,
                                           mixed_counter * mixed_queue_counters,
                                           uint * queue_counters, uint * queue_free_counters,
                                           Block *blocks, uint64_t num_segments,
                                           uint64_t blocks_per_segment) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= num_segments) return;

  malloc_counters[tid] = -1;
  free_counters[tid] = -1;

  active_counts[tid] = -1;


  mixed_queue_counters[tid].init(0);
  queue_counters[tid] = 0;
  queue_free_counters[tid] = 0;

  uint64_t base_offset = blocks_per_segment * tid;

  for (uint64_t i = 0; i < blocks_per_segment; i++) {
    Block *my_block = &blocks[base_offset + i];

    my_block->init();
  }

  __threadfence();
}

// The alloc table owns all blocks live in the system
// and information for each segment
template <uint64_t bytes_per_segment, uint64_t min_size>
struct alloc_table {
  using my_type = alloc_table<bytes_per_segment, min_size>;

  // the tree id of each chunk
  uint16_t *chunk_ids;

  // list of all blocks live in the system.
  Block *blocks;

  //queues hold freed blocks for fast turnaround
  Block ** queues;


  mixed_counter * mixed_queue_counters;

  //queue counters record position in queue
  uint * queue_counters;

  //free counters holds which index newly freed blocks are emplaced.
  uint * queue_free_counters;

  //active counts make sure that the # of blocks in movement are acceptable.
  int * active_counts;


  // pair of counters for each segment to track use.
  int *malloc_counters;
  int *free_counters;

  // all memory live in the system.
  char *memory;

  uint64_t num_segments;

  uint64_t blocks_per_segment;

  // generate structure on device and return pointer.
  static __host__ my_type *generate_on_device(uint64_t max_bytes) {
    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    uint64_t num_segments =
        poggers::utils::get_max_chunks<bytes_per_segment>(max_bytes);

    //printf("Booting memory table with %llu chunks\n", num_segments);

    uint16_t *ext_chunks;

    cudaMalloc((void **)&ext_chunks, sizeof(uint16_t) * num_segments);

    cudaMemset(ext_chunks, ~0U, sizeof(uint16_t) * num_segments);

    host_version->chunk_ids = ext_chunks;

    host_version->num_segments = num_segments;

    // init blocks

    uint64_t blocks_per_segment = bytes_per_segment / (min_size * 4096);

    Block *ext_blocks;

    cudaMalloc((void **)&ext_blocks,
               sizeof(Block) * blocks_per_segment * num_segments);

    cudaMemset(ext_blocks, 0U,
               sizeof(Block) * (num_segments * blocks_per_segment));

    host_version->blocks = ext_blocks;

    host_version->blocks_per_segment = blocks_per_segment;


    Block ** ext_queues;
    cudaMalloc((void **)&ext_queues, sizeof(Block *)*blocks_per_segment*num_segments);

    host_version->queues = ext_queues;

    host_version->memory = poggers::utils::get_device_version<char>(
        bytes_per_segment * num_segments);

    cudaMemset(host_version->memory, 0, bytes_per_segment*num_segments);

    // generate counters and set them to 0.
    host_version->active_counts = poggers::utils::get_device_version<int>(num_segments);


    host_version->mixed_queue_counters = poggers::utils::get_device_version<mixed_counter>(num_segments);
    host_version->queue_counters = poggers::utils::get_device_version<uint>(num_segments);
    host_version->queue_free_counters = poggers::utils::get_device_version<uint>(num_segments);



    host_version->malloc_counters =
        poggers::utils::get_device_version<int>(num_segments);
    host_version->free_counters =
        poggers::utils::get_device_version<int>(num_segments);
    betta_init_counters_kernel<<<(num_segments - 1) / 512 + 1, 512>>>(
        host_version->malloc_counters, host_version->free_counters,
        host_version->active_counts, host_version->mixed_queue_counters,
        host_version->queue_counters, host_version->queue_free_counters,
        host_version->blocks, num_segments,
        blocks_per_segment);

    GPUErrorCheck(cudaDeviceSynchronize());


   




    // move to device and free host memory.
    my_type *dev_version;

    cudaMalloc((void **)&dev_version, sizeof(my_type));

    cudaMemcpy(dev_version, host_version, sizeof(my_type),
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaFreeHost(host_version);

    return dev_version;
  }

  // return memory to GPU
  static __host__ void free_on_device(my_type *dev_version) {
    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    cudaMemcpy(host_version, dev_version, sizeof(my_type),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(host_version->blocks);

    cudaFree(host_version->chunk_ids);

    cudaFree(host_version->memory);

    cudaFree(host_version->malloc_counters);

    cudaFree(dev_version);

    cudaFreeHost(host_version);
  }

  // register a tree component
  __device__ void register_tree(uint64_t segment, uint16_t id) {
    if (segment >= num_segments) {

      #if BETA_MEM_TABLE_DEBUG
      printf("Chunk issue: %llu > %llu\n", segment, num_segments);
      #endif

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

    }

    chunk_ids[segment] = id;
  }

  // register a segment from the table.
  __device__ void register_size(uint64_t segment, uint16_t size) {
    if (segment >= num_segments) {

      #if BETA_MEM_TABLE_DEBUG
      printf("Chunk issue\n");
      #endif

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

    }

    size += 16;

    chunk_ids[segment] = size;
  }

  // get the void pointer to the start of a segment.
  __device__ char *get_segment_memory_start(uint64_t segment) {
    return memory + bytes_per_segment * segment;
  }

  // claim segment
  // to claim segment
  // set tree ID, set malloc_counter
  // free_counter is set
  // return;
  __device__ bool setup_segment(uint64_t segment, uint16_t tree_id) {
    uint64_t tree_alloc_size = get_tree_alloc_size(tree_id);

    // should stop interlopers
    bool did_set = set_tree_id(segment, tree_id);

    int num_blocks = get_blocks_per_segment(tree_id);

#if BETA_MEM_TABLE_DEBUG

    if (!did_set){
      printf("Failed to set tree id for segment %llu\n", segment);

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

    }

    int old_free_count =
        atomicExch(&free_counters[segment], -1);

    if (old_free_count != -1) {
      printf(
          "Memory free counter for segment %llu not properly reset: value is "
          "%d\n",
          segment, old_free_count);

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

    }



#endif


    //Segments now give out negative counters...
    //this allows us to A) specify # of blocks exactly on construction.
    // and B) still give out exact addresses when requesting (still 1 atomic.)
    //the trigger for a failed block alloc is going negative

    //int old_free = atomicExch(&free_counters[segment], num_blocks-1);
    //int old_malloc = atomicExch(&malloc_counters[segment], num_blocks-1);

    //modification, boot queue elements
    //as items can always interact with this, we simply reset.
    //init with blocks per segment so that mallocs always understand a true count
    int old_active_count = atomicExch(&active_counts[segment], num_blocks-1);

    //init queue counters.
    mixed_queue_counters[segment].init(num_blocks);

    //mixed_queue_counters[segment].atomicInit(num_blocks);
    __threadfence();

    atomicExch(&queue_counters[segment], 0);
    atomicExch(&queue_free_counters[segment], 0);



    #if BETA_MEM_TABLE_DEBUG

    if (old_active_count != -1){
      printf("Old active count has live threads: %d\n", old_active_count);
    }


    // if (old_malloc < 0){
    //   printf("Did not fully reset segment %llu: %d malloc %d free\n", segment, old_malloc, old_free);

    // }
    #endif

    // if (old_malloc >= 0){
    //   #if BETA_TRAP_ON_ERR
    //     asm("trap;");
    //   #endif
    // }


    // gate to init is init_new_universe
    return true;
  }


  // set the tree id of a segment atomically
  //  returns true on success.
  __device__ bool set_tree_id(uint64_t segment, uint16_t tree_id) {
    return (atomicCAS((unsigned short int *)&chunk_ids[segment],
                      (unsigned short int)~0U,
                      (unsigned short int)tree_id) == (unsigned short int)~0U);
  }

  // atomically read tree id.
  // this may be faster with global load lcda instruction
  __device__ uint16_t read_tree_id(uint64_t segment) {

    #if BETA_TABLE_GLOBAL_READ

      return poggers::utils::global_read_uint16_t(&chunk_ids[segment]);

    #else

      return atomicCAS((unsigned short int *)&chunk_ids[segment],
                (unsigned short int)~0U, (unsigned short int)~0U);

    #endif

  }

  // return tree id to ~0
  __device__ bool reset_tree_id(uint64_t segment, uint16_t tree_id) {
    return (atomicCAS((unsigned short int *)&chunk_ids[segment],
                      (unsigned short int)tree_id,
                      (unsigned short int)~0U) == (unsigned short int)tree_id);
  }

  // atomic wrapper to get block
  __device__ int get_block_from_segment(uint64_t segment) {
    return atomicSub(&malloc_counters[segment], 1);
  }

  // atomic wrapper to free block.
  __device__ int return_block_to_segment(uint64_t segment) {


    #if BETA_MEM_TABLE_DEBUG

    int free_count = atomicSub(&free_counters[segment], 1);

    int malloc_count = atomicCAS(&malloc_counters[segment], 0, 0);

    if (malloc_count >= free_count){
      printf("Mismatch: malloc %d >= freed %d\n", malloc_count, free_count);

      #if BETA_TRAP_ON_ERR
      asm("trap;");
      #endif

    }

    return free_count;

    #else 

    return atomicSub(&free_counters[segment], 1);

    #endif
  }



  /******
  Set of helper functions to control queue entry and exit
  
  These allow threads to request slots from the queue and check if the queue is entirely full

  or entirely empty. 

  ******/

  //pull a slot from the segment
  //this acts as a gate over the malloc counters

  __device__ int return_slot_to_segment(uint64_t segment){
    //return atomicAdd(&active_counts[segment], 1);
    return (int) mixed_queue_counters[segment].release();
  }

  //helper to check if block is entirely free.
  //requires you to have a valid tree_id
  __device__ bool all_blocks_free(int active_count, uint64_t blocks_per_segment){

    return (active_count == blocks_per_segment-1);

  }

  //check if the count for a thread is valid
  //current condition is that negative numbers represent invalid requests.
  __device__ bool active_count_valid(int active_count){

    return (active_count >= 0);

  }

  __device__ uint64_t get_mixed_queue_position(uint64_t segment, bool & last){

    return mixed_queue_counters[segment].count_and_increment_check_last(last);

  }

  __device__ uint increment_queue_position(uint64_t segment){

    return atomicAdd(&queue_counters[segment], 1);

  }

  __device__ uint increment_free_queue_position(uint64_t segment){

    return atomicAdd(&queue_free_counters[segment], 1);

  }

  // request a segment from a block
  // this verifies that the segment is initialized correctly
  // and returns nullptr on failure.
  __device__ Block *get_block(uint64_t segment_id, uint16_t tree_id,
                              bool &empty, bool &valid) {


    //empty = false;

    uint64_t active_count = get_mixed_queue_position(segment_id, empty);

    if (active_count == ~0ULL){
      return nullptr;
    }

    //if global tree id's don't match, discard.
    uint16_t global_tree_id = read_tree_id(segment_id);

    //Discard needs to acquire and then release segment
    //can combine max_count and current_tree_ID into one value.
    //that is a job for a different day

    uint64_t blocks_in_segment = get_blocks_per_segment(global_tree_id);

    //first, pull block.

    Block * my_block;

    if (active_count < blocks_in_segment){

      my_block = get_block_from_global_block_id(segment_id*blocks_per_segment+active_count);

    } else {


      int queue_pos_wrapped = active_count % blocks_in_segment;

      //swap out the queue element for nullptr.
      my_block = (Block *) atomicExch((unsigned long long int *)&queues[segment_id*blocks_per_segment+queue_pos_wrapped], 0ULL);

    }

    //only valid if they match.
    valid = (global_tree_id == tree_id);


    my_block->init_malloc(tree_id);

    return my_block;
    
    }

  //given a global block_id, return the block
  __device__ Block * get_block_from_global_block_id(uint64_t global_block_id){

  	return &blocks[global_block_id];

  }

  // snap a block back to its segment
  // needed for returning
  __device__ uint64_t get_segment_from_block_ptr(Block *block) {
    // this returns the stride in blocks
    uint64_t offset = (block - blocks);

    return offset / blocks_per_segment;
  }

  // get relative offset of a block in its segment.
  __device__ int get_relative_block_offset(Block *block) {
    uint64_t offset = (block - blocks);

    return offset % blocks_per_segment;
  }

  // given a pointer, find the associated block for returns
  // not yet implemented
  __device__ Block *get_block_from_ptr(void *ptr) {}

  // given a pointer, get the segment the pointer belongs to
  __device__ uint64_t get_segment_from_ptr(void *ptr) {
    uint64_t offset = ((char *)ptr) - memory;

    return offset / bytes_per_segment;
  }

  __device__ uint64_t get_segment_from_offset(uint64_t offset){

    return offset/get_max_allocations_per_segment();

  }

  // get the tree the segment currently belongs to
  __device__ int get_tree_from_segment(uint64_t segment) {
    return chunk_ids[segment];
  }

  // helper function for moving from power of two exponent to index
  static __host__ __device__ uint64_t get_p2_from_index(int index) {
    return (1ULL) << index;
  }

  // given tree id, return size of allocations.
  static __host__ __device__ uint64_t get_tree_alloc_size(uint16_t tree) {
    // scales up by smallest.
    return min_size * get_p2_from_index(tree);
  }

  // get relative position of block in list of all blocks
  __device__ uint64_t get_global_block_offset(Block *block) {
    return block - blocks;
  }

  // get max blocks per segment when formatted to a given tree size.
  static __host__ __device__ uint64_t get_blocks_per_segment(uint16_t tree) {
    uint64_t tree_alloc_size = get_tree_alloc_size(tree);

    return bytes_per_segment / (tree_alloc_size * 4096);
  }

  //get maximum # of allocations per segment
  //useful for converting alloc offsets into void *
  static __host__ __device__ uint64_t get_max_allocations_per_segment(){

  	//get size of smallest tree
  	return bytes_per_segment / min_size;

  }

  __device__ void * offset_to_allocation(uint64_t allocation, uint16_t tree_id){

  	uint64_t segment_id = allocation/get_max_allocations_per_segment();

  	uint64_t relative_offset = allocation % get_max_allocations_per_segment();

  	char * segment_mem_start = get_segment_memory_start(segment_id);


  	uint64_t alloc_size = get_tree_alloc_size(tree_id);

  	return (void *) (segment_mem_start + relative_offset*alloc_size);


  }


  //given a known tree id, snap an allocation back to the correct offset
  __device__ uint64_t allocation_to_offset(void * alloc, uint16_t tree_id){


      uint64_t byte_offset = (uint64_t) ((char *) alloc - memory);

      //segment id_should agree with upper function.
      uint64_t segment_id = byte_offset/bytes_per_segment;


      #if BETA_DEBUG_PRINTS

      uint64_t alt_segment = get_segment_from_ptr(alloc);

      if (segment_id != alt_segment){
        printf("Mismatch on segments in allocation to offset, %llu != %llu\n", segment_id, alt_segment)

        #if BETA_TRAP_ON_ERR
        asm("trap;");
        #endif
      }



      #endif





      char * segment_start = (char *) get_segment_memory_start(segment_id);

      uint64_t segment_byte_offset = (uint64_t) ((char *) alloc - segment_start);

      return segment_byte_offset/get_tree_alloc_size(tree_id) + segment_id*get_max_allocations_per_segment();



  }


  //return which index in the queue structure is valid
  //and start swap
  //this does not increment find index yet.
  __device__ uint reserve_segment_slot(Block * block_ptr, uint64_t & segment, uint16_t & global_tree_id, uint64_t & num_blocks){


    //get enqueue position.
    uint enqueue_position = increment_free_queue_position(segment) % num_blocks;

    //swap into queue
    atomicExch((unsigned long long int *)&queues[segment*blocks_per_segment+enqueue_position], (unsigned long long int) block_ptr);


    __threadfence();

    return enqueue_position;

  }


  //once the messy logic of the tree reset is done, clean up
  __device__ bool finish_freeing_block(uint64_t segment, uint64_t num_blocks){

    int return_id = return_slot_to_segment(segment);

    if (all_blocks_free(return_id, num_blocks)){

      if (atomicCAS(&active_counts[segment], num_blocks-1, -1) == num_blocks-1){

        //exclusive owner
        return true;
      }
    }

    return false;

  }

  // free block, returns true if this block was the last section needed.
  //split this into two sections
  //section one reserves free index
  //  returns index and flag if segment should be reinserted
  //Section two adds the item back.
  //this guarantees that the system is visible *before* being returned.
  __device__ bool free_block(Block *block_ptr) {


    uint64_t segment = get_segment_from_block_ptr(block_ptr);

    uint16_t global_tree_id = read_tree_id(segment);

    uint64_t num_blocks = get_blocks_per_segment(global_tree_id);


    //get enqueue position.
    uint enqueue_position = increment_free_queue_position(segment) % num_blocks;

    //swap into queue
    atomicExch((unsigned long long int *)&queues[segment*blocks_per_segment+enqueue_position], (unsigned long long int) block_ptr);


    __threadfence();

    //determines how many other blocks are live, and signals to the system that re-use is possible
    int return_id = return_slot_to_segment(segment);

    if (all_blocks_free(return_id, num_blocks)){

      if (atomicCAS(&active_counts[segment], num_blocks-1, -1) == num_blocks-1){

        //exclusive owner
        return true;
      }
    }

    return false;

}



  __host__ uint64_t report_free(){

    uint64_t * counter;

    cudaMallocManaged((void **)&counter, sizeof(uint64_t));

    cudaDeviceSynchronize();

    counter[0] = 0;

    cudaDeviceSynchronize();


    //this will probs break

    uint64_t local_num_segments;

    cudaMemcpy(&local_num_segments, &this->num_segments, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    uint64_t local_blocks_per_segment;

    cudaMemcpy(&local_blocks_per_segment, &this->blocks_per_segment, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    uint64_t total_num_blocks = local_blocks_per_segment*local_num_segments;

    count_block_free_kernel<my_type><<<(total_num_blocks-1)/256+1,256>>>(this, total_num_blocks, counter);

    cudaDeviceSynchronize();

    uint64_t return_val = counter[0];

    cudaFree(counter);

    return return_val;

  }

  __host__ uint64_t report_live(){

    uint64_t * counter;

    cudaMallocManaged((void **)&counter, sizeof(uint64_t));

    cudaDeviceSynchronize();

    counter[0] = 0;

    cudaDeviceSynchronize();


    //this will probs break

    uint64_t local_num_segments;

    cudaMemcpy(&local_num_segments, &this->num_segments, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    uint64_t local_blocks_per_segment;

    cudaMemcpy(&local_blocks_per_segment, &this->blocks_per_segment, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    uint64_t total_num_blocks = local_blocks_per_segment*local_num_segments;

    count_block_live_kernel<my_type><<<(total_num_blocks-1)/256+1,256>>>(this, total_num_blocks, counter);

    cudaDeviceSynchronize();

    uint64_t return_val = counter[0];

    cudaFree(counter);

    return return_val;

  }


  __device__ uint64_t calculate_overhead(){

    return sizeof(my_type) + num_segments*(8 + blocks_per_segment*sizeof(Block));

  }


};

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard