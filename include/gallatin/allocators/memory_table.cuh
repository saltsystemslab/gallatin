#ifndef GALLATIN_MEMORY_TABLE
#define GALLATIN_MEMORY_TABLE
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
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/allocators/block.cuh>
#include <gallatin/allocators/veb.cuh>
#include <gallatin/allocators/murmurhash.cuh>


//This locks the ability of blocks to be returned to the system.
//so blocks accumulate as normal, but segments are not recycled.
//used to test consistency
#define DEBUG_NO_FREE 0

#define GALLATIN_MEM_TABLE_DEBUG 0

namespace gallatin {

namespace allocators {


enum Gallatin_memory_type {device_only, host_only, managed};


//get the total # of allocs freed in the system.
//max # blocks - this says something about the current state
template <typename table>
__global__ void count_block_free_kernel(table * alloc_table, uint64_t num_blocks, uint64_t * counter){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= num_blocks) return;

  uint64_t fill = alloc_table->blocks[tid].free_counter;

  atomicAdd((unsigned long long int *)counter, fill);


}


template <typename table>
__global__ void count_block_live_kernel(table * alloc_table, uint64_t num_blocks, uint64_t * counter){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= num_blocks) return;

  uint64_t merged_fill = alloc_table->blocks[tid].malloc_counter;

  uint64_t fill = alloc_table->blocks[tid].clip_count(merged_fill);

  if (fill > 4096) fill = 4096;

  atomicAdd((unsigned long long int *)counter, fill);


}

// alloc table associates chunks of memory with trees
// using uint16_t as there shouldn't be that many trees.
// register atomically insert tree num, or registers memory from chunk_tree.

static __global__ void gallatin_init_counters_kernel(
                                           int * active_counts,
                                           uint * queue_counters, uint * queue_free_counters,
                                           Block *blocks, Block ** queues, uint64_t num_segments,
                                           uint64_t blocks_per_segment) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= num_segments) return;

  active_counts[tid] = -1;

  queue_counters[tid] = 0;
  queue_free_counters[tid] = 0;

  uint64_t base_offset = blocks_per_segment * tid;

  for (uint64_t i = 0; i < blocks_per_segment; i++) {
    Block *my_block = &blocks[base_offset + i];

    my_block->init();

    queues[base_offset+i] = nullptr;

  }
  // No __threadfence needed — kernel completion is the synchronization
  // boundary; subsequent kernels in the same stream observe all writes.
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

  //queue counters record position in queue
  uint * queue_counters;

  //free counters holds which index newly freed blocks are emplaced.
  uint * queue_free_counters;

  //active counts make sure that the # of blocks in movement are acceptable.
  int * active_counts;


  // all memory live in the system.
  char *memory;

  uint64_t num_segments;

  uint64_t blocks_per_segment;

  Gallatin_memory_type memory_control;


  // generate structure on device and return pointer.
  static __host__ my_type *generate_on_device(uint64_t max_bytes,  Gallatin_memory_type ext_memory_control=device_only) {
    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    uint64_t num_segments =
        gallatin::utils::get_max_chunks<bytes_per_segment>(max_bytes);

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


    if (ext_memory_control == device_only){

      host_version->memory = gallatin::utils::get_device_version<char>(
        bytes_per_segment * num_segments);

      cudaMemset(host_version->memory, 0, bytes_per_segment*num_segments);


    } else if (ext_memory_control == host_only){

      char * host_memory;
      char * dev_ptr_host_memory;

      cudaDeviceProp prop;
      GPUErrorCheck(cudaGetDeviceProperties(&prop, 0));
      if (!prop.canMapHostMemory)
      {
          throw std::runtime_error{"Device does not supported mapped memory."};
      }

      GPUErrorCheck(cudaHostAlloc((void **)&host_memory, bytes_per_segment*num_segments, cudaHostAllocMapped));


      //GPUErrorCheck(cudaHostAlloc((void **)&host_memory, bytes_per_segment*num_segments, cudaHostAllocDefault));

      //memset(host_memory, 0, bytes_per_segment*num_segments);

      CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&dev_ptr_host_memory, host_memory, 0));

      //cudaMemset(dev_ptr_host_memory, 0, bytes_per_segment*num_segments);

      gallatin::utils::clear_device_host_memory(dev_ptr_host_memory, bytes_per_segment*num_segments);

      host_version->memory = dev_ptr_host_memory;

    } else if (ext_memory_control == managed) {


      char * host_memory;

      cudaMallocManaged((void **)&host_memory, bytes_per_segment*num_segments);

      // Memory-oversubscription placement hint (host-scale-out): keep the pool
      // device-resident until it oversubscribes VRAM, then let the driver
      // LRU-spill pages to host. Pure hint; skipped where the device lacks
      // concurrentManagedAccess.
      {
        int _mm_dev = 0; cudaGetDevice(&_mm_dev);
        int _mm_concurrent = 0;
        cudaDeviceGetAttribute(&_mm_concurrent, cudaDevAttrConcurrentManagedAccess, _mm_dev);
        if (_mm_concurrent) {
          cudaMemAdvise(host_memory, bytes_per_segment*num_segments, cudaMemAdviseSetPreferredLocation, _mm_dev);
          cudaMemAdvise(host_memory, bytes_per_segment*num_segments, cudaMemAdviseSetAccessedBy, _mm_dev);
        }
      }

      // The boot zero-fill touches every page, forcing the ENTIRE pool resident
      // (and host-allocated) at boot — untenable for a very large (e.g. TB-scale)
      // managed pool. Segment memory is raw user storage (block bitmaps live in
      // the separate, still-zeroed `blocks` array), so skipping the fill is safe
      // for callers that do not rely on zeroed allocations. Define
      // GALLATIN_MANAGED_SKIP_BOOT_MEMSET to skip it.
#ifndef GALLATIN_MANAGED_SKIP_BOOT_MEMSET
      cudaMemset(host_memory, 0, bytes_per_segment*num_segments);
#endif

      host_version->memory = host_memory;


    }



    // generate counters and set them to 0.
    host_version->active_counts = gallatin::utils::get_device_version<int>(num_segments);

    host_version->queue_counters = gallatin::utils::get_device_version<uint>(num_segments);
    host_version->queue_free_counters = gallatin::utils::get_device_version<uint>(num_segments);



    gallatin_init_counters_kernel<<<(num_segments - 1) / 512 + 1, 512>>>(
        host_version->active_counts,
        host_version->queue_counters, host_version->queue_free_counters,
        host_version->blocks, host_version->queues, num_segments,
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


    // generate structure on device and return pointer.
  static __host__ my_type *generate_on_device_nowait(uint64_t max_bytes, Gallatin_memory_type ext_memory_control=device_only) {
    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    uint64_t num_segments =
        gallatin::utils::get_max_chunks<bytes_per_segment>(max_bytes);

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

    if (ext_memory_control == device_only){

      host_version->memory = gallatin::utils::get_device_version<char>(
        bytes_per_segment * num_segments);

      cudaMemset(host_version->memory, 0, bytes_per_segment*num_segments);


    } else if (ext_memory_control == host_only){

      char * host_memory;
      char * dev_ptr_host_memory;

      cudaDeviceProp prop;
      GPUErrorCheck(cudaGetDeviceProperties(&prop, 0));
      if (!prop.canMapHostMemory)
      {
          throw std::runtime_error{"Device does not supported mapped memory."};
      }

      //GPUErrorCheck(cudaHostAlloc((void **)&host_memory, bytes_per_segment*num_segments, cudaHostAllocDefault));

      GPUErrorCheck(cudaHostAlloc((void **)&host_memory, bytes_per_segment*num_segments, cudaHostAllocMapped));

      memset(host_memory, 0, bytes_per_segment*num_segments);

      CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&dev_ptr_host_memory, host_memory, 0));

      host_version->memory = dev_ptr_host_memory;

    } else if (ext_memory_control == managed) {


      char * host_memory;

      cudaMallocManaged((void **)&host_memory, bytes_per_segment*num_segments);

      // Memory-oversubscription placement hint (host-scale-out): keep the pool
      // device-resident until it oversubscribes VRAM, then let the driver
      // LRU-spill pages to host. Pure hint; skipped where the device lacks
      // concurrentManagedAccess.
      {
        int _mm_dev = 0; cudaGetDevice(&_mm_dev);
        int _mm_concurrent = 0;
        cudaDeviceGetAttribute(&_mm_concurrent, cudaDevAttrConcurrentManagedAccess, _mm_dev);
        if (_mm_concurrent) {
          cudaMemAdvise(host_memory, bytes_per_segment*num_segments, cudaMemAdviseSetPreferredLocation, _mm_dev);
          cudaMemAdvise(host_memory, bytes_per_segment*num_segments, cudaMemAdviseSetAccessedBy, _mm_dev);
        }
      }

      // The boot zero-fill touches every page, forcing the ENTIRE pool resident
      // (and host-allocated) at boot — untenable for a very large (e.g. TB-scale)
      // managed pool. Segment memory is raw user storage (block bitmaps live in
      // the separate, still-zeroed `blocks` array), so skipping the fill is safe
      // for callers that do not rely on zeroed allocations. Define
      // GALLATIN_MANAGED_SKIP_BOOT_MEMSET to skip it.
#ifndef GALLATIN_MANAGED_SKIP_BOOT_MEMSET
      cudaMemset(host_memory, 0, bytes_per_segment*num_segments);
#endif

      host_version->memory = host_memory;


    }

    host_version->memory_control = ext_memory_control;

    // generate counters and set them to 0.
    host_version->active_counts = gallatin::utils::get_device_version<int>(num_segments);

    host_version->queue_counters = gallatin::utils::get_device_version<uint>(num_segments);
    host_version->queue_free_counters = gallatin::utils::get_device_version<uint>(num_segments);

    gallatin_init_counters_kernel<<<(num_segments - 1) / 512 + 1, 512>>>(
        host_version->active_counts, 
        host_version->queue_counters, host_version->queue_free_counters,
        host_version->blocks, host_version->queues, num_segments,
        blocks_per_segment);

    //GPUErrorCheck(cudaDeviceSynchronize());


    // move to device and free host memory.
    my_type *dev_version;

    cudaMalloc((void **)&dev_version, sizeof(my_type));

    cudaMemcpy(dev_version, host_version, sizeof(my_type),
               cudaMemcpyHostToDevice);

    //cudaDeviceSynchronize();

    cudaFreeHost(host_version);

    return dev_version;
  }

  // return memory/resources to GPU
  static __host__ void free_on_device(my_type *dev_version) {
    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    cudaMemcpy(host_version, dev_version, sizeof(my_type),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(host_version->blocks);

    cudaFree(host_version->chunk_ids);


    if (host_version->memory_control == device_only || host_version->memory_control == managed){
      cudaFree(host_version->memory);
    } else {
      cudaFreeHost(host_version->memory);
    }
    

    cudaFree(dev_version);

    cudaFreeHost(host_version);

  }

  // get the void pointer to the start of a segment.
  __device__ char *get_segment_memory_start(uint64_t segment) {
    return memory + bytes_per_segment * segment;
  }

  // Claim a segment for a tree.
  //
  // Initialization order matters: the tree_id CAS is the *publication* event,
  // so it must happen LAST, with release ordering. Any thread that later sees
  // a non-sentinel tree_id (via the acquire-load in read_tree_id) is then
  // guaranteed to also see the per-segment state initialized below.
  __device__ bool setup_segment(uint64_t segment, uint16_t tree_id) {
    int num_blocks = get_blocks_per_segment(tree_id);

    // 1. Initialize per-segment state. Plain stores: the release-CAS at the
    // end orders these for any acquire-side reader.
    for (int i = 0; i < num_blocks; i++) {
      queues[segment * blocks_per_segment + i] = nullptr;
    }

    queue_counters[segment] = 0;
    queue_free_counters[segment] = 0;
    active_counts[segment] = num_blocks - 1;

    // 2. Publish. set_tree_id is a release-CAS: ~0 -> tree_id. Returns false
    // if some other thread already claimed the segment (shouldn't happen
    // while the per-tree lock is held, but the CAS keeps the invariant honest).
    return set_tree_id(segment, tree_id);
  }


  // Publish a segment as belonging to `tree_id`. Release ordering pairs with
  // the acquire-load in read_tree_id, so prior initialization writes become
  // visible alongside the new tree_id.
  __device__ bool set_tree_id(uint64_t segment, uint16_t tree_id) {
    uint16_t expected = static_cast<uint16_t>(~0U);
    return gallatin::utils::cas_release<uint16_t>(&chunk_ids[segment], expected,
                                                  tree_id);
  }

  // Acquire-load: synchronizes with the release-CAS in set_tree_id /
  // reset_tree_id. If the returned value is anything other than the ~0
  // sentinel, all writes the publisher made before that CAS are visible to
  // this thread.
  __device__ uint16_t read_tree_id(uint64_t segment) {
    return gallatin::utils::load_acquire(&chunk_ids[segment]);
  }

  // Publish: this segment is no longer owned by `tree_id` (~0 sentinel).
  // Release ordering — any thread that subsequently observes the sentinel via
  // an acquire load will also see any state the caller wrote before reset.
  __device__ bool reset_tree_id(uint64_t segment, uint16_t tree_id) {
    uint16_t expected = tree_id;
    return gallatin::utils::cas_release<uint16_t>(
        &chunk_ids[segment], expected, static_cast<uint16_t>(~0U));
  }



  /******
  Set of helper functions to control queue entry and exit
  
  These allow threads to request slots from the queue and check if the queue is entirely full

  or entirely empty. 

  ******/

  //pull a slot from the segment
  //this acts as a gate over the malloc counters.
  __device__ int get_slot_in_segment(uint64_t segment){
    return atomicSub(&active_counts[segment], 1);
  }

  __device__ int return_slot_to_segment(uint64_t segment){
    return atomicAdd(&active_counts[segment], 1);
  }

  //helper to check if block is entirely free.
  //requires you to have a valid tree_id
  __device__ bool all_blocks_free(int active_count, uint64_t blocks_per_segment){

    return (active_count == blocks_per_segment-2);

  }

  //check if the count for a thread is valid
  //current condition is that negative numbers represent invalid requests.
  __device__ bool active_count_valid(int active_count){

    return (active_count >= 0);

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
                              bool &empty) {


    empty = false;


    //precondition that if it's available we go for it...
    int active_count = get_slot_in_segment(segment_id);

    if (!active_count_valid(active_count)){

      return_slot_to_segment(segment_id);

      return nullptr;

    }

    //if global tree id's don't match, discard.
    uint16_t global_tree_id = read_tree_id(segment_id);

    // tree changed in interim - this can happen in correct behavior.
    // we correct by releasing back to the system, potentially rolling the segment back.
    if (global_tree_id != tree_id) {

      #if GALLATIN_MEM_TABLE_DEBUG

      printf("Segment %llu: Read old tree value: %u != %u\n", segment_id, tree_id, global_tree_id);

      #endif

      //slot can go back to a worthy thread
      //this saves the reset having to be pushed to the main manager.
      return_slot_to_segment(segment_id);

      return nullptr;
    }


    uint64_t blocks_in_segment = get_blocks_per_segment(tree_id);

    //if we have a valid spot, a queue position must exist
    int queue_pos = increment_queue_position(segment_id);

    Block * my_block;

    if (queue_pos < blocks_in_segment){

      my_block = get_block_from_global_block_id(segment_id*blocks_per_segment+queue_pos);

    } else {

      int queue_pos_wrapped = queue_pos % blocks_in_segment;

      // Acquire-exchange: pair with the release-exchange in reserve_segment_slot.
      // Any non-null pointer we take out of the queue carries the freeing
      // thread's reset_free() write with it.
      //
      // Retry on nullptr: a producer at the same position has reserved its
      // slot but not yet exchanged the block_ptr into place. active_counts
      // pacing guarantees a producer exists for this slot; we just need to
      // wait for it to publish.
      Block * empty_marker = nullptr;
      do {
        my_block = gallatin::utils::exchange_acquire<Block *>(
            &queues[segment_id * blocks_per_segment + queue_pos_wrapped],
            empty_marker);
      } while (my_block == nullptr);

    }


    my_block->init_malloc(tree_id);

    if (active_count == 0) {
      empty = true;
    }

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


      #if GALLATIN_MEM_TABLE_DEBUG

      uint64_t alt_segment = get_segment_from_ptr(alloc);

      if (segment_id != alt_segment){
        printf("Mismatch on segments in allocation to offset, %llu != %llu\n", segment_id, alt_segment);

        #if GALLATIN_TRAP_ON_ERR
        asm volatile ("trap;");
        #endif
      }

      #endif





      char * segment_start = (char *) get_segment_memory_start(segment_id);

      uint64_t segment_byte_offset = (uint64_t) ((char *) alloc - segment_start);

      return segment_byte_offset/get_tree_alloc_size(tree_id) + segment_id*get_max_allocations_per_segment();



  }

  // Publish a freed block back into the per-segment queue.
  //
  // Release-exchange on the slot pairs with the acquire-exchange in get_block:
  // any consumer that pulls this block out of the queue is also guaranteed to
  // see the freeing thread's writes — most importantly the reset_free() that
  // zeroed the block's free_counter.
  //
  // No explicit producer-side serialization: the active_counts pacing bounds
  // (num_active_mallocs - num_active_frees) within [0, blocks_per_segment], so
  // each ring slot has at most one in-flight producer at any moment. Consumers
  // that observe nullptr in their target slot are racing a slow producer and
  // simply retry (see get_block).
  __device__ uint reserve_segment_slot(Block *block_ptr, uint64_t &segment,
                                       uint16_t &global_tree_id,
                                       uint64_t &num_blocks) {
    uint current_enqueue_position = increment_free_queue_position(segment);
    uint live_enqueue_position = current_enqueue_position % num_blocks;

    gallatin::utils::exchange_release<Block *>(
        &queues[segment * blocks_per_segment + live_enqueue_position],
        block_ptr);

    return live_enqueue_position;
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

  __device__ uint64_t get_bytes_per_segment(){
    return bytes_per_segment;
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

    //overhead per segment
    //4 bytes active count
    //4 bytes queue_inc
    //4 bytes_queue_dec
    //2 bytes tree_id
    //+ blocks_per_segment*sizeof(block) 
    //+ blocks+per_segment*sizeof(block *)  - this is the queue structure.


    return sizeof(my_type) + num_segments*(14 + blocks_per_segment*(sizeof(Block)+sizeof(Block *)));

  }

  __device__ bool owns_allocation(void * alloc){


    uint64_t byte_difference = ( (char *) alloc - (char *) memory);

    return (byte_difference < num_segments*bytes_per_segment);

  }


};

}  // namespace allocators

}  // namespace gallatin

#endif  // End of VEB guard