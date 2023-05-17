#ifndef BETA_MEMORY_TABLE
#define BETA_MEMORY_TABLE
// A CUDA implementation of the alloc table, made by Hunter McCoy
// (hunter@cs.utah.edu) Copyright (C) 2023 by Hunter McCoy

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without l> imitation the
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
//  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
//  OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
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
#include <poggers/beta/block.cuh>
#include <poggers/beta/veb.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

#define BETA_MEM_TABLE_DEBUG 1

namespace beta {

namespace allocators {

// alloc table associates chunks of memory with trees
// using uint16_t as there shouldn't be that many trees.
// register atomically insert tree num, or registers memory from chunk_tree.

__global__ void betta_init_counters_kernel(unsigned int *malloc_counters,
                                           unsigned int *free_counters,
                                           block *blocks, uint64_t num_segments,
                                           uint64_t blocks_per_segment) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= num_segments) return;

  malloc_counters[tid] = 0;
  free_counters[tid] = 0;

  uint64_t base_offset = blocks_per_segment * tid;

  for (uint64_t i = 0; i < blocks_per_segment; i++) {
    block *my_block = &blocks[base_offset + i];

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
  block *blocks;

  // pair of counters for each segment to track use.
  unsigned int *malloc_counters;
  unsigned int *free_counters;

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

    printf("Booting memory table with %llu chunks\n", num_segments);

    uint16_t *ext_chunks;

    cudaMalloc((void **)&ext_chunks, sizeof(uint16_t) * num_segments);

    cudaMemset(ext_chunks, ~0U, sizeof(uint16_t) * num_segments);

    host_version->chunk_ids = ext_chunks;

    host_version->num_segments = num_segments;

    // init blocks

    uint64_t blocks_per_segment = bytes_per_segment / (min_size * 4096);

    block *ext_blocks;

    cudaMalloc((void **)&ext_blocks,
               sizeof(block) * blocks_per_segment * num_segments);

    cudaMemset(ext_blocks, 0U,
               sizeof(block) * (num_segments * blocks_per_segment));

    host_version->blocks = ext_blocks;

    host_version->blocks_per_segment = blocks_per_segment;

    host_version->memory = poggers::utils::get_device_version<char>(
        bytes_per_segment * num_segments);

    // generate counters and set them to 0.
    host_version->malloc_counters =
        poggers::utils::get_device_version<unsigned int>(num_segments);
    host_version->free_counters =
        poggers::utils::get_device_version<unsigned int>(num_segments);
    betta_init_counters_kernel<<<(num_segments - 1) / 512 + 1, 512> > >(
        host_version->malloc_counters, host_version->free_counters,
        host_version->blocks, num_segments, blocks_per_segment);

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
      printf("Chunk issue: %llu > %llu\n", segment, num_segments);
    }

    chunk_ids[segment] = id;
  }

  // register a segment from the table.
  __device__ void register_size(uint64_t segment, uint16_t size) {
    if (segment >= num_segments) {
      printf("Chunk issue\n");
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
    set_tree_id(segment, tree_id);

#if BETA_MEM_TABLE_DEBUG

    uint old_free_count =
        atomicExch((unsigned int *)&free_counters[segment], 0U);

    if (old_free_count != 0) {
      printf(
          "Memory free counter for segment %llu not properly reset: value "
          "is %u\n",
          segment, old_free_count);
    }

#endif

    atomicExch((unsigned int *)&malloc_counters[segment], 0U);

    // gate to init is init_new_universe
    return true;
  }

  // initialize the blocks in a segment
  __device__ bool setup_blocks(uint64_t segment, uint64_t num_blocks) {
    __threadfence();

    uint64_t base_offset = blocks_per_segment * segment;

    for (uint64_t i = 0; i < num_blocks; i++) {
      block *block = &blocks[base_offset + i];

      block->init();

      uint64_t global_offset = get_global_block_offset(block);

      // if (global_offset != base_offset+i){
      // 	printf("Issue in setting blocks\n");
      // }

      // if (global_offset*4096 != 4096*(base_offset+i)){
      // 	printf("Rounding issue\n");
      // }

      // block->attach_allocation(global_offset*4096);

      __threadfence();
    }

    return (atomicCAS(&malloc_counters[segment], 2 * blocks_per_segment, 0) ==
            2 * blocks_per_segment);
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
    return atomicCAS((unsigned short int *)&chunk_ids[segment],
                     (unsigned short int)~0U, (unsigned short int)~0U);
  }

  // return tree id to ~0
  __device__ bool reset_tree_id(uint64_t segment, uint16_t tree_id) {
    return (atomicCAS((unsigned short int *)&chunk_ids[segment],
                      (unsigned short int)tree_id,
                      (unsigned short int)~0U) == (unsigned short int)tree_id);
  }

  // atomic wrapper to get block
  __device__ uint get_block_from_segment(uint64_t segment) {
    return atomicAdd(&malloc_counters[segment], 1);
  }

  // atomic wrapper to free block.
  __device__ uint return_block_to_segment(uint64_t segment) {
    return atomicAdd(&free_counters[segment], 1);
  }

  // request a segment from a block
  // this verifies that the segment is initialized correctly
  // and returns nullptr on failure.
  __device__ block *get_block(uint64_t segment_id, uint16_t tree_id,
                              bool &empty) {
    uint my_count = get_block_from_segment(segment_id);

    uint16_t global_tree_id = read_tree_id(segment_id);

    uint64_t num_blocks = get_blocks_per_segment(global_tree_id);

    if (my_count >= num_blocks) {
      return nullptr;
    }

    // tree changed in interim.
    if (global_tree_id != tree_id) {
      return_block_to_segment(segment_id);
    }

    if (my_count == (num_blocks - 1)) {
      empty = true;
    }

    return &blocks[segment_id * blocks_per_segment + my_count];
  }

  // snap a block back to its segment
  // needed for returning
  __device__ uint64_t get_segment_from_block_ptr(block *block) {
    // this returns the stride in blocks
    uint64_t offset = (block - blocks);

    return offset / blocks_per_segment;
  }

  // get relative offset of a block in its segment.
  __device__ int get_relative_block_offset(block *block) {
    uint64_t offset = (block - blocks);

    return offset % blocks_per_segment;
  }

  // given a pointer, find the associated block for returns
  // not yet implemented
  __device__ block *get_block_from_ptr(void *ptr) {}

  // given a pointer, get the segment the pointer belongs to
  __device__ uint64_t get_segment_from_ptr(void *ptr) {
    uint64_t offset = ((char *)ptr) - memory;

    return offset / bytes_per_segment;
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
  __device__ uint64_t get_global_block_offset(block *block) {
    return block - blocks;
  }

  // get max blocks per segment when formatted to a given tree size.
  static __host__ __device__ uint64_t get_blocks_per_segment(uint16_t tree) {
    uint64_t tree_alloc_size = get_tree_alloc_size(tree);

    return bytes_per_segment / (tree_alloc_size * 4096);
  }

  // free block, returns true if this block was the last section needed.
  __device__ bool free_block(block *block_ptr, bool &need_to_reregister) {
    need_to_reregister = false;

    uint64_t segment = get_segment_from_block_ptr(block_ptr);

    uint old_count = return_block_to_segment(segment);

    uint16_t global_tree_id = read_tree_id(segment);

    uint64_t num_blocks = get_blocks_per_segment(global_tree_id);

    if (old_count == (num_blocks - 1)) {
      // can maybe free section.
      // attempt CAS
      // on success, you are the exclusive owner of the segment.

      uint leftover = atomicExch((unsigned int *)&free_counters[segment], 0U);

#if BETA_MEM_TABLE_DEBUG

      if (leftover != num_blocks) {
        printf("Weird leftover: %u != %lu\n", leftover, num_blocks);
      }

#endif

      return true;
    }

    return false;
  }
};

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard