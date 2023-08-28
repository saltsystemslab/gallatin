#ifndef EXT_VEB_TREE
#define EXT_VEB_TREE
// A CUDA implementation of the Extending Van Emde Boas tree, made by Hunter
// McCoy (hunter@cs.utah.edu) Copyright (C) 2023 by Hunter McCoy

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

// The Extending Van Emde Boas Tree, or EVEB, is a data structure that supports
// efficient data grouping and allocation/deallocation based on use. given a
// target size and a memory chunk size, the tree dynamically pulls/pushes chunks
// to the free list based on usage. the metadata supports up to the maximum size
// passed in, and persists so that the true structure does not mutate over the
// runtime.

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/allocators/offset_slab.cuh>
#include <poggers/allocators/sub_veb.cuh>
#include <poggers/allocators/veb.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

#define EXT_VEB_RESTART_CUTOFF 30

#define EXT_VEB_GLOBAL_LOAD 1
#define EXT_VEB_MAX_ATTEMPTS 15

namespace poggers {

namespace allocators {

#define FILL_NEW .8

// contains array of sub_veb
// each of which has space for arrays, along with pointer to blocks?
// dishing out array memory can use that as well...

// planning
// each tree will use bytes bytes for the main allocation.

// universe_size = bytes_per_chunk/alloc_size...

template <typename tree_type>
__global__ void init_subtree_kernel(tree_type *tree) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= tree->max_segments) return;

  tree->counters[tid] = 0;

  tree->memory_segments[tid] = nullptr;

  // no trees start valid.
  tree->active_trees->remove(tid);

  // tree->dead_trees->remove(tid);

  sub_veb_tree *my_tree = tree->get_sub_tree(tid);

  // sub_tree_type

  sub_veb_tree::init((void *)my_tree, tree->allocations_per_segment, tid);
}

// template <typename tree_type, typename bit_allocator, typename
// memory_allocator>
// __global__ void dev_version_cleanup_kernel(tree_type * tree, bit_allocator *
// balloc, memory_allocator * mem_alloc){

// 	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

// 	if (tid >= tree->max_chunks) return;

// 	if (tree->memory_segments[tid] != nullptr){

// 		mem_alloc->free(tree->memory_segments[tid]);

// 		tree->counters[tid];

// 		balloc->free(tree->trees[tid].get_mem());

// 		tree->trees[tid].reset_mem();

// 	}

// }

// removing the size has a couple of benefits.
// all trees can no go in an array - simplifies system.
// calculation not necessary for most types, especially as blocks do not
// understand sizing.

// need to add block chunks.

// internal logic -
// two trees - one to register what slots are available
// and another to show what trees are live.
template <uint64_t bytes_per_chunk, int extra_blocks>
struct extending_veb_allocator_nosize {
  using my_type = extending_veb_allocator_nosize<bytes_per_chunk, extra_blocks>;

  uint64_t max_segments;
  uint64_t alloc_size;

  uint64_t allocations_per_segment;

  uint64_t size_of_sub_tree;

  veb_tree *active_trees;

  veb_tree *dead_trees;

  int *counters;

  char *tree_memory;

  void **memory_segments;

  smid_pinned_container<extra_blocks> *block_containers;

  pinned_storage *storage_containers;

  // boot the tree.
  static __host__ my_type *generate_on_device(uint64_t ext_alloc_size,
                                              uint64_t ext_seed) {
    assert(bytes_per_chunk % alloc_size == 0);

    uint64_t num_allocs_per_segment = bytes_per_chunk / (4096 * ext_alloc_size);

    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    uint64_t max_chunks = poggers::utils::get_max_chunks<bytes_per_chunk>();

    host_version->max_segments = max_chunks;
    host_version->alloc_size = ext_alloc_size;

    uint64_t size_of_sub_tree =
        sub_veb_tree::get_size_bytes_noarray(num_allocs_per_segment);

    char *dev_trees;

    cudaMalloc((void **)&dev_trees, max_chunks * size_of_sub_tree);

    int *dev_counters;
    cudaMalloc((void **)&dev_counters, sizeof(int) * max_chunks);

    void **ext_memory_segments;

    cudaMalloc((void **)&ext_memory_segments, sizeof(void *) * max_chunks);

    host_version->active_trees =
        veb_tree::generate_on_device(max_chunks, ext_seed);
    host_version->dead_trees =
        veb_tree::generate_on_device(max_chunks, ext_seed);

    host_version->size_of_sub_tree = size_of_sub_tree;
    host_version->tree_memory = dev_trees;
    host_version->counters = dev_counters;
    host_version->memory_segments = ext_memory_segments;

    host_version->block_containers =
        smid_pinned_container<extra_blocks>::generate_on_device(
            host_version->block_allocator, 4096);

    host_version->storage_containers = pinned_storage::generate_on_device();

    host_version->allocations_per_segment = num_allocs_per_segment;

    printf(
        "Tree with %llu bytes per chunk and %llu alloc size has %llu total "
        "chunks and %llu num_allocs_per_segment\n",
        bytes_per_chunk, ext_alloc_size, max_chunks, num_allocs_per_segment);

    my_type *dev_version;

    cudaMalloc((void **)&dev_version, sizeof(my_type));

    cudaMemcpy(dev_version, host_version, sizeof(my_type),
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    init_subtree_kernel<my_type>
        <<<(max_chunks - 1) / 512 + 1, 512>>>(dev_version);

    cudaFreeHost(host_version);

    cudaDeviceSynchronize();

    return dev_version;
  }

  static __host__ my_type *generate_on_device(uint64_t ext_alloc_size,
                                              uint64_t ext_seed,
                                              uint64_t max_bytes) {
    assert(bytes_per_chunk % alloc_size == 0);

    uint64_t num_allocs_per_segment = bytes_per_chunk / (4096 * ext_alloc_size);

    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    uint64_t max_chunks =
        poggers::utils::get_max_chunks<bytes_per_chunk>(max_bytes);

    host_version->max_segments = max_chunks;
    host_version->alloc_size = ext_alloc_size;

    uint64_t size_of_sub_tree =
        sub_veb_tree::get_size_bytes_noarray(num_allocs_per_segment);

    char *dev_trees;

    cudaMalloc((void **)&dev_trees, max_chunks * size_of_sub_tree);

    int *dev_counters;
    cudaMalloc((void **)&dev_counters, sizeof(int) * max_chunks);

    void **ext_memory_segments;

    cudaMalloc((void **)&ext_memory_segments, sizeof(void *) * max_chunks);

    host_version->active_trees =
        veb_tree::generate_on_device(max_chunks, ext_seed);
    host_version->dead_trees =
        veb_tree::generate_on_device(max_chunks, ext_seed);

    host_version->tree_memory = dev_trees;
    host_version->counters = dev_counters;
    host_version->memory_segments = ext_memory_segments;
    host_version->size_of_sub_tree = size_of_sub_tree;

    host_version->block_containers =
        smid_pinned_container<extra_blocks>::generate_on_device();

    host_version->storage_containers = pinned_storage::generate_on_device();

    host_version->allocations_per_segment = num_allocs_per_segment;

    printf(
        "Tree with %llu bytes per chunk and %llu alloc size has %llu total "
        "chunks and %llu num_allocs_per_segment\n",
        bytes_per_chunk, ext_alloc_size, max_chunks, num_allocs_per_segment);

    my_type *dev_version;

    cudaMalloc((void **)&dev_version, sizeof(my_type));

    cudaMemcpy(dev_version, host_version, sizeof(my_type),
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    init_subtree_kernel<my_type>
        <<<(max_chunks - 1) / 512 + 1, 512>>>(dev_version);

    cudaFreeHost(host_version);

    cudaDeviceSynchronize();

    return dev_version;
  }

  // free up all memory used - included blocks handed to you by the allocators
  // template <typename bit_allocator, typename memory_segment_allocator>
  static __host__ void free_on_device(my_type *dev_version) {
    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    cudaMemcpy(host_version, dev_version, sizeof(my_type),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // dev_version_cleanup_kernel<my_type, bit_allocator,
    // memory_segment_allocator><<<(host_version->max_segments-1)/512+1,
    // 512>>>(dev_version, balloc, memalloc);

    cudaDeviceSynchronize();

    veb_tree::free_on_device(host_version->active_trees);

    veb_tree::free_on_device(host_version->dead_trees);

    smid_pinned_container<extra_blocks>::free_on_device(
        host_version->block_containers);

    pinned_storage::free_on_device(host_version->storage_containers);

    cudaFree(host_version->tree_memory);

    cudaFree(host_version->counters);

    cudaFree(host_version->memory_segments);

    cudaFree(dev_version);
  }

  __device__ offset_alloc_bitarr *malloc_block(bool &need_more_allocations) {
    // pick a first active tree.
    // increment counter

    // if counter is in range (0, num_allocs-1), we are valid! pull from the
    // tree

    uint64_t first_block_offset = active_trees->find_first_valid_index();

    if (first_block_offset == veb_tree::fail()) {
      // printf("ext_veb_nosize failed to query\n");

      // have to deal with this at some point

      return nullptr;
    }

    // increment counter;

    int count = atomicAdd(&counters[first_block_offset], 1);

    if (count < allocations_per_segment) {
      uint64_t local_offset = get_sub_tree(first_block_offset)->malloc();

      if (local_offset == sub_veb_tree::fail()) {
        // this may become disabled
        // if it results in massive amounts of thread allocation for no reason.
        // we are trying to amortize away the cost, not allocate 100% memory
        // quickly.
        need_more_allocations = true;
        return nullptr;
      }

      // I own local offset

      // uint64_t global_offset =
      // local_offset*allocations_per_segment+first_block_offset;

      if (count == allocations_per_segment * .8) {
        need_more_allocations = true;
      }

      auto block =
          get_sub_tree(first_block_offset)->get_block_from_offset(local_offset);

      block->init();

      // this is the global tree offset
      uint64_t global_offset =
          (first_block_offset * allocations_per_segment + local_offset) * 4096;
      block->attach_allocation(global_offset);

      return block;

    } else {
      // mark tree full
      active_trees->remove(first_block_offset);
    }

    return nullptr;
  }

  __device__ void free_block(void *block) {
    printf("Not implemented\n");

    return;
  }

  // translate global uint64_t to tree relative, then to void *
  __device__ void *offset_to_ptr(uint64_t offset) {
    uint64_t tree_id = offset / allocations_per_segment;

    uint64_t local_offset = offset % allocations_per_segment;

    return (void *)((uint64_t)memory_segments[tree_id] +
                    alloc_size * local_offset);
  }

  __device__ uint64_t malloc_offset(bool &need_more_allocations);

  __device__ void *malloc(bool &need_more_allocations) {
    // get active tree

    smid_pinned_container<extra_blocks> *my_storage =
        block_containers->get_pinned_storage();

    offset_storage_bitmap *my_storage_bitmap =
        storage_containers->get_pinned_storage();

    // int num_attempts = 0;

    //    		while (num_attempts < EXT_VEB_MAX_ATTEMPTS){

    // 	//auto team = cg::coalesced_threads();

    // 	//printf("Stalling in the main loop\n");

    // 	offset_alloc_bitarr * bitarr = my_storage->get_primary();

    // 	if (bitarr == nullptr){
    // 		//team.sync();

    // 		//printf("Bitarr empty\n");
    // 		num_attempts+=1;
    // 		continue;
    // 	}

    // 	uint64_t allocation;

    // 	bool alloced = alloc_with_locks(allocation, bitarr, my_storage_bitmap);

    // 	if (!alloced){

    // 		int result = my_storage->pivot_primary(bitarr);

    // 		if (result != -1){

    // 			//malloc and replace pivot slab

    // 			#if SLAB_DEBUG_CHECKS
    // 			if (!bitarr->atomic_check_unpinned()){
    // 				printf("Unpinning bug\n");
    // 			}
    // 			#endif

    // 			{
    // 				// uint64_t slab_offset =
    // block_allocator->get_offset();

    // 				// if (slab_offset ==
    // one_size_allocator::fail()){

    // 				// 	return nullptr;

    // 				// }

    // 				offset_alloc_bitarr * slab = (offset_alloc_bitarr *)
    // malloc_block(need_more_allocations);

    // 				slab->init();

    // 				//this seems to be the bug?
    // 				uint64_t slab_buffer_offset = slab_offset*4096;

    // 				slab->attach_allocation(slab_buffer_offset);

    // 				slab->mark_pinned();

    // 				__threadfence();

    // 				//printf("Attaching buffa\n");

    // 				if (!my_storage->attach_new_buffer(result,
    // slab)){ 					#if SLAB_DEBUG_CHECKS 					printf("Bug attaching buffer\n"); 					#endif
    // 				}

    // 			}

    // 		}

    // 	} else {

    // 		return (void *) (extra_memory + allocation*offset_size);

    // 	}

    // 	num_attempts+=1;

    // }

    return nullptr;
  }

  __device__ sub_veb_tree *get_sub_tree(uint64_t tree_id) {
    return (sub_veb_tree *)(tree_memory + size_of_sub_tree * tree_id);
  }

  // to register a block, use the veb tree to find a new region that is valid.
  // this block has already been registered in the main tree.
  __device__ void register_new_segment(void *new_bits, void *ext_memory) {
    //__device__ void register_new_block(void * new_bits, void *
    //ext_allocation){

    // pick a random dead tree and boot it.
    uint64_t my_offset = dead_trees->malloc();

    sub_veb_tree *sub_tree = get_sub_tree(my_offset);

    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
      printf("Each subtree needs %llu bits\n", sub_tree->get_size_arrays());
    }

    sub_tree->init_arrays(
        new_bits);  //, bytes_per_chunk/alloc_size, my_offset);

    memory_segments[my_offset] = ext_memory;

    counters[my_offset] = 0;

    // tree is ready to go! add it to the system

    active_trees->insert(my_offset);
  }

  template <typename bit_allocator, typename memory_allocator>
  __device__ void deregister_segment(uint64_t offset, bit_allocator *balloc,
                                     memory_allocator *mem_alloc) {
    active_trees->remove(offset);

    __threadfence();

    // this time

    // if this passes, we can deregister
    if (atomicCAS(&counters[offset], 0, bytes_per_chunk) == 0) {
      // safe to register!
      balloc->free(get_sub_tree(offset)->free_memory());

      mem_alloc->free(memory_segments[offset]);

      return;

    } else {
      // the CAS failed, so reinsert the tree.
      active_trees->insert(offset);
    }
  }

  __device__ bool free(uint64_t relative_offset) {
    // Free process

    // get tree offset and internal offset
    // free internal offset.

    // if counter is now 0 - free
    // else if counter < threshold, reactivate, tree

    // free memory, then decrement counter.

    uint64_t tree_id = relative_offset / (allocations_per_segment);

    uint64_t tree_offset = relative_offset % (allocations_per_segment);

    get_sub_tree(tree_id)->insert(tree_offset);
    // trees[tree_id].insert(tree_offset);

    if (atomicSub(&counters[tree_id], 1) == 1) {
      // tree may be empty
      return true;
    }

    return false;
  }
};

}  // namespace allocators

}  // namespace poggers

#endif  // End of VEB guard