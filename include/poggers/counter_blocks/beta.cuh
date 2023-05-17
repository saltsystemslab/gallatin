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

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/beta/allocator_context.cuh>
#include <poggers/counter blocks/one_size_allocator.cuh>
#include <poggers/counter_blocks/block.cuh>
#include <poggers/counter_blocks/memory_table.cuh>
#include <poggers/counter_blocks/shared_block_storage.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

namespace beta {

namespace allocators {

#define REQUEST_BLOCK_MAX_ATTEMPTS 1

// alloc table associates chunks of memory with trees

// using uint16_t as there shouldn't be that many trees.

// register atomically inserst tree num, or registers memory from segment_tree.

using namespace poggers::utils;

__global__ void boot_segment_trees(veb_tree **segment_trees,
                                   uint64_t max_chunks, int num_trees) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= max_chunks) return;

  for (int i = 0; i < num_trees; i++) {
    segment_trees[i]->remove(tid);
  }
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
  using thread_storage_type = pinned_thread_storage;

  // internal structures
  veb_tree *segment_tree;

  alloc_table<bytes_per_segment, smallest> *table;

  sub_tree_type **sub_trees;

  pinned_block_type *local_blocks;

  thread_storage_type **storage_containers;

  int num_trees;

  int smallest_bits;

  uint locks;

  // generate the allocator on device.
  // this takes in the number of bytes owned by the allocator (does not include
  // the space of the allocator itself.)
  static __host__ my_type *generate_on_device(uint64_t max_bytes,
                                              uint64_t seed) {
    my_type *host_version = get_host_version<my_type>();

    // plug in to get max chunks
    uint64_t max_chunks = get_max_chunks<bytes_per_segment>(max_bytes);

    host_version->segment_tree = veb_tree::generate_on_device(max_chunks, seed);

    // estimate the max_bits
    uint64_t num_bits = bytes_per_segment / (4096 * smallest);

    host_version->local_blocks =
        pinned_block_type::generate_on_device(num_bits);

    uint64_t num_bytes = 0;

    do {
      printf("Bits is %llu, bytes is %llu\n", num_bits, num_bytes);

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

    // boot pinned storage

    thread_storage_type **host_pinned_storage =
        poggers::utils::get_host_version<thread_storage_type *>(num_trees);

    for (int i = 0; i < num_trees; i++) {
      host_pinned_storage[i] = thread_storage_type::generate_on_device();
    }

    host_version->storage_containers =
        move_to_device<thread_storage_type *>(host_pinned_storage, num_trees);
    host_version->sub_trees =
        move_to_device<sub_tree_type *>(ext_sub_trees, num_trees);

    boot_segment_trees<<<(max_chunks - 1) / 512 + 1, 512>>>(
        host_version->sub_trees, max_chunks, num_trees);

    host_version->locks = 0;

    host_version
        ->table = alloc_table<bytes_per_segment, smallest>::generate_on_device(
        max_bytes);  // host_version->segment_tree->get_allocator_memory_start());

    printf("Booted BETA with %llu trees\n", num_trees);

    return move_to_device(host_version);
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

    thread_storage_type **host_pinned_storage =
        poggers::utils::move_to_host<thread_storage_type *>(
            host_version->storage_containers, num_trees);

    for (int i = 0; i < num_trees; i++) {
      thread_storage_type::free_on_device(host_pinned_storage[i]);
    }

    cudaFreeHost(host_pinned_storage);

    veb_tree::free_on_device(host_version->segment_tree);

    pinned_block_type::free_on_device(host_version->local_blocks);

    cudaFreeHost(host_subtrees);

    cudaFreeHost(host_version);
  }

  // generate context
  //  this boots the warp lock for the thread allocators.
  __device__ context *generate_kernel_context(bool load_context) {
    __shared__ context local_context;

    if (load_context) {
      local_context.init_context_lock_only();
    }

    return &local_context;
  }

  // create the context for the first time
  //  this must be called at the start of the kernel
  __device__ context *create_local_context() {
    return generate_kernel_context(true);
  }

  // regenerate the context
  //  this is called in malloc to generate the warp lock.
  __device__ context *reload_kernel_context() {
    return generate_kernel_context(false);
  }

  // given a pointer, return the segment it belongs to
  __device__ inline uint64_t snap_pointer_to_block(void *ext_ptr) {
    char *memory_start = table->get_segment_memory_start(segment);

    uint64_t snapped_offset =
        ((uint64_t)(ext_ptr - memory_start)) / bytes_per_segment;

    return snapped_offset;
  }

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

  // malloc an individual allocation
  // returns an offset that can be cast into the associated void *
  __device__ uint64_t malloc(uint64_t bytes_needed) {
    int tree_id = get_first_bit_bigger(smallest) - smallest_bits;

    if (tree_id >= num_trees) {
      // get big allocation
      // this is currently unfinished, is a todo after ouroboros
      printf("Larger allocations not yet implemented\n");

      return 0;

    } else {
      // get local block storage and thread storage
      per_size_pinned_blocks *my_local_blocks =
          local_blocks->get_tree_local_blocks(tree_id);
      auto *my_local_thread_storage =
          storage_containers[tree_id]->get_thread_storage();

      // and regenerate context
      auto context = reload_kernel_context();

      // global attempt loop
      // this cycles until we either receive an allocation or fail to request a
      // new block
      while (true) {
        // select block to pull from and get global stats
        Block my_block = my_local_blocks->get_my_block();
        uint64_t global_offset = table->get_global_block_offset(my_block);

        if (my_block == nullptr) {
          // if block is nullptr, attempt to reset with new block
          if (my_local_blocks->lock_my_block()) {
            Block new_block = request_new_block_from_tree(tree_id);

            // failure to get a block is failure.
            if (new_block == nullptr) {
              my_local_blocks->unlock_my_block();
              return ~0ULL;
            }

            if (!my_local_blocks->swap_out_nullptr(new_block)) {
              // this isn't bad - just means read was stale.
              free_block(new_block);
            }

            my_local_blocks->unlock_my_block();

            __threadfence();
            continue;

          } else {
            // did not acquire right to get new block, retry
            continue;
          }
        }

        // at this point we have a block! good to go.
        // this subroutine returns allocation or ~0ULL on failure.
        uint64_t allocation =
            alloc_with_locks(context->get_local_lock(), global_offset, my_block,
                             my_local_thread_storage);

        // bool alloced = alloc_with_locks(allocation, my_block,
        // my_local_thread_storage);

        if (allocation == ~0ULL) {
          // on allocation failure, replace block
          if (my_local_blocks->lock_my_block()) {
            Block new_block = request_new_block_from_tree(tree_id);
            if (new_block == nullptr) {
              my_local_blocks->unlock_my_block();
              return ~0ULL;
            }

            if (!my_local_blocks->replace_block(my_block, new_block)) {
              // this is not an err, block was already set
              free_block(new_block);
              __threadfence();

              new_block = nullptr;
            }

            my_local_blocks->unlock_my_block();
          }

          __threadfence();
          continue;
        }

        // allocation is done, return.
        return allocation;
      }
    }

    return 0;
  }

  // get a new segment for a given tree!
  __device__ bool gather_new_segment(uint16_t tree) {
    // request new segment
    uint64_t id = segment_tree->malloc_first();

    if (id == veb_tree::fail()) {
      // no segment available - this signals allocator full, return nullptr.
      return false;
    }

    // otherwise, both initialized
    // register segment
    if (!table->setup_segment(id, tree)) {
      // printf("Failed to acquire updatable segment\n");

      segment_tree->insert_force_update(id);
      // abort, but not because no segments are available.
      // this is fine.
      return true;
    }

    __threadfence();

    // insertion with forced flush
    sub_trees[tree]->insert_force_update(id);

    return true;
  }

  // lock given tree to prevent oversubscription
  __device__ bool acquire_tree_lock(uint16_t tree) {
    return ((atomicOr(&locks, SET_BIT_MASK(tree)) & SET_BIT_MASK(tree)) == 0);
  }

  __device__ bool release_tree_lock(uint16_t tree) {
    atomicAnd(&locks, ~SET_BIT_MASK(tree));
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

          release_tree_lock(tree);

          __threadfence();

          // failure to acquire a tree segment means we are full.
          if (!success) {
            // timeouts should be rare...
            // if this failed its more probable that someone else added a
            // segment!
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

      // valid segment, get new block.
      Block new_block = table->get_block(segment, tree, last_block);

      if (last_block) {
        // if the segment is fully allocated, it can be detached
        // and returned to the segment tree when empty
        sub_trees[tree]->remove(segment);
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
    bool need_to_reregister = false;
    bool need_to_deregister =
        table->free_block(block_to_free, need_to_reregister);

    uint64_t segment = table->get_segment_from_block_ptr(block_to_free);

    if (need_to_deregister) {
      // returning segment
      // don't need to reset anything, just pull from table and threadfence
      uint16_t tree = table->read_tree_id(segment);

      // pull from tree
      // should be fine, no one can update till this point
      sub_trees[tree]->remove(segment);

      table->reset_tree_id(segment, tree);
      __threadfence();

      // insert with threadfence
      segment_tree->insert_force_update(segment);
    }
  }

  // return a pointer
  // not finished.
  __device__ void free_ptr(void *ptr) {
    // get block

    // free block

    return;
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

      printf("Tree %d owns %llu/%llu\n", i, sub_segments, sub_max);
    }

    cudaFreeHost(host_trees);

    cudaFreeHost(host_version);
  }

  static __host__ __device__ uint64_t get_blocks_per_segment(uint16_t tree) {
    return alloc_table<bytes_per_segment, smallest>::get_blocks_per_segment(
        tree);
  }
};

}  // namespace allocators

}  // namespace beta

#endif  // End of VEB guard