#ifndef BETTA_ALLOCATOR
#define BETTA_ALLOCATOR
// Betta, the block-based extending-tree thread allocaotor, made by Hunter
// McCoy (hunter@cs.utah.edu) Copyright (C) 2023 by Hunter McCoy

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
#include <poggers/allocators/alloc_memory_table.cuh>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/allocators/block_storage.cuh>
#include <poggers/allocators/ext_veb_nosize.cuh>
#include <poggers/allocators/offset_slab.cuh>
#include <poggers/allocators/one_size_allocator.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

namespace poggers {

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

template <uint64_t bytes_per_segment, uint64_t smallest, uint64_t biggest>
struct betta_allocator {
  using my_type = betta_allocator<bytes_per_segment, smallest, biggest>;
  // using sub_tree_type = extending_veb_allocator_nosize<bytes_per_segment,
  // 5>;
  using sub_tree_type = veb_tree;

  using pinned_block_type = pinned_block_container<smallest, biggest>;

  veb_tree *segment_tree;
  // one_size_allocator * segment_tree;

  alloc_table<bytes_per_segment, smallest> *table;

  sub_tree_type **sub_trees;

  // smid_pinned_container<2> ** malloc_containers;

  pinned_block_type *local_blocks;

  pinned_storage **storage_containers;

  int num_trees;

  int smallest_bits;

  uint locks;

  static __host__ my_type *generate_on_device(uint64_t max_bytes,
                                              uint64_t seed) {
    my_type *host_version = get_host_version<my_type>();

    // plug in to get max chunks
    uint64_t max_chunks = get_max_chunks<bytes_per_segment>(max_bytes);

    host_version->segment_tree = veb_tree::generate_on_device(max_chunks, seed);

    // one_size_allocator::generate_on_device(max_chunks, bytes_per_segment,
    // seed);

    // estimate the max_bits
    uint64_t num_bits = bytes_per_segment / (4096 * smallest);

    host_version->local_blocks =
        pinned_block_type::generate_on_device(num_bits);

    uint64_t num_bytes = 0;

    // this underestimates the total #bytes needed.

    do {
      printf("Bits is %llu, bytes is %llu\n", num_bits, num_bytes);

      num_bytes += ((num_bits - 1) / 64 + 1) * 8;

      num_bits = num_bits / 64;
    } while (num_bits > 64);

    num_bytes += 8 + num_bits * sizeof(offset_alloc_bitarr);

    // printf("Final bits is %llu, bytes is %llu\n", num_bits, num_bytes);

    // need to verify, but this should be sufficient for all sizes.
    // host_version->bit_tree =
    // one_size_allocator::generate_on_device(max_chunks, num_bytes, seed);

    // printf("Each bit tree array gets %llu\n", num_bytes);

    uint64_t num_trees =
        get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest) + 1;

    host_version->smallest_bits = get_first_bit_bigger(smallest);

    host_version->num_trees = num_trees;

    sub_tree_type **ext_sub_trees =
        get_host_version<sub_tree_type *>(num_trees);

    for (int i = 0; i < num_trees; i++) {
      // debugging
      // sub_tree_type * temp_tree =
      // sub_tree_type::generate_on_device(get_p2_from_index(get_first_bit_bigger(smallest)+i),
      // seed+i, max_bytes);

      sub_tree_type *temp_tree =
          sub_tree_type::generate_on_device(max_chunks, i + seed);
      ext_sub_trees[i] = temp_tree;
    }

    // boot pinned storage

    pinned_storage **host_pinned_storage =
        poggers::utils::get_host_version<pinned_storage *>(num_trees);

    for (int i = 0; i < num_trees; i++) {
      host_pinned_storage[i] = pinned_storage::generate_on_device();
    }

    host_version->storage_containers =
        move_to_device<pinned_storage *>(host_pinned_storage, num_trees);

    host_version->sub_trees =
        move_to_device<sub_tree_type *>(ext_sub_trees, num_trees);

    boot_segment_trees<<<(max_chunks - 1) / 512 + 1, 512> > >(
        host_version->sub_trees, max_chunks, num_trees);

    host_version->locks = 0;

    // printf("Host sub tree %lx, dev %lx\n", (uint64_t) ext_sub_trees,
    // (uint64_t) host_version->sub_trees);

    host_version
        ->table = alloc_table<bytes_per_segment, smallest>::generate_on_device(
        max_bytes);  // host_version->segment_tree->get_allocator_memory_start());

    printf("Booted BETA with %llu trees\n", num_trees);

    return move_to_device(host_version);
  }

  // return the index of the largest bit set
  static __host__ __device__ int get_first_bit_bigger(uint64_t counter) {
    //	if (__builtin_popcountll(counter) == 1){

    // 0th bit would give 63

    // 63rd bit would give 0

    return poggers::utils::get_first_bit_bigger(counter);
  }

  static __host__ __device__ int get_num_trees() {
    return get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest) + 1;
  }

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

    // one_size_allocator::free_on_device(host_version->bit_tree);

    pinned_storage **host_pinned_storage =
        poggers::utils::move_to_host<pinned_storage *>(
            host_version->storage_containers, num_trees);

    for (int i = 0; i < num_trees; i++) {
      pinned_storage::free_on_device(host_pinned_storage[i]);
    }

    cudaFreeHost(host_pinned_storage);

    veb_tree::free_on_device(host_version->segment_tree);

    pinned_block_type::free_on_device(host_version->local_blocks);

    cudaFreeHost(host_subtrees);

    cudaFreeHost(host_version);
  }

  __device__ inline uint64_t snap_pointer_to_block(void *ext_ptr) {
    uint64_t snapped_offset = ((uint64_t)ext_ptr) / bytes_per_segment;

    return snapped_offset;
  }

  // to cast an offset, need a couple of things
  __device__ void *alloc_offset_to_ptr(uint64_t offset, uint16_t tree_id) {
    uint64_t block_id = offset / 4096;

    uint64_t relative_offset = offset % 4096;

    uint64_t segment = block_id / table->blocks_per_segment;

    uint64_t alloc_size = table->get_tree_alloc_size(tree_id);

    char *memory_start = table->get_segment_memory_start(segment);

    // printf("Alloc from block %llu with offset %llu\n", block_id,
    // relative_offset);

    // to start, lets get to the start of the block

    return (void *)(memory_start + relative_offset * alloc_size);
  }

  // for small allocations, attempt to grab locks.
  __device__ uint64_t malloc(uint64_t bytes_needed) {
    int tree_id = get_first_bit_bigger(smallest) - smallest_bits;

    if (tree_id >= num_trees) {
      // get big allocation
      printf("Larger allocations not yet implemented\n");

      return 0;
    } else {
      per_size_pinned_blocks *my_local_blocks =
          local_blocks->get_tree_local_blocks(tree_id);

      auto *my_pinned_storage =
          storage_containers[tree_id]->get_pinned_storage();

      while (true) {
        // printf("%llu Spinning...\n", threadIdx.x+blockIdx.x*blockDim.x);

        auto my_block = my_local_blocks->get_my_block();

        if (my_block == nullptr) {
          if (my_local_blocks->lock_my_block()) {
            auto new_block = request_new_block_from_tree(tree_id);

            // failure to get a block is failure.
            if (new_block == nullptr) {
              my_local_blocks->unlock_my_block();
              return 0;
            }

            if (!my_local_blocks->swap_out_nullptr(new_block)) {
              // this isn't bad - just means read was stale.
              // printf("Failed to swap out %llu\n",
              // table->get_global_block_offset(new_block));
              free_block(new_block);
            }

            my_local_blocks->unlock_my_block();

            __threadfence();
            continue;
          } else {
            // acquire different block

            // for the moment, this loops.

            my_block = my_local_blocks->get_alt_block();

            //__threadfence();
            // continue;
          }
        }

        if (my_block == nullptr) {
          continue;
        }

        // at this point we have a block! good to go.

        uint64_t allocation;

        bool alloced =
            alloc_with_locks(allocation, my_block, my_pinned_storage);

        if (!alloced) {
          if (my_local_blocks->lock_my_block()) {
            auto new_block = request_new_block_from_tree(tree_id);
            if (new_block == nullptr) {
              my_local_blocks->unlock_my_block();
              return 0;
            }

            if (!my_local_blocks->replace_block(my_block, new_block)) {
              // this is not an err.
              // printf("Someone else already set... %llu\n",
              // table->get_global_block_offset(new_block));

              free_block(new_block);
              __threadfence();

              new_block = nullptr;
            }

            my_local_blocks->unlock_my_block();
          }

          __threadfence();
          continue;
        }

        uint64_t global_offset = table->get_global_block_offset(my_block);

        if (my_block->get_offset() / 4096 != global_offset) {
          printf("Read err in block %llu: %llu != %llu\n", global_offset,
                 global_offset, my_block->get_offset() / 4096);
        }

        if (!alloced) printf("Looping incorrectly\n");

        if (alloced) return allocation;

        // return allocation; //global_offset*4096 + (allocation  % 4096);
        // //alloc_offset_to_ptr(allocation, tree_id);
      }
    }

    return 0;
  }

  // get a new segment for a tree!
  __device__ bool gather_new_segment(uint16_t tree) {
    // void * bits = bit_tree->malloc();

    // if (bits == nullptr){
    // 	return false;
    // }

    // void * memory_segment = segment_tree->malloc();

    // if (memory_segment == nullptr){
    // 	bit_tree->free(bits);

    // 	return false;
    // }

    // todo
    // swap to first
    __threadfence();

    uint64_t id = segment_tree->malloc_first();

    if (id == veb_tree::fail()) {
      // printf("No segments available\n");
      return false;
    }

    // otherwise, both initialized
    // register segment
    if (!table->setup_segment(id, tree)) {
      printf("Failed to acquire updatable segment\n");

      segment_tree->insert_force_update(id);
      // abort, but not because no segments are available.
      // this is fine.
      return true;
    }

    __threadfence();

    sub_trees[tree]->insert_force_update(id);

    // printf("Attached %llu to %d\n", id, tree);

    return true;
  }

  __device__ bool acquire_tree_lock(uint16_t tree) {
    return ((atomicOr(&locks, SET_BIT_MASK(tree)) & SET_BIT_MASK(tree)) == 0);
  }

  __device__ bool release_tree_lock(uint16_t tree) {
    atomicAnd(&locks, ~SET_BIT_MASK(tree));
  }

  // gather a new block for a tree.
  // this attempts to pull from a memory segment.
  //  and will atteach a new segment if now
  __device__ offset_alloc_bitarr *request_new_block_from_tree(uint16_t tree) {
    int attempts = 0;

    while (attempts < REQUEST_BLOCK_MAX_ATTEMPTS) {
      __threadfence();

      uint64_t segment = sub_trees[tree]->find_first_valid_index();

      if (segment == veb_tree::fail()) {
        if (true) {
          // if (acquire_tree_lock(tree)){

          bool success = gather_new_segment(tree);

          // release_tree_lock(tree);

          __threadfence();

          // failure to acquire a tree segment means we are full.
          if (!success) {
            // timeouts should be rare...
            // if this failed its more probable that someone else added
            // a segment!
            __threadfence();
            attempts++;
          }
        } else {
          // this is inf loop?
          // printf("Looping on tree lock\n");
        }

        __threadfence();

        // for the moment, failures due to not being full enough aren't
        // penalized.
        continue;
      }

      bool last_block = true;

      auto block = table->get_block(segment, tree, last_block);

      if (last_block) {
        sub_trees[tree]->remove(segment);

        // gather_new_segment(tree);
      }

      if (block != nullptr) {
        return block;
      }

      // test: only threads that try and fail to acquire a segment can drop
      // attempts++;
    }

    return nullptr;
  }

  __device__ void free_block(offset_alloc_bitarr *block) {
    bool need_to_reregister = false;
    bool need_to_deregister = table->free_block(block, need_to_reregister);

    uint64_t segment = table->get_segment_from_block_ptr(block);

    // uint16_t tree = table->read_tree_id(segment);

    if (need_to_deregister) {
      // printf("Deregistering block! %llu\n",
      // table->get_segment_from_block_ptr(block));

      // don't need to reset anything, just pull from table and threadfence

      uint16_t tree = table->read_tree_id(segment);

      // pull from tree
      // should be fine, no one can update till this point
      sub_trees[tree]->remove(segment);

      table->reset_tree_id(segment, tree);
      __threadfence();

      segment_tree->insert_force_update(segment);
    }
  }

  __device__ void free_ptr(void *ptr) { return; }

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

}  // namespace poggers

#endif  // End of VEB guard