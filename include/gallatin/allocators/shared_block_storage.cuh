#ifndef GALLATIN_SHARED_BLOCK_STORAGE
#define GALLATIN_SHARED_BLOCK_STORAGE

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/allocators/block.cuh>
#include <gallatin/allocators/murmurhash.cuh>
#include <vector>

#include "assert.h"
#include "stdio.h"

// These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>



namespace cg = cooperative_groups;

// a pointer list managing a set section of device memory
namespace gallatin {

namespace allocators {

#define SLAB_PRINT_DEBUG 0

#define SHARED_BLOCK_COUNTER_CUTOFF 30

// should these start initialized? I can try it.
static __global__ void gallatin_set_block_bitarrs(Block **blocks, uint64_t num_blocks) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= num_blocks) return;

  blocks[tid] = nullptr;
}

// per size pinned blocks have one block per size (wow).
// in your lifetime.
// read your block.
// if
struct per_size_pinned_blocks {
  uint64_t num_blocks;

  Block **blocks;

  static __host__ per_size_pinned_blocks *generate_on_device(
      uint64_t num_blocks) {
    if (num_blocks == 0) num_blocks = 1;

    per_size_pinned_blocks *host_version =
        gallatin::utils::get_host_version<per_size_pinned_blocks>();

    host_version->blocks =
        gallatin::utils::get_device_version<Block *>(num_blocks);

    host_version->num_blocks = num_blocks;

    gallatin_set_block_bitarrs<<<(num_blocks - 1) / 512 + 1, 512>>>(
        host_version->blocks, num_blocks);

    return gallatin::utils::move_to_device<per_size_pinned_blocks>(host_version);
  }


  static __host__ per_size_pinned_blocks * generate_on_device_nowait(
      uint64_t num_blocks) {
    if (num_blocks == 0) num_blocks = 1;

    per_size_pinned_blocks *host_version =
        gallatin::utils::get_host_version<per_size_pinned_blocks>();

    host_version->blocks =
        gallatin::utils::get_device_version<Block *>(num_blocks);

    host_version->num_blocks = num_blocks;

    gallatin_set_block_bitarrs<<<(num_blocks - 1) / 512 + 1, 512>>>(
        host_version->blocks, num_blocks);

    return gallatin::utils::move_to_device_nowait<per_size_pinned_blocks>(host_version);
  }

  static __host__ void free_on_device(per_size_pinned_blocks *dev_version) {
    per_size_pinned_blocks *host_version =
        gallatin::utils::move_to_host<per_size_pinned_blocks>(dev_version);

    cudaFree(host_version->blocks);

    cudaFreeHost(host_version);
  }

  // Probe the SM-indexed block table for an entry that's been published
  // (non-null). Single acquire-load per slot: the previous split into
  // get_valid_block_index + get_my_block did two redundant acquire-loads of
  // the same address on every malloc. Acquire ordering pairs with the
  // release-CAS in swap_out_nullptr — any non-null Block* we see carries the
  // publisher's init_malloc/free_counter writes with it.
  //
  // NOTE: an experimental per-warp slot key (smid ^ warp ^ blockIdx) was
  // measured at a 40% regression at the current `MIN_PINNED_CUTOFF=32` on
  // Blackwell — spreading 8 same-SM warps to 8 different `blocks[slot]`
  // cachelines bouncing across SMs cost more than the atomic-contention win.
  // Per-warp keying needs a much larger pool to pay off; revisit when the
  // boot-cost / per-tree cutoff design lands.
  __device__ Block *get_valid_block(int &out_smid) {
    int my_smid = gallatin::utils::get_smid() % num_blocks;
    int original_smid = my_smid;
    int counter = 0;

    Block *block = gallatin::utils::load_acquire(&blocks[my_smid]);
    while (block == nullptr && my_smid != (original_smid - 1)) {
      my_smid = (my_smid + 1) % num_blocks;
      counter += 1;
      if (counter >= SHARED_BLOCK_COUNTER_CUTOFF) break;
      block = gallatin::utils::load_acquire(&blocks[my_smid]);
    }

    out_smid = my_smid;
    return block;
  }

  // Detach: CAS(block_to_swap -> nullptr). Release ordering pairs with
  // acquire-load above so any consumer that subsequently sees nullptr is also
  // guaranteed to see prior writes from the detaching thread.
  __device__ bool swap_out_block(int my_smid, Block *block_to_swap) {
    Block *expected = block_to_swap;
    return gallatin::utils::cas_release<Block *>(&blocks[my_smid], expected,
                                                 nullptr);
  }

  // Publish: CAS(nullptr -> block_to_swap). Release ordering ensures the
  // block's prior init_malloc write is visible to any acquire-load of the
  // slot.
  __device__ bool swap_out_nullptr(int my_smid, Block *block_to_swap) {
    Block *expected = nullptr;
    return gallatin::utils::cas_release<Block *>(&blocks[my_smid], expected,
                                                 block_to_swap);
  }

  __device__ uint64_t calculate_overhead(){
    return num_blocks * sizeof(Block *);
  }

};

// container has one of these per size.
template <uint64_t smallest, uint64_t biggest>
struct pinned_shared_blocks {
  using my_type = pinned_shared_blocks<smallest, biggest>;

  per_size_pinned_blocks **block_containers;

  static __host__ my_type *generate_on_device(uint64_t blocks_per_segment, uint16_t min) {
    my_type *host_version = gallatin::utils::get_host_version<my_type>();

    uint64_t num_trees = gallatin::utils::get_first_bit_bigger(biggest) -
                         gallatin::utils::get_first_bit_bigger(smallest) + 1;

    per_size_pinned_blocks **host_block_containers =
        gallatin::utils::get_host_version<per_size_pinned_blocks *>(num_trees);

    for (uint64_t i = 0; i < num_trees; i++) {
      host_block_containers[i] =
          per_size_pinned_blocks::generate_on_device(blocks_per_segment);

      blocks_per_segment = blocks_per_segment / 2;

      if (blocks_per_segment < min) blocks_per_segment = min;
    }

    host_version->block_containers =
        gallatin::utils::move_to_device<per_size_pinned_blocks *>(
            host_block_containers, num_trees);

    return gallatin::utils::move_to_device<my_type>(host_version);
  }


  static __host__ my_type *generate_on_device(uint64_t blocks_per_segment){
  	return generate_on_device(blocks_per_segment, 1);
  }


  static __host__ my_type *generate_on_device_nowait(uint64_t blocks_per_segment, uint16_t min) {
    my_type *host_version = gallatin::utils::get_host_version<my_type>();

    uint64_t num_trees = gallatin::utils::get_first_bit_bigger(biggest) -
                         gallatin::utils::get_first_bit_bigger(smallest) + 1;

    per_size_pinned_blocks **host_block_containers =
        gallatin::utils::get_host_version<per_size_pinned_blocks *>(num_trees);

    for (uint64_t i = 0; i < num_trees; i++) {
      host_block_containers[i] =
          per_size_pinned_blocks::generate_on_device_nowait(blocks_per_segment);

      blocks_per_segment = blocks_per_segment / 2;

      if (blocks_per_segment < min) blocks_per_segment = min;
    }

    host_version->block_containers =
        gallatin::utils::move_to_device<per_size_pinned_blocks *>(
            host_block_containers, num_trees);

    return gallatin::utils::move_to_device_nowait<my_type>(host_version);
  }


  static __host__ my_type *generate_on_device_nowait(uint64_t blocks_per_segment){
    return generate_on_device_nowait(blocks_per_segment, 1);
  }


  // Variant that accepts an explicit per-tree slot count. Used by
  // generate_on_device_impl to size each tree's wavefront based on what
  // the allocator can actually afford. `tree_slot_counts[i]` is the
  // number of pinned-block slots tree i should reserve.
  static __host__ my_type *generate_on_device_nowait_per_tree(
      const uint16_t *tree_slot_counts, uint64_t num_trees) {
    my_type *host_version = gallatin::utils::get_host_version<my_type>();

    per_size_pinned_blocks **host_block_containers =
        gallatin::utils::get_host_version<per_size_pinned_blocks *>(num_trees);

    for (uint64_t i = 0; i < num_trees; i++) {
      host_block_containers[i] =
          per_size_pinned_blocks::generate_on_device_nowait(
              tree_slot_counts[i]);
    }

    host_version->block_containers =
        gallatin::utils::move_to_device<per_size_pinned_blocks *>(
            host_block_containers, num_trees);

    return gallatin::utils::move_to_device_nowait<my_type>(host_version);
  }


  static __host__ void free_on_device(my_type *dev_version) {
    my_type *host_version = gallatin::utils::move_to_host<my_type>(dev_version);

    uint64_t num_trees = gallatin::utils::get_first_bit_bigger(biggest) -
                         gallatin::utils::get_first_bit_bigger(smallest) + 1;

    per_size_pinned_blocks **host_block_containers =
        gallatin::utils::move_to_host<per_size_pinned_blocks *>(
            host_version->block_containers, num_trees);

    for (uint64_t i = 0; i < num_trees; i++) {
      per_size_pinned_blocks::free_on_device(host_block_containers[i]);
    }

    cudaFreeHost(host_version);

    cudaFreeHost(host_block_containers);
  }

  __device__ per_size_pinned_blocks *get_tree_local_blocks(int tree) {
    return block_containers[tree];
  }
};

// was just curious - this verifies that the host does not boot items on kernel
// start so __shared just get initialized to 0

// struct kernel_init_test {

// 	__device__ kernel_init_test(){
// 		printf("Booting up! controlled by %llu\n",
// threadIdx.x+blockIdx.x*blockDim.x);
// 	}

// 	__device__ ~kernel_init_test(){
// 		printf("Shutting down! controlled by %llu\n",
// threadIdx.x+blockIdx.x*blockDim.x);
// 	}

// };

}  // namespace allocators

}  // namespace gallatin

#endif  // GPU_BLOCK_