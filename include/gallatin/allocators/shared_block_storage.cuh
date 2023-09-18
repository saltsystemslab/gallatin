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

#define GAL_BLOCK_STORAGE_READ_BLOCK_ATOMIC 1

// should these start initialized? I can try it.
__global__ void gallatin_set_block_bitarrs(Block **blocks, uint64_t num_blocks) {
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

  uint64_t *block_bitmap;

  Block **blocks;

  static __host__ per_size_pinned_blocks *generate_on_device(
      uint64_t num_blocks) {
    if (num_blocks == 0) num_blocks = 1;

    per_size_pinned_blocks *host_version =
        gallatin::utils::get_host_version<per_size_pinned_blocks>();

    uint64_t num_uints = (num_blocks - 1) / 64 + 1;

    host_version->block_bitmap =
        gallatin::utils::get_device_version<uint64_t>(num_uints);

    cudaMemset(host_version->block_bitmap, 0ULL, num_uints * sizeof(uint64_t));

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

    uint64_t num_uints = (num_blocks - 1) / 64 + 1;

    host_version->block_bitmap =
        gallatin::utils::get_device_version<uint64_t>(num_uints);

    cudaMemset(host_version->block_bitmap, 0ULL, num_uints * sizeof(uint64_t));

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

    // malloc_bitarr::free_on_device(host_version->block_bitmap);

    cudaFree(host_version->block_bitmap);

    cudaFree(host_version->blocks);

    cudaFreeHost(host_version);
  }

  __device__ int get_valid_block_index() {
    int my_smid = gallatin::utils::get_smid() % num_blocks;
    int original_smid = my_smid;

    int counter = 0;

    // addition - loop to find valid block.
    while (blocks[my_smid] == nullptr && my_smid != (original_smid - 1)) {
      my_smid = (my_smid + 1) % num_blocks;

      counter += 1;
      if (counter >= SHARED_BLOCK_COUNTER_CUTOFF) break;
    }

    return my_smid;
  }

  __device__ Block *get_my_block(int id) {

    #if GAL_BLOCK_STORAGE_READ_BLOCK_ATOMIC
      return (Block *) gallatin::utils::ldca((uint64_t *)&blocks[id]);
    #else 
      return blocks[id]; 
    #endif
    

 }

  __device__ Block *get_alt_block() {
    int my_smid = gallatin::utils::get_smid();

    my_smid = my_smid * my_smid % num_blocks;

    return blocks[my_smid];
  }

  // replace block with nullptr.
  __device__ bool swap_out_block(int my_smid, Block *block_to_swap) {
    return (atomicCAS((unsigned long long int *)&blocks[my_smid],
                      (unsigned long long int)block_to_swap,
                      0ULL) == (unsigned long long int)block_to_swap);
  }

  __device__ bool replace_block(Block *old_block, Block *new_block) {
    int my_smid = gallatin::utils::get_smid() % num_blocks;

    return (atomicCAS((unsigned long long int *)&blocks[my_smid],
                      (unsigned long long int)old_block,
                      (unsigned long long int)new_block) ==
            (unsigned long long int)old_block);
  }

  __device__ bool swap_out_nullptr(int my_smid, Block *block_to_swap) {
    return (atomicCAS((unsigned long long int *)&blocks[my_smid], 0ULL,
                      (unsigned long long int)block_to_swap) == 0ULL);
  }

  __device__ bool lock_my_block() {
    int my_smid = gallatin::utils::get_smid() % num_blocks;

    int high = my_smid / 64;
    int low = my_smid % 64;

    uint64_t mask = BITMASK(low);

    uint64_t old_bits =
        atomicOr((unsigned long long int *)&block_bitmap[high], mask);

    return !(old_bits & mask);
  }

  __device__ bool unlock_my_block() {
    int my_smid = gallatin::utils::get_smid() % num_blocks;

    int high = my_smid / 64;
    int low = my_smid % 64;

    uint64_t mask = BITMASK(low);

    uint64_t old_bits =
        atomicAnd((unsigned long long int *)&block_bitmap[high], ~mask);

    // true if old bit was 1
    return (old_bits & mask);
  }

  __device__ uint64_t calculate_overhead(){

    return num_blocks/8+num_blocks*sizeof(Block *);

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