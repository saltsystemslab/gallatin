#ifndef BETA_SHARED_BLOCK_STORAGE
#define BETA_SHARED_BLOCK_STORAGE

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/alloc_utils.cuh>
//#include <poggers/allocators/uint64_bitarray.cuh>
#include <poggers/counter_blocks/block.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/representations/representation_helpers.cuh>
#include <vector>

#include "assert.h"
#include "stdio.h"

// These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#define SLAB_PRINT_DEBUG 0

#define SHARED_BLOCK_COUNTER_CUTOFF 30

namespace cg = cooperative_groups;

// a pointer list managing a set section of device memory
namespace beta {

namespace allocators {

// should these start initialized? I can try it.
__global__ void beta_set_block_bitarrs(Block **blocks, uint64_t num_blocks) {
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
        poggers::utils::get_host_version<per_size_pinned_blocks>();

    uint64_t num_uints = (num_blocks - 1) / 64 + 1;

    host_version->block_bitmap =
        poggers::utils::get_device_version<uint64_t>(num_uints);

    cudaMemset(host_version->block_bitmap, 0ULL, num_uints * sizeof(uint64_t));

    host_version->blocks =
        poggers::utils::get_device_version<Block *>(num_blocks);

    host_version->num_blocks = num_blocks;

    beta_set_block_bitarrs<<<(num_blocks - 1) / 512 + 1, 512>>>(
        host_version->blocks, num_blocks);

    return poggers::utils::move_to_device<per_size_pinned_blocks>(host_version);
  }

  static __host__ void free_on_device(per_size_pinned_blocks *dev_version) {
    per_size_pinned_blocks *host_version =
        poggers::utils::move_to_host<per_size_pinned_blocks>(dev_version);

    // malloc_bitarr::free_on_device(host_version->block_bitmap);

    cudaFree(host_version->block_bitmap);

    cudaFree(host_version->blocks);

    cudaFreeHost(host_version);
  }

  __device__ int get_valid_block_index() {
    int my_smid = poggers::utils::get_smid() % num_blocks;
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

  __device__ Block *get_my_block(int id) { return blocks[id]; }

  __device__ Block *get_alt_block() {
    int my_smid = poggers::utils::get_smid();

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
    int my_smid = poggers::utils::get_smid() % num_blocks;

    return (atomicCAS((unsigned long long int *)&blocks[my_smid],
                      (unsigned long long int)old_block,
                      (unsigned long long int)new_block) ==
            (unsigned long long int)old_block);
  }

  __device__ bool swap_out_nullptr(int my_smid, Block *block_to_swap) {
    return (atomicCAS((unsigned long long int *)&blocks[my_smid], 0ULL,
                      (unsigned long long int)block_to_swap) == 0ULL);
  }


  //this function returns a valid block (if one exists)
  // in the first loop of this shared storage
  // this scheme maintains exclusive access to the block
  // by acquiring via atomicExch, thereby avoiding stale pointers
  __device__ Block * get_valid_block(int & my_smid){

    my_smid = poggers::utils::get_smid() % num_blocks;

    int original_smid = my_smid;

    while (true){


       Block * swapped = exchange_block(my_smid);

       if (swapped != nullptr){

        return swapped;

       }


       my_smid = (my_smid+1) % num_blocks;

       //captures exactly one loop before returning.
       if (my_smid == original_smid) return nullptr;

    }





  }

  //swap a block to nullptr, returning the ptr to said block
  __device__ Block * exchange_block(int my_smid){

    return (Block *) atomicExch((unsigned long long int *)&blocks[my_smid], 0ULL);

  }

  __device__ bool lock_my_block() {
    int my_smid = poggers::utils::get_smid() % num_blocks;

    int high = my_smid / 64;
    int low = my_smid % 64;

    uint64_t mask = BITMASK(low);

    uint64_t old_bits =
        atomicOr((unsigned long long int *)&block_bitmap[high], mask);

    return !(old_bits & mask);
  }

  __device__ bool unlock_my_block() {
    int my_smid = poggers::utils::get_smid() % num_blocks;

    int high = my_smid / 64;
    int low = my_smid % 64;

    uint64_t mask = BITMASK(low);

    uint64_t old_bits =
        atomicAnd((unsigned long long int *)&block_bitmap[high], ~mask);

    // true if old bit was 1
    return (old_bits & mask);
  }
};

// container has one of these per size.
template <uint64_t smallest, uint64_t biggest>
struct pinned_shared_blocks {
  using my_type = pinned_shared_blocks<smallest, biggest>;

  per_size_pinned_blocks **block_containers;

  static __host__ my_type *generate_on_device(uint64_t blocks_per_segment, uint16_t min) {
    my_type *host_version = poggers::utils::get_host_version<my_type>();

    uint64_t num_trees = poggers::utils::get_first_bit_bigger(biggest) -
                         poggers::utils::get_first_bit_bigger(smallest) + 1;

    per_size_pinned_blocks **host_block_containers =
        poggers::utils::get_host_version<per_size_pinned_blocks *>(num_trees);

    for (uint64_t i = 0; i < num_trees; i++) {
      host_block_containers[i] =
          per_size_pinned_blocks::generate_on_device(blocks_per_segment);

      blocks_per_segment = blocks_per_segment / 2;

      if (blocks_per_segment < min) blocks_per_segment = min;
    }

    host_version->block_containers =
        poggers::utils::move_to_device<per_size_pinned_blocks *>(
            host_block_containers, num_trees);

    return poggers::utils::move_to_device<my_type>(host_version);
  }


  static __host__ my_type *generate_on_device(uint64_t blocks_per_segment){
  	return generate_on_device(blocks_per_segment, 1);
  }

  static __host__ void free_on_device(my_type *dev_version) {
    my_type *host_version = poggers::utils::move_to_host<my_type>(dev_version);

    uint64_t num_trees = poggers::utils::get_first_bit_bigger(biggest) -
                         poggers::utils::get_first_bit_bigger(smallest) + 1;

    per_size_pinned_blocks **host_block_containers =
        poggers::utils::move_to_host<per_size_pinned_blocks *>(
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

}  // namespace beta

#endif  // GPU_BLOCK_