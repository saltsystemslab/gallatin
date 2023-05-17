#ifndef SHARED_SLAB_ONE_SIZE
#define SHARED_SLAB_ONE_SIZE

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/counter_blocks/block.cuh>
#include <poggers/counter_blocks/one_size_allocator.cuh>
#include <poggers/counter_blocks/shared_block_storage.cuh>
#include <vector>

#include "assert.h"
#include "stdio.h"

// These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#define SLAB_ONE_SIZE_MAX_ATTEMPTS 50

#define SHARED_SLAB_DEBUG 0

namespace cg = cooperative_groups;

// a pointer list managing a set section of device memory
namespace beta {

namespace allocators {

#if SHARED_SLAB_DEBUG

template <typename allocator>
__global__ void check_block_kernel(allocator *alloc, uint64_t num_blocks) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= num_blocks) return;

  alloc->check_block(tid);
}

#endif

template <typename allocator>
__global__ void boot_memory_kernel(allocator *alloc, uint64_t num_blocks) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= num_blocks) return;

  alloc->boot_block(tid);
}

// a modification on the one_size_slab_allocator
//  this uses the shared block component from BETA
//  to debug it in an incremental environment.
template <int extra_blocks>
struct shared_slab_allocator {
  using my_type = shared_slab_allocator<extra_blocks>;

  // doesn't seem necessary tbh
  // uint64_t offset_size;
  uint64_t offset_size;
  uint64_t num_blocks;
  one_size_allocator *block_allocator;
  // one_size_allocator * mem_alloc;
  char *extra_memory;

  per_size_pinned_blocks *malloc_containers;

#if SHARED_SLAB_DEBUG
  uint64_t *debug_array;
#endif

  // add hash table type here.
  // map hashes to bytes?

  static __host__ my_type *generate_on_device(uint64_t num_allocs,
                                              uint64_t ext_size) {
    my_type *host_version;

    cudaMallocHost((void **)&host_version, sizeof(my_type));

    host_version->offset_size = ext_size;

    uint64_t num_pinned_blocks = (num_allocs - 1) / 4096 + 1;

    host_version->num_blocks = num_pinned_blocks;

    host_version->block_allocator = one_size_allocator::generate_on_device(
        num_pinned_blocks, sizeof(Block), 17);

    // host_version->mem_allocator =
    // one_size_allocator::generate_on_device(num_pinned_blocks, 4096*ext_size,
    // 1324);

    char *host_ptr_ext_mem;
    cudaMalloc((void **)&host_ptr_ext_mem, num_pinned_blocks * ext_size * 4096);

    if (host_ptr_ext_mem == nullptr) {
      throw std::runtime_error("main malloc buffer failed to be acquired.\n");
    }

    host_version->extra_memory = host_ptr_ext_mem;

    host_version->malloc_containers =
        per_size_pinned_blocks::generate_on_device(extra_blocks);

#if SHARED_SLAB_DEBUG

    uint64_t *debug_array_ptr;

    cudaMalloc((void **)&debug_array_ptr, sizeof(uint64_t) * num_pinned_blocks);

    cudaMemset(debug_array_ptr, 0, sizeof(uint64_t) * num_pinned_blocks);

    host_version->debug_array = debug_array_ptr;

#endif

    my_type *dev_version;

    cudaMalloc((void **)&dev_version, sizeof(my_type));

    cudaMemcpy(dev_version, host_version, sizeof(my_type),
               cudaMemcpyHostToDevice);

    cudaFreeHost(host_version);

    cudaDeviceSynchronize();

    boot_memory_kernel<my_type><<<(extra_blocks - 1) / 512 + 1, extra_blocks>>>(
        dev_version, extra_blocks);
    cudaDeviceSynchronize();

    return dev_version;
  }

  static __host__ void free_on_device(my_type *dev_version) {
    my_type *host_version;
    cudaMallocHost((void **)&host_version, sizeof(my_type));

    cudaMemcpy(host_version, dev_version, sizeof(my_type),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    one_size_allocator::free_on_device(host_version->block_allocator);
    // one_size_allocator::free_on_device(host_version->mem_allocator);

    cudaFree(host_version->extra_memory);

    per_size_pinned_blocks::free_on_device(host_version->malloc_containers);

#if SHARED_SLAB_DEBUG

    cudaFree(host_version->debug_array);

#endif

    cudaFree(dev_version);

    cudaFreeHost(host_version);

    return;
  }

  // returns address universe.
  __device__ uint64_t get_largest_allocation_offset() {
    return block_allocator->get_largest_allocation() * 4096;
  }

  // initialize a block for the first time.
  __device__ void boot_block(int smid) {
    uint64_t slab_offset = block_allocator->get_offset();

    if (slab_offset == one_size_allocator::fail()) {
      printf("Error: Not enough space to boot block %d\n", smid);
      return;
    }

    Block *new_block =
        (Block *)block_allocator->get_mem_from_offset(slab_offset);

    if (!malloc_containers->swap_out_nullptr(smid, new_block)) {
      printf("Error: Block in position %d already set to not nullptr\n", smid);
    }
  }

  // maybe replace block.
  // on failure we return true
  // this keeps all blocks at maximum throughput
  //  but assumes that blocks must come pre-loaded
  //  Space is saved on larger versions by reducing the # of live blocks.
  __device__ bool replace_block(int smid, Block *my_block) {
    if (malloc_containers->swap_out_block(smid, my_block)) {
      uint64_t slab_offset = block_allocator->get_offset();

      if (slab_offset == one_size_allocator::fail()) {
        return false;
      }

      Block *new_block =
          (Block *)block_allocator->get_mem_from_offset(slab_offset);

      // this is a bug as only one thread should enter this phase
      // so ownership should be exclusive.
      if (!malloc_containers->swap_out_nullptr(smid, new_block)) {
        printf("Incorrect behavior when swapping out blocks");
        block_allocator->free_offset(slab_offset);
        return false;
      }
    }

    return true;
  }

  // request an allocation
  // returns a void * to the allocation
  // or a nullptr if no allocations available.

  __device__ void *malloc() {
    //__shared__ warp_lock team_lock;

    // declare block ahead of time, all threads copy from leader.
    int malloc_container_index;
    Block *my_block;

    int num_attempts = 0;

    while (num_attempts < SLAB_ONE_SIZE_MAX_ATTEMPTS) {
      cg::coalesced_group coalesced_team = cg::coalesced_threads();

      if (coalesced_team.thread_rank() == 0) {
        malloc_container_index = malloc_containers->get_valid_block_index();
        my_block = malloc_containers->get_my_block(malloc_container_index);
      }

      // blocking - distribute block info to all threads in the team.
      malloc_container_index = coalesced_team.shfl(malloc_container_index, 0);
      my_block = coalesced_team.shfl(my_block, 0);

      // if block is stale, someone is queued to replace it.
      // this should only happen rarely.
      if (my_block == nullptr) {
        // team.sync();
        num_attempts += 1;
        continue;
      }

      uint64_t block_id = block_allocator->get_offset_from_address(my_block);

      if (block_id >= num_blocks) {
        printf("Wrong block\n");
      }

      // get offset from block.
      uint64_t allocation = my_block->block_malloc(coalesced_team);

      if (allocation == ~0ULL) {
        if (coalesced_team.thread_rank() == 0) {
          replace_block(malloc_container_index, my_block);
        }

      } else {
        return (void *)(extra_memory +
                        (block_id * 4096 + allocation) * offset_size);
      }

      num_attempts += 1;
    }

    return nullptr;
  }

  __device__ uint64_t get_offset_from_ptr(void *ext_ptr) {
    // first off cast to uint64_t

    uint64_t ext_as_bits = (uint64_t)ext_ptr;

    // now downshift and subtract

    ext_as_bits = ext_as_bits - (uint64_t)extra_memory;

    ext_as_bits = ext_as_bits / offset_size;

    return ext_as_bits;
  }

  __device__ void *get_ptr_from_offset(uint64_t offset) {
    uint64_t ext_as_bits = offset * offset_size;

    uint64_t mem_offset = (uint64_t)extra_memory + ext_as_bits;

    return (void *)mem_offset;
  }

  __device__ uint64_t malloc_offset() {
    void *alloc = malloc();

    if (alloc == nullptr) return ~0ULL;

    return get_offset_from_ptr(alloc);
  }

  // in the one allocator scheme free is simplified - get the block and free
  // if the block we free to is unpinned, we can safely return the memory to the
  // veb tree
  __device__ void free(void *ext_allocation) {
    uint64_t allocation_offset = get_offset_from_ptr(ext_allocation);

    uint64_t slab_offset = allocation_offset / 4096;

    // this is nonatomic- disable
    //  if (block_allocator->query(slab_offset)){
    //  	printf("Slab %llu is attached before its time\n", slab_offset);
    //  }

    Block *slab = (Block *)block_allocator->get_mem_from_offset(slab_offset);

    if (slab->block_free()) {
      // slabs that are marked unpinned cannot be reattached - therefore, this
      // read succeeding guarantees correctness.

      // printf("Returning block\n");

      slab->reset_block();

      block_allocator->free_offset(slab_offset);

#if SHARED_SLAB_DEBUG
      if (!block_allocator->query(slab_offset)) {
        printf("Slab %llu failed to attach to the tree\n", slab_offset);
      }
#endif
    }
  }

  __device__ void free_count_misses(void *ext_allocation, uint64_t *misses) {
    uint64_t allocation_offset = get_offset_from_ptr(ext_allocation);

    uint64_t slab_offset = allocation_offset / 4096;

    // if (block_allocator->query(slab_offset)){
    // 	printf("Slab %llu is attached before its time\n", slab_offset);
    // }

    Block *slab = (Block *)block_allocator->get_mem_from_offset(slab_offset);

    if (slab->block_free()) {
      // slab may be available for free - no pins in this version, can always
      // free multiple people may unpin?

      slab->reset_block();

      block_allocator->free_offset(slab_offset);

      atomicAdd((unsigned long long int *)&misses[0], 1ULL);

      if (!block_allocator->query(slab_offset)) {
        printf("Slab %llu failed to attach to the tree\n", slab_offset);
      }
    }
  }

  __host__ one_size_allocator *get_block_allocator_host() {
    my_type *host_full_alloc;

    cudaMallocHost((void **)&host_full_alloc, sizeof(my_type));

    cudaMemcpy(host_full_alloc, this, sizeof(my_type), cudaMemcpyDeviceToHost);

    one_size_allocator *host_block_allocator;

    cudaMallocHost((void **)&host_block_allocator, sizeof(one_size_allocator));

    cudaMemcpy(host_block_allocator, host_full_alloc->block_allocator,
               sizeof(one_size_allocator), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFreeHost(host_full_alloc);

    return host_block_allocator;
  }

  // report number of single size allocations available.
  __host__ uint64_t report_fill() {
    one_size_allocator *host_block_allocator = get_block_allocator_host();

    uint64_t fill = host_block_allocator->report_fill();

    cudaFreeHost(host_block_allocator);

    return fill;
  }

  __host__ uint64_t report_max() {
    one_size_allocator *host_block_allocator = get_block_allocator_host();

    uint64_t max_fill = host_block_allocator->report_max();

    cudaFreeHost(host_block_allocator);

    return max_fill;
  }
};

}  // namespace allocators

}  // namespace beta

#endif  // GPU_BLOCK_