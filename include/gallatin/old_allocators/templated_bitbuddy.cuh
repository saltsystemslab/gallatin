#ifndef POGGERS_TEMPLATE_BITBUDDY
#define POGGERS_TEMPLATE_BITBUDDY

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/allocators/free_list.cuh>
#include <poggers/allocators/uint64_bitarray.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/representations/representation_helpers.cuh>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

#define LEVEL_CUTOF 0

#define PROG_CUTOFF 3

// a pointer list managing a set section of device memory
namespace poggers {

namespace allocators {

// compress index into 32 bit index
__host__ __device__ static int shrink_index(int index) {
  if (index == -1) return index;

  return index / 2;
}

template <int depth, uint64_t size_in_bytes>
struct templated_bitbuddy {
  using my_type = templated_bitbuddy<depth, size_in_bytes>;

  enum { size = size_in_bytes / 32 };

  using child_type = templated_bitbuddy<depth - 1, size>;

  enum { lowest_size = child_type::lowest_size };

  static_assert(size > 0);

  uint64_t_bitarr mask;

  child_type children[32];

  static __host__ my_type *generate_on_device() {
    my_type *dev_version;

    cudaMalloc((void **)&dev_version, sizeof(my_type));

    cudaMemset(dev_version, ~0U, sizeof(my_type));

    return dev_version;
  }

  static __host__ void free_on_device(my_type *dev_version) {
    cudaFree(dev_version);
  }

  __host__ __device__ bool valid_for_alloc(uint64_t ext_size) {
    return (size >= ext_size && ext_size >= child_type::size);
  }

  __device__ uint64_t malloc_at_level() {
    while (true) {
      int index = shrink_index(mask.get_random_active_bit_full());

      if (index == -1) return (~0ULL);

      uint64_t old = mask.unset_both_atomic(index);

      if (__popcll(old & READ_BOTH(index)) == 2) {
        return index * size;

      } else {
        mask.reset_both_atomic(old, index);
      }
    }
  }

  __device__ uint64_t malloc_child_old(uint64_t bytes_needed) {
    while (true) {
      int index = shrink_index(mask.get_random_active_bit_control());

      if (index == -1) return (~0ULL);

      if (mask.unset_lock_bit_atomic(index) & SET_SECOND_BIT(index)) {
        // valid

        uint64_t offset = children[index].malloc_offset(bytes_needed);

        if (offset == (~0ULL)) {
          mask.unset_control_bit_atomic(index);
          continue;
        }

        return index * size + offset;
      }
    }
  }

  __device__ uint64_t malloc_child(uint64_t bytes_needed) {
    while (true) {
      // mask.global_load_this();

      int index = shrink_index(mask.get_first_active_bit_control());

      if (index == -1) return (~0ULL);

      uint64_t offset;

      if ((mask & SET_SECOND_BIT(index)) && (!(mask & SET_FIRST_BIT(index)))) {
        offset = children[index].malloc_offset(bytes_needed);

      } else if (mask.unset_lock_bit_atomic(index) & SET_SECOND_BIT(index)) {
        offset = children[index].malloc_offset(bytes_needed);

      } else {
        mask.global_load_this();
        continue;
      }

      if (offset == (~0ULL)) {
        mask.unset_control_bit_atomic(index);
        continue;
      }

      return index * size + offset;
    }
  }

  // __device__ uint64_t malloc_child_v3_temp(uint64_t bytes_needed){

  // 	while (true){

  // 		//mask.global_load_this();

  // 		int index = shrink_index(mask.get_random_active_bit_control());

  // 		if (index == -1) return (~0ULL);

  // 		uint64_t offset = children[index].malloc_offset(bytes_needed);

  // 		if (offset == (~0ULL)){
  // 			mask.unset_control_bit_atomic(index);
  // 			continue;
  // 		}

  // 		if (mask & SET_FIRST_BIT(index)){

  // 			if (mask.unset_lock_bit_atomic() & SET_SECOND_BIT){

  // 			} else {

  // 				//00
  // 				children[index].free(offset);
  // 			}
  // 		}

  // 		if ((mask & SET_SECOND_BIT(index)) && (!(mask &
  // SET_FIRST_BIT(index)))){

  // 			offset = children[index].malloc_offset(bytes_needed);

  // 		} else if (mask.unset_lock_bit_atomic(index) &
  // SET_SECOND_BIT(index)){

  // 			offset = children[index].malloc_offset(bytes_needed);

  // 		} else {

  // 			mask.global_load_this();
  // 			continue;
  // 		}

  // 		if (offset == (~0ULL)){
  // 			mask.unset_control_bit_atomic(index);
  // 			continue;
  // 		}

  // 		return index*size+offset;

  // 	}

  // }

  __device__ uint64_t malloc_offset(uint64_t bytes_needed) {
    uint64_t offset;

    if (valid_for_alloc(bytes_needed)) {
      offset = malloc_at_level();

    } else {
      offset = malloc_child(bytes_needed);
    }

    return offset;
  }

  __device__ bool free_at_level(uint64_t offset) {
    if (__popcll(mask.set_both_atomic(offset) & READ_BOTH(offset)) == 0) {
      return true;
    }

    return false;
  }

  __device__ bool free(uint64_t offset) {
    uint64_t local_offset = offset / size;

    assert(local_offset < 32);

    if (children[local_offset].free(offset % size)) {
      mask.set_control_bit_atomic(local_offset);
      return true;
    }

    return free_at_level(local_offset);
  }
};

template <uint64_t size_in_bytes>
struct templated_bitbuddy<0, size_in_bytes> {
  using my_type = templated_bitbuddy<0, size_in_bytes>;

  enum { size = size_in_bytes };
  // TODO double check this, feels like it should be sizeinbytes/32;
  enum { lowest_size = size_in_bytes / 32 };

  uint64_t_bitarr mask;

  __device__ uint64_t malloc_offset(uint64_t bytes_needed) {
    return malloc_at_level();
  }

  __device__ uint64_t malloc_at_level() {
    while (true) {
      int index = shrink_index(mask.get_random_active_bit_full());

      if (index == -1) return (~0ULL);

      if (__popcll(mask.unset_both_atomic(index) & READ_BOTH(index)) == 2) {
        return index;
      }
    }
  }

  // returns true if entirely full
  __device__ bool free_at_level(uint64_t offset) {
    if (__popcll(mask.set_both_atomic(offset) & READ_BOTH(offset)) == 0) {
      return true;
    }

    return false;
  }

  __device__ bool free(uint64_t offset) { return free_at_level(offset); }

  __host__ __device__ bool valid_for_alloc(uint64_t size) { return true; }

  static __host__ my_type *generate_on_device() {
    my_type *dev_version;

    cudaMalloc((void **)&dev_version, sizeof(my_type));

    cudaMemset(dev_version, ~0U, sizeof(my_type));

    return dev_version;
  }

  static __host__ void free_on_device(my_type *dev_version) {
    cudaFree(dev_version);
  }
};

// 32 should return 0

template <uint64_t num_allocations>
struct determine_depth {
  enum {
    depth = num_allocations > 32
                ? determine_depth<num_allocations / 32>::depth + 1
                : 0
  };
};

template <>
struct determine_depth<0> {
  enum { depth = 32 };
};

template <int depth>
struct determine_num_allocations {
  enum { count = 32 * determine_num_allocations<depth - 1>::count };
};

template <>
struct determine_num_allocations<0> {
  enum { count = 32 };
};

template <uint64_t num_allocations, uint64_t size_of_allocation>
struct bitbuddy_allocator {
  using my_type = bitbuddy_allocator<num_allocations, size_of_allocation>;

  enum {
    total_size =
        size_of_allocation * determine_num_allocations<
                                 determine_depth<num_allocations>::depth>::count
  };

  using bitbuddy_type =
      templated_bitbuddy<determine_depth<num_allocations>::depth,
                         determine_num_allocations<
                             determine_depth<num_allocations>::depth>::count>;

  bitbuddy_type *internal_allocator;

  void *memory;

  static __host__ my_type *generate_on_device() {
    my_type host_version;

    void *ext_memory;

    host_version.internal_allocator = bitbuddy_type::generate_on_device();

    if (host_version.internal_allocator == nullptr) {
      printf("Allocator could not get enough space\n");
      assert(1 == 0);
    }

    cudaMalloc((void **)&ext_memory, num_allocations * size_of_allocation);

    if (ext_memory == nullptr) {
      cudaFree(host_version.internal_allocator);

      printf(
          "Allocator could not get enough memory to handle requested # "
          "allocations.\n");
      assert(1 == 0);
    }

    host_version.memory = ext_memory;

    my_type *dev_version;

    // my type is 16 bytes. I'm gonna conservatively estimate that this will
    // always go through.
    cudaMalloc((void **)&dev_version, sizeof(my_type));

    cudaMemcpy(dev_version, &host_version, sizeof(my_type),
               cudaMemcpyHostToDevice);

    return dev_version;
  }

  static __host__ void free_on_device(my_type *dev_version) {
    my_type host_version;

    cudaMemcpy(&host_version, dev_version, sizeof(my_type),
               cudaMemcpyDeviceToHost);

    cudaFree(dev_version);

    cudaFree(host_version.memory);

    bitbuddy_type::free_on_device(host_version.internal_allocator);
  }

  __device__ void *malloc(uint64_t bytes_needed) {
    uint64_t offset = internal_allocator->malloc_offset(
        (bytes_needed - 1) / size_of_allocation + 1);

    if (offset == (~0ULL)) return nullptr;

    return (void *)((uint64_t)memory + offset * size_of_allocation);
  }

  __device__ bool free(void *allocation) {
    uint64_t offset =
        ((uint64_t)allocation - (uint64_t)memory) / size_of_allocation;

    return internal_allocator->free(offset);
  }

  __host__ __device__ bool is_bitbuddy_alloc(void *allocation) {
    // all allocations from the bitbuddy must be in the stride of the bitbuddy.
    // since internals are offset by light of their allocation this is an easy
    // check.
    return ((uint64_t)allocation % bitbuddy_type::lowest_size == allocation);

    // could also make sure offset is internal but whatevs.
    // kind of on you to not free from the wrong allocator lol
    // uint64_t offset = ((uint64_t allocation) - (uint64_t) memory);
  }
};

}  // namespace allocators

}  // namespace poggers

#endif  // GPU_BLOCK_