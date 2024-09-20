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

namespace internals {

  struct block_storage {

    GALLATIN_TODO(make nblocks constant on device.)
    uint32_t n_blocks;
    uint32_t * packed_block_segments;

    __host__ static block_storage * generate_on_device(uint32_t ext_n_blocks){

      block_storage * host_version = gallatin::utils::get_host_version<block_storage>();

      host_version->n_blocks = ext_n_blocks;

      host_version->packed_block_segments = gallatin::utils::get_device_version<uint32_t>(ext_n_blocks);

      return gallatin::utils::move_to_device_nowait(host_version);

    }

    __host__ static void free_on_device(block_storage * device_version){


      block_storage * host_version = gallatin::utils::move_to_host(device_version);

      cudaFree(host_version->packed_block_segments);

      cudaFreeHost(host_version);

    }


    //pack segment and block together, mark claimed and not dead
    __device__ static uint32_t pack_segment_block(uint32_t segment, uint16_t block){

      return segment << BS_BLOCK_ID_CUTOFF | block << 2 | SET_BIT_MASK(BS_SET_BIT);


    }

    //segment is upper "BS_BLOCK_ID_CUTOFF"
    __device__ uint32_t extract_segment(uint32_t packed_value){

      return packed_value >> BS_BLOCK_ID_CUTOFF;

    }

    __device__ uint16_t extract_block(uint32_t packed_value){


      return (packed_value & BITMASK(BS_BLOCK_ID_CUTOFF)) >> 2;


    }

    __device__ bool is_set(uint32_t read_value){
      return read_value & SET_BIT_MASK(BS_SET_BIT);
    }


    //set to 1 if "dead"
    //dead markers should always be replaced
    __device__ bool is_not_dead(uint32_t read_value){
      return !(read_value & SET_BIT_MASK(BS_DEAD_BIT));
    }

    __device__ bool is_dead(uint32_t read_value){
      return read_value & SET_BIT_MASK(BS_DEAD_BIT);
    }

    __device__ bool claim_to_set(uint index){

      //unsets the "set" bit, marking the segment as uninitialized.
      return atomicAnd((unsigned int *)&packed_block_segments[index], ~((uint32_t) SET_BIT_MASK(BS_SET_BIT))) & SET_BIT_MASK(BS_SET_BIT);

    }

    __device__ bool claim_to_set_exact(uint index, uint16_t block, uint32_t segment){

      #if GALLATIN_DEBUG

      if (index >= n_blocks){
        printf("Index %u greater than max %u\n", index, n_blocks);
      }


      #endif

      uint32_t packed = pack_segment_block(segment, block);

      uint32_t packed_cleaned = packed & ~((uint32_t) SET_BIT_MASK(BS_SET_BIT));

      return (atomicCAS(&packed_block_segments[index], packed, packed_cleaned) == packed);

    }

    //marks object as both "dead" and "set"
    // this is weird but allows for the pair to be "claimed" by a future thread with the intent of reviving.
    __device__ void mark_dead(uint index){

      constexpr uint32_t dead_marker = (SET_BIT_MASK(BS_SET_BIT) | SET_BIT_MASK(BS_DEAD_BIT));

      atomicExch((unsigned int *)&packed_block_segments[index], dead_marker);

    }

    __device__ void set(uint index, uint32_t packed_value){

      atomicExch((unsigned int *)&packed_block_segments[index], packed_value);

    }

    //find a valid block if one exists, return false otherwise.
    //if true populates.
    //if not true returns my home index for set.
    __device__ bool get_valid_block(uint32_t & segment, uint16_t & block, uint & index){

      uint my_smid = gallatin::utils::get_smid() % n_blocks;


      uint32_t loaded_packed_val = gallatin::utils::ld_acq(&packed_block_segments[my_smid]);

      
      if (is_set(loaded_packed_val) && is_not_dead(loaded_packed_val)){

        block = extract_block(loaded_packed_val);

        segment = extract_segment(loaded_packed_val);

        index = my_smid;

        return true;
      }

      if (is_dead(loaded_packed_val)){
        
        index = my_smid;

        return false;
      }



      for (uint i =1; i < n_blocks; i++){

        index = (i+my_smid) % n_blocks;

        uint32_t loaded_packed_val = gallatin::utils::ld_acq(&packed_block_segments[index]);

        if (is_set(loaded_packed_val) && is_not_dead(loaded_packed_val)){

        block = extract_block(loaded_packed_val);

        segment = extract_segment(loaded_packed_val);

        return true;


        }


      }

      //fail!

      index = my_smid;

      return false;

    }



  };


}  // namespace allocators

}  // namespace gallatin

#endif  // GPU_BLOCK_