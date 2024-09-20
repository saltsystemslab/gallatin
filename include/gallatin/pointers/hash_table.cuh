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

    __host__ static free_on_device(block_storage * device_version){


      block_storage * host_version = gallatin::utils::move_to_host(device_version);

      cudaFree(host_version->packed_block_segments);

      cudaFreeHost(host_version);

    }


    //pack segment and block together, mark claimed.
    __device__ static uint32_t pack_segment_block(uint32_t segment, uint16_t block){

      return segment << BS_BLOCK_ID_CUTOFF | block << 2 | SET_BIT_MASK(BS_SET_BIT);


    }

    __device__ bool is_set(uint32_t read_value){
      return read_value & SET_BIT_MASK(BS_SET_BIT)
    }


    //set to 1 if "dead"
    //dead markers should always be replaced
    __device__ bool is_not_dead(uint32_t read_value){
      return !(read_value & SET_BIT_MASK(BS_SET_BIT));
    }

    __device__ bool claim_to_set(uint index){

    }

    //find a valid block if one exists, return false otherwise.
    //if true populates.
    //if not true returns my home index for set.
    __device__ bool get_valid_block(uint32_t & segment, uint16_t & block, uint & index){

      uint my_smid = gallatin::utils::get_smid();

      for (uint i =0; i < n_blocks; i++){

       index = i+my_smid % n_blocks;

        uint32_t loaded_packed_val = gallatin::utils::ld_acq(*packed_block_segments[index]);

        if (is_set(loaded_packed_val) && is_not_dead(loaded_packed_val)){

        block = extract_block(loaded_packed_val);

        segment = extract_segment(loaded_packed_val);

        return index;


        }
      }

      //fail!

      index = my_smid;

      return false;

    }

    __device__ int claim_dead_index(uint32_t & segment, uint16_t &  block, uint & index){
      
    }



  };


}  // namespace allocators

}  // namespace gallatin

#endif  // GPU_BLOCK_