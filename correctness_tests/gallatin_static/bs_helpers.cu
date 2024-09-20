#ifndef BS_HELPER_CU
#define BS_HELPER_CU

/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <bs_helpers.hpp>
#include <gallatin/allocators/block_storage.cuh>


using namespace gallatin::internals;


//using internal_bitarry_type = internal_bitarray;


__global__ void test_pack_kernel(block_storage * storage){


  for (uint segment = 0; segment < 1000; segment++){

    for (uint16_t block = 0; block < 256; block++){

      uint32_t pack = storage->pack_segment_block(segment, block);

      uint32_t extracted_segment = storage->extract_segment(pack);

      uint16_t packed_block = storage->extract_block(pack);

      if (extracted_segment != segment || packed_block != block){
        write_global_log(4, 0, segment, block);
      }

    }

  }


}


__global__ void test_set_kernel(block_storage * storage){


  for (uint i = 0; i < 10; i++){

    uint16_t block = i;


    storage->set(i, storage->pack_segment_block(i, block));


  }

  for (uint i = 0; i < 10; i++){

    uint32_t segment;

    uint16_t block;
    uint index;

    if (!storage->get_valid_block(segment, block, index)){
      write_global_log(4, 1, i);
    }

    if (index != i){
      write_global_log(4, 2, i);
    }

    if (segment != i || block != i){
      write_global_log(4,4,i);
    }

    if (!storage->claim_to_set(index)){
      write_global_log(4, 3, i);
    }


  }


}

//kernels

bool bs_helper_tests::testStorageInit(){

  open_global_log();

  block_storage * storage = block_storage::generate_on_device(1);


  block_storage::free_on_device(storage);



  return (close_global_log() == 0);


}

bool bs_helper_tests::testPacking(){

  open_global_log();

  block_storage * storage = block_storage::generate_on_device(1);

  test_pack_kernel<<<1,1>>>(storage);

  GPUErrorCheck(cudaDeviceSynchronize());

  block_storage::free_on_device(storage);

  return (close_global_log() == 0);


}

bool bs_helper_tests::testSetUnset(){

  open_global_log();

  block_storage * storage = block_storage::generate_on_device(10);

  test_set_kernel<<<1,1>>>(storage);

  GPUErrorCheck(cudaDeviceSynchronize());

  block_storage::free_on_device(storage);

  return (close_global_log() == 0);


}


void bs_helper_tests::open_global_log(){

  gallatin::internals::init_global_error_log();
}

int bs_helper_tests::close_global_log(){
  return gallatin::internals::close_global_error_log();
}






#endif