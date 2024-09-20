#ifndef SEGMENT_HELPER_CU
#define SEGMENT_HELPER_CU

/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <segment_helpers.hpp>
#include <gallatin/allocators/segment.cuh>


using namespace gallatin::internals;


//using internal_bitarry_type = internal_bitarray;

//kernels



__global__ void set_reset_kernel(segment<1, 1, 4096> * segment_ptr){


  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  if (!segment_ptr->set_size_capacity(0, 1)){

    write_global_log(4, 0);

  }

  if (segment_ptr->set_size_capacity(2, 1)){
    write_global_log(4,1);
  }

  if (!segment_ptr->set_invalid(0, 1)){
    write_global_log(4, 2);
  }

  if (!segment_ptr->set_size_capacity(2, 1)){

    write_global_log(4, 3);

  }

  if (segment_ptr->set_size_capacity(0, 1)){
    write_global_log(4,4);
  }

}


__global__ void test_block_malloc_kernel(segment<1, 1, 4096> * segment_ptr){


  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  if (!segment_ptr->set_size_capacity(0, 1)){

    write_global_log(4, 0);

  }

  bool last = false;
  uint16_t block = segment_ptr->reserve_block(0,last);

  if (block != 0 || !last){
    write_global_log(4,1);
  }

  for (uint64_t i = 0; i < 4096; i++){

    auto active_threads = cg::coalesced_threads();

    bool reset_segment = false;

    uint64_t offset = segment_ptr->allocate_offset_from_block(active_threads, 1, block, 0, 0, reset_segment);

    if (offset != i){

      write_global_log(4, 2, i);
      return;
    }

  }

  for (uint64_t i = 0; i < 4095; i++){

    bool first_block = false;

    uint16_t read_size = ~0;

    if (segment_ptr->return_offset(i, read_size, first_block) || first_block || read_size != 0){
      write_global_log(4,3,i);
    }

  }

  if (segment_ptr->set_invalid(0, 1)){
    write_global_log(4,4);
  }

  bool first_block = false;

  uint16_t read_size = ~0;

  if (!segment_ptr->return_offset(4095, read_size, first_block) || ! first_block || read_size != 0){
    write_global_log(4,5);
  }

  if (!segment_ptr->set_invalid(0, 1)){
    write_global_log(4,6);
  }

}


__global__ void test_fail_block_malloc_kernel(segment<1, 1, 4096> * segment_ptr){


  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  if (!segment_ptr->set_size_capacity(0, 1)){

    write_global_log(4, 0);

  }

  bool last = false;
  uint16_t block = segment_ptr->reserve_block(0, last);

  if (block != 0 || !last){
    write_global_log(4,1);
  }

  for (uint64_t i = 0; i < 4096; i++){

    auto active_threads = cg::coalesced_threads();

    bool reset_segment = false;

    uint64_t offset = segment_ptr->allocate_offset_from_block(active_threads, 1, block, 0, 1, reset_segment);

    if (offset != ~0ULL){

      write_global_log(4, 2, i);
      return;
    }

    if (i == 4095){

      if (!reset_segment){
        write_global_log(4,8);
      }

      if (!segment_ptr->set_invalid(0, 1)){
        write_global_log(4,6);
      }

    }

  }


}



__global__ void setup_parallel_segment_kernel(segment<256, 16, 16777216> * device_segment){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  device_segment->set_size_capacity(0, 256);

  for (uint64_t i =0; i < 256; i++){
    bool last;
    device_segment->reserve_block(0, last);
  }

}

__global__ void test_parallel_malloc_kernel(segment<256, 16, 16777216> * device_segment, uint64_t * allocs, uint64_t n_allocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint64_t block = tid/4096;


  auto active_threads = cg::coalesced_threads();

  bool reset_segment = false;


  uint64_t offset = device_segment->allocate_offset_from_block(active_threads, 1, block, 1025, 0, reset_segment);

  allocs[tid] = offset;

}

__global__ void test_parallel_free_kernel(segment<256, 16, 16777216> * device_segment, uint64_t * allocs, uint64_t n_allocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint16_t read_size;

  bool first_freed;

  device_segment->return_offset(allocs[tid] % 16777216, read_size, first_freed);

}



__global__ void test_claim_loop_kernel(segment<256, 1, 1048576> * segment_ptr){


  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  if (!segment_ptr->set_size_capacity(0, 256)){

    write_global_log(4, 0);

  }


  for (uint16_t i = 0; i < 256; i++){

    bool last = false;
    uint16_t block = segment_ptr->reserve_block(0, last);

    if (block != i){
      write_global_log(4,1,i);
    }

    if (i==255 && !last){
      write_global_log(4,1,256);
    }
  }

  bool last = false;
  uint16_t fail_block = segment_ptr->reserve_block(0, last);

  if (fail_block != segment_ptr->fail_size()){
    write_global_log(4,2);
  }

  //return them in opposite order
  for (uint16_t i = 256; i > 0; i--){
    uint16_t return_index = i-1;
    bool first_block = false;

    segment_ptr->free_block(return_index, first_block);

    if (i==256 && !first_block){
      write_global_log(4,4,i);
    }
  }

  for (uint16_t i = 256; i > 0; i--){

    uint16_t alloc_index = i-1;
    bool last = false;
    if (segment_ptr->reserve_block(0,last) != alloc_index){
      write_global_log(4,3,i);
    }

    if (alloc_index == 0 && !last){
      write_global_log(4,3,256);
    }

  }

}

//end of kernels

//end of helper functions

void segment_helper_tests::open_global_log(){

  gallatin::internals::init_global_error_log();
}

int segment_helper_tests::close_global_log(){
  return gallatin::internals::close_global_error_log();
}


bool segment_helper_tests::testInit(){

  open_global_log();

  auto device_segment = segment<1, 1, 4096>::generate_on_device(1);


  segment<1, 1, 4096>::free_on_device(device_segment);

  GPUErrorCheck(cudaDeviceSynchronize());



  return (close_global_log() == 0);

}

bool segment_helper_tests::testSetReset(){

  open_global_log();

  auto device_segment = segment<1, 1, 4096>::generate_on_device(1);


  set_reset_kernel<<<1,1>>>(device_segment);

  GPUErrorCheck(cudaDeviceSynchronize());

  segment<1, 1, 4096>::free_on_device(device_segment);

  GPUErrorCheck(cudaDeviceSynchronize());



  return (close_global_log() == 0);

}


bool segment_helper_tests::testBlockMalloc(){

  open_global_log();

  auto device_segment = segment<1, 1, 4096>::generate_on_device(1);


  test_block_malloc_kernel<<<1,1>>>(device_segment);

  GPUErrorCheck(cudaDeviceSynchronize());

  segment<1, 1, 4096>::free_on_device(device_segment);

  GPUErrorCheck(cudaDeviceSynchronize());



  return (close_global_log() == 0);


}

bool segment_helper_tests::testBlockMallocFail(){

  open_global_log();

  auto device_segment = segment<1, 1, 4096>::generate_on_device(1);



  test_fail_block_malloc_kernel<<<1,1>>>(device_segment);

  GPUErrorCheck(cudaDeviceSynchronize());

  segment<1, 1, 4096>::free_on_device(device_segment);

  GPUErrorCheck(cudaDeviceSynchronize());



  return (close_global_log() == 4096);


}


bool segment_helper_tests::testClaimAllLoop(){

  open_global_log();

  auto device_segment = segment<256, 1, 1048576>::generate_on_device(1);


  
  test_claim_loop_kernel<<<1,1>>>(device_segment);

  GPUErrorCheck(cudaDeviceSynchronize());

  segment<256, 1, 1048576>::free_on_device(device_segment);

  GPUErrorCheck(cudaDeviceSynchronize());



  return (close_global_log() == 0);


}


bool segment_helper_tests::testParallel(){

  open_global_log();

  auto device_segment = segment<256, 16, 16777216>::generate_on_device(1);

  uint64_t n_allocs = 1048576;

  uint64_t * allocs = gallatin::utils::get_device_version<uint64_t>(n_allocs);

  setup_parallel_segment_kernel<<<1,1>>>(device_segment);

  cudaDeviceSynchronize();

  test_parallel_malloc_kernel<<<(n_allocs-1)/256+1,256>>>(device_segment, allocs, n_allocs);

  test_parallel_free_kernel<<<(n_allocs-1)/256+1,256>>>(device_segment, allocs, n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());

  cudaFree(allocs);

  return (close_global_log() == 0);


}


#endif