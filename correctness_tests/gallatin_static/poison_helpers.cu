#ifndef POISON_HELPER_CU
#define POISON_HELPER_CU

/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */


#define GALLATIN_DEBUG 1


#include <gallatin/allocators/global_allocator.cuh>
#include <poison_helpers.hpp>
#include <gallatin/pointers/poison.cuh>

using namespace gallatin::allocators;


template <typename T>
struct poisoned_array {

  using poison_type = gallatin::pointers::poison<T>;

  poison_type internal_array;

  // __device__ poisoned_array(){
  //   internal_array = nullptr;
  // }

  __device__ void init(uint64_t n_items){
    internal_array = poison_type::get_poisoned_reference(n_items);
  }

  __device__ void free(){
    internal_array.free();
  }

  __device__ T& operator[](std::size_t idx){

      return internal_array[idx];
  }



};


__global__ void init_poison_kernel(){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  auto poison = gallatin::pointers::poison<uint64_t>::get_poisoned_reference();

  poison.free();

}


__global__ void check_poison_kernel(){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  auto poison = gallatin::pointers::poison<uint64_t>::get_poisoned_reference();

  poison[0] = 16;

  poison[1] = 15;

  poison.free();

}

__global__ void check_poison_multi_kernel(){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  auto poison = gallatin::pointers::poison<uint64_t>::get_poisoned_reference(2);

  poison[0] = 16;

  poison[1] = 15;
  

  uint64_t read_value = poison[1];

  if (poison[0] != 16){
    write_global_log(4,0);
  }

  if (poison[1] != 15){
    write_global_log(4,1);
  }
  poison[2] = 0;

  poison.free();

}


__global__ void run_poisoned_array_kernel(){


  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  poisoned_array<uint64_t> array;

  array.init(64);

  for (uint i = 0; i < 64; i++){

    array[i] = i;

  }

  __threadfence();

  for (uint i = 0; i < 64; i++){

    if (array[i] != i){
      write_global_log(4,i);
    }
    

  }

  array.free();

}


__global__ void double_references_kernel(){


  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  auto array = gallatin::pointers::poison<poisoned_array<uint64_t>>::get_poisoned_reference(1);

  array->init(64);

  for (uint i = 0; i < 64; i++){

    array[0][i] = i;

  }

  __threadfence();

  for (uint i = 0; i < 64; i++){

    if (array[0][i] != i){
      write_global_log(4,i);
    }
    

  }

  array->free();
  array.free();

}



bool poison_helper_tests::testInit(){

  open_global_log();


  init_global_allocator(8ULL*1024*1024*1024, 42);

  GPUErrorCheck(cudaDeviceSynchronize());


  init_poison_kernel<<<1,1>>>();

  GPUErrorCheck(cudaDeviceSynchronize());


  free_global_allocator();

  
  return (close_global_log() == 0);


}


bool poison_helper_tests::testError(){

  open_global_log();


  init_global_allocator(8ULL*1024*1024*1024, 42);

  GPUErrorCheck(cudaDeviceSynchronize());


  check_poison_kernel<<<1,1>>>();

  GPUErrorCheck(cudaDeviceSynchronize());


  free_global_allocator();

  
  return (close_global_log() == 2);


}

bool poison_helper_tests::testErrorMulti(){

  open_global_log();


  init_global_allocator(8ULL*1024*1024*1024, 42);

  GPUErrorCheck(cudaDeviceSynchronize());


  check_poison_multi_kernel<<<1,1>>>();

  GPUErrorCheck(cudaDeviceSynchronize());


  free_global_allocator();

  
  return (close_global_log() == 2);


}


bool poison_helper_tests::testArray(){

  open_global_log();


  init_global_allocator(8ULL*1024*1024*1024, 42);

  GPUErrorCheck(cudaDeviceSynchronize());


  run_poisoned_array_kernel<<<1,1>>>();

  double_references_kernel<<<1,1>>>();

  GPUErrorCheck(cudaDeviceSynchronize());


  free_global_allocator();

  
  return (close_global_log() == 0);


}

void poison_helper_tests::open_global_log(){

  gallatin::internals::init_global_error_log();
}

int poison_helper_tests::close_global_log(){
  return gallatin::internals::close_global_error_log();
}




#endif




