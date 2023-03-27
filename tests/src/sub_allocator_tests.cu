/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */


#define DEBUG_ASSERTS 0

#define DEBUG_PRINTS 0



#include <poggers/allocators/sub_allocator.cuh>
#include <poggers/allocators/free_list.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

#define stack_bytes 32768




using allocator = poggers::allocators::sub_allocator_wrapper<stack_bytes, 4>::sub_allocator_type;
//using allocator = poggers::allocators::sub_allocator<stack_bytes, 7>;

using global_ptr = poggers::allocators::header;


__global__ void test_allocator_one_thread(global_ptr * heap){


   //this test size is illegal I think
   //but....
   //we have a dynamic memory manager
   //so lets just request from that lol
   const uint64_t test_size = 32000;


   uint ** addresses = (uint **) heap->malloc_aligned(test_size*sizeof(uint *), 16, 0);

   allocator * new_allocator = allocator::init(heap);

   for (int i = 0; i < test_size; i++){

      addresses[i] = (uint *) new_allocator->malloc(4, heap);

      addresses[i][0] = (uint) i;

   }

   for (uint i = 0; i < test_size; i++){
      if (addresses[i][0] != i){
         printf("%u failed\n", i);
      }
   }


   for (int i = 0; i < test_size; i++){

      if (addresses[i] != nullptr){
          new_allocator->stack_free(addresses[i]);
      }
     
   }

   heap->free(addresses);

   allocator::free_allocator(heap, new_allocator);


}

__global__ void test_allocator_many_threads(global_ptr * heap, allocator * new_allocator){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   const uint64_t test_size = 32000;

   if (tid >= test_size) return;


   uint * my_address = (uint *) new_allocator->malloc(4, heap);


   __syncthreads();

   if (my_address == nullptr){
      //printf("f");
      printf("Allocator failed?\n");
   } else {
      new_allocator->stack_free(my_address);
   }

  

   return;


}

//given the allocator is warmed up with a few runs, generate a dataset that will repeatedly malloc and free
//to force shifting.
__global__ void test_allocator_synthetic_workload(global_ptr * heap, allocator * new_allocator, uint64_t test_size, int num_rounds){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   //const uint64_t test_size = 32000;

   if (tid >= test_size) return;


   uint * my_addresses[2];

   //This is not the source of the stall
   //if we make it to the loops we always succeed
   //It's something about booting new managers

   for (int i = 0; i < num_rounds; i++){

      my_addresses[0] = nullptr;
      my_addresses[1] = nullptr;

      while (my_addresses[0] == nullptr){
         my_addresses[0] = (uint *) new_allocator->malloc(4, heap);
      }

      //printf("Progress on 0\n");

      while (my_addresses[1] == nullptr){
         my_addresses[1] = (uint *) new_allocator->malloc(4, heap);
      }

      new_allocator->stack_free(my_addresses[0]);

      my_addresses[0] = nullptr;

      while (my_addresses[0] == nullptr){
         my_addresses[0] = (uint *) new_allocator->malloc(4, heap);
      }

      new_allocator->stack_free(my_addresses[1]);
      new_allocator->stack_free(my_addresses[0]);

      //printf("Making progress\n");


   }



   return;


}

__global__ void test_allocator_variations(global_ptr * heap){


   using allocator_4 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 4>::sub_allocator_type;

   using allocator_8 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 8>::sub_allocator_type;

   using allocator_16 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 16>::sub_allocator_type;

   using allocator_32 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 32>::sub_allocator_type;

   using allocator_64 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 64>::sub_allocator_type;

   using allocator_128 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 128>::sub_allocator_type;

   using allocator_256 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 256>::sub_allocator_type;

   allocator_4 * new_allocator_4 = allocator_4::init(heap);
   allocator_4::free_allocator(heap, new_allocator_4);

   allocator_8 * new_allocator_8 = allocator_8::init(heap);
   allocator_8::free_allocator(heap, new_allocator_8);

   allocator_16 * new_allocator_16 = allocator_16::init(heap);
   allocator_16::free_allocator(heap, new_allocator_16);

   allocator_32 * new_allocator_32 = allocator_32::init(heap);
   allocator_32::free_allocator(heap, new_allocator_32);

   allocator_64 * new_allocator_64 = allocator_64::init(heap);
   allocator_64::free_allocator(heap, new_allocator_64);

   allocator_128 * new_allocator_128 = allocator_128::init(heap);
   allocator_128::free_allocator(heap, new_allocator_128);

   allocator_256 * new_allocator_256 = allocator_256::init(heap);
   allocator_256::free_allocator(heap, new_allocator_256);




}


//allocator code to get host handles

__global__ void allocate_stack(allocator ** stack_ptr, global_ptr * heap){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   allocator * my_stack = allocator::init(heap);

   stack_ptr[0] = my_stack;

   return;

}


__host__ allocator * host_allocate_allocator(global_ptr * heap){

   allocator ** stack_ptr;

   cudaMallocManaged((void **)&stack_ptr, sizeof(allocator *));

   allocate_stack<<<1,1>>>(stack_ptr, heap);

   cudaDeviceSynchronize();

   allocator * to_return = stack_ptr[0];

   cudaFree(stack_ptr);

   return to_return;



}

__global__ void dev_free_stack(global_ptr * heap, allocator * stack_to_free){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   allocator::free_allocator(heap, stack_to_free);

}

__host__ void host_free_allocator(global_ptr * heap, allocator * stack_to_free){


   dev_free_stack<<<1,1>>>(heap, stack_to_free);

   cudaDeviceSynchronize();

   return;


}

int main(int argc, char** argv) {


   //allocate 
   const uint64_t bytes_in_use = 800000;

   global_ptr * heap = global_ptr::init_heap(bytes_in_use);

   cudaDeviceSynchronize();

   test_allocator_one_thread<<<1,1>>>(heap);

   cudaDeviceSynchronize();

   test_allocator_variations<<<1,1>>>(heap);


   cudaDeviceSynchronize();

   printf("Starting multi threaded tests\n");

   allocator * my_allocator = host_allocate_allocator(heap);

   cudaDeviceSynchronize();

   test_allocator_many_threads<<<(32000-1)/1024+1,1024>>>(heap, my_allocator);

   cudaDeviceSynchronize();

   printf("Starting synthetic tests\n");

   uint64_t test_size = 32000;

   test_allocator_synthetic_workload<<<(test_size -1)/1024+1, 1024>>>(heap, my_allocator, test_size, 1);

   cudaDeviceSynchronize();

   host_free_allocator(heap, my_allocator);

   cudaDeviceSynchronize();

   global_ptr::free_heap(heap);


}
