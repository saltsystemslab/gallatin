/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/free_list.cuh>
#include <poggers/allocators/aligned_stack.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

#define stack_bytes 32768

using global_ptr = poggers::allocators::header;

using stack = poggers::allocators::aligned_manager<stack_bytes, false>;

__global__ void test_manager_one_thread(global_ptr * heap){


   const uint64_t test_size = 1000;

   uint * addresses[test_size];

   stack * my_stack = stack::init_from_free_list(heap, 4);

   for (int i = 0; i < test_size; i++){

      addresses[i] = (uint *) my_stack->malloc();

   }


   for (int i = 0; i < test_size; i++){

      if (addresses[i] != nullptr){
          my_stack->free(addresses[i]);
      }
     
   }


   stack::free_stack(heap, my_stack);

}


__global__ void test_multi_thread(stack * my_stack){

   //uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   const uint64_t test_size = 20;

   uint * addresses[test_size];

   //stack * my_stack = stack::init_from_free_list(heap, 4);

   for (int i = 0; i < test_size; i++){

      addresses[i] = nullptr;

      while (addresses[i] == nullptr){

         addresses[i] = (uint *) my_stack->malloc();

      }

      

   }


   for (int i = 0; i < test_size; i++){

      if (addresses == nullptr){
         printf("Something wrong, address set to nullptr\n");
      }
      else {
         my_stack->free(addresses[i]);
      }
     
   }


   //stack::free_stack(heap, my_stack);

}


__global__ void test_allocated_stack(stack * my_stack){


   const uint64_t maximum_threads = (stack_bytes - sizeof(stack))/4;

   uint * addresses[maximum_threads];


  // stack * my_stack = stack::init_from_free_list(heap, 4);

   for (int i = 0; i < maximum_threads; i++){
      addresses[i] = (uint *) my_stack->malloc();

      if (addresses[i] == nullptr){
         printf("Failure on %d\n", i);
      }
   }

   if (my_stack->malloc() != nullptr){
      printf("Not all allocated. This is a bug\n");
   }

   for (int i = 0; i < maximum_threads; i++){

      addresses[i][0] = i;

   }

   for (int i = 0; i < maximum_threads; i++){

      my_stack->free(addresses[i]);

   }

   //stack::free_stack(heap, my_stack);

}


__global__ void test_manager_maximum(global_ptr * heap){

   const uint64_t maximum_threads = (stack_bytes - sizeof(stack))/4;

   uint * addresses[maximum_threads];


   stack * my_stack = stack::init_from_free_list(heap, 4);

   for (int i = 0; i < maximum_threads; i++){
      addresses[i] = (uint *) my_stack->malloc();

      if (addresses[i] == nullptr){
         printf("Failure on %d\n", i);
      }
   }

   if (my_stack->malloc() != nullptr){
      printf("Not all allocated. This is a bug\n");
   }

   for (int i = 0; i < maximum_threads; i++){

      addresses[i][0] = i;

   }

   for (int i = 0; i < maximum_threads; i++){

      my_stack->free(addresses[i]);

   }

   stack::free_stack(heap, my_stack);


}


// __global__ void test_host_initialized_stack


__global__ void allocate_stack(stack ** stack_ptr, global_ptr * heap){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   stack * my_stack = stack::init_from_free_list(heap, 4);

   stack_ptr[0] = my_stack;

   return;

}


__host__ stack * host_allocate_stack(global_ptr * heap){

   stack ** stack_ptr;

   cudaMallocManaged((void **)&stack_ptr, sizeof(uint64_t *));

   allocate_stack<<<1,1>>>(stack_ptr, heap);

   cudaDeviceSynchronize();

   stack * to_return = stack_ptr[0];

   cudaFree(stack_ptr);

   return to_return;



}

__global__ void dev_free_stack(global_ptr * heap, stack * stack_to_free){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   stack::free_stack(heap, stack_to_free);

}

__host__ void host_free_stack(global_ptr * heap, stack * stack_to_free){


   dev_free_stack<<<1,1>>>(heap, stack_to_free);

   cudaDeviceSynchronize();

   return;


}

int main(int argc, char** argv) {






   uint64_t bytes_in_use = 800000;

  

   global_ptr * heap = global_ptr::init_heap(bytes_in_use);

   cudaDeviceSynchronize();

   test_manager_one_thread<<<1,1>>>(heap);

   cudaDeviceSynchronize();

   for (int i = 0; i< 10; i++){

      test_manager_maximum<<<1,1>>>(heap);

      cudaDeviceSynchronize();

      printf("Done with single-threaded test %d\n", i);

   }


   stack * dev_stack = host_allocate_stack(heap);

   cudaDeviceSynchronize();


   for (int i = 0; i < 10; i++){

      test_allocated_stack<<<1,1>>>(dev_stack);
      cudaDeviceSynchronize();

      printf("Done with host-stack test %d\n", i);

   }

   printf("Testing multi threaded!\n");


   test_multi_thread<<<1,1>>>(dev_stack);

   cudaDeviceSynchronize();

   printf("Done with 1\n");

   test_multi_thread<<<1, 10>>>(dev_stack);

   printf("Done with 10\n");

   test_multi_thread<<<1, 100>>>(dev_stack);

   cudaDeviceSynchronize();

   printf("Done with 100\n");

   test_multi_thread<<<10, 40>>>(dev_stack);

   cudaDeviceSynchronize();

   printf("Done with 400\n");

   cudaDeviceSynchronize();

   host_free_stack(heap, dev_stack);



   global_ptr::free_heap(heap);

   cudaDeviceSynchronize();



}
