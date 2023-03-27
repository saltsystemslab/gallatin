/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/free_list.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using global_ptr = poggers::allocators::header;



__global__ void single_thread_malloc_and_free_tests(global_ptr * head){

   const uint64_t test_size = 10;

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   printf("Starting\n");

   uint64_t * array_list [test_size];

   for (int i = 0; i < test_size; i++){

      array_list[i] = (uint64_t *) head->malloc_safe(sizeof(uint64_t)*20);

   }


   printf("%llu Malloc done\n\n", tid);

   global_ptr::print_heap(head);


   printf("And allocated nodes:\n");

   for (int i=0; i< test_size; i++){

      if (array_list[i] != nullptr){
          global_ptr * node = global_ptr::get_header_from_address(array_list[i]);
          node->printnode();
          printf("Printed Node\n");
      }
     

   }


   for (int i = 0; i < test_size; i++){

      if (array_list[i] != nullptr){
         head->free_safe(array_list[i]);
      }

      printf("%d:\n", i);
      global_ptr::print_heap(head);
     
   }


   global_ptr::print_heap(head);

   return;

}

__global__ void test_aligned_alloc_and_free(global_ptr * head){

   const uint64_t test_size = 10;

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   printf("Starting\n");

   uint64_t * array_list [test_size];

   for (int i = 0; i < test_size; i++){

      array_list[i] = (uint64_t *) head->malloc_aligned(sizeof(uint64_t)*20, 16, 0);

   }


   printf("%llu Malloc done\n\n", tid);

   global_ptr::print_heap(head);


   printf("And allocated nodes:\n");

   for (int i=0; i< test_size; i++){

      if (array_list[i] != nullptr){
          global_ptr * node = global_ptr::get_header_from_address(array_list[i]);
          node->printnode();
          printf("Printed Node\n");
      }
     

   }


   for (int i = 0; i < test_size; i++){

      if (array_list[i] != nullptr){
         head->free_safe(array_list[i]);
      }

      printf("%d:\n", i);
      global_ptr::print_heap(head);
     
   }


   global_ptr::print_heap(head);

   printf("End of case 1/2\n");


   for (int i = 0; i < test_size; i++){

      array_list[i] = (uint64_t *) head->malloc_aligned(sizeof(uint64_t), 512, -16);

   }

   printf("Done with large alignment alloc\n");

      global_ptr::print_heap(head);


   printf("And allocated nodes:\n");

   for (int i=0; i< test_size; i++){

      if (array_list[i] != nullptr){
          global_ptr * node = global_ptr::get_header_from_address(array_list[i]);
          node->printnode();
          //printf("Printed Node\n");
      }
     

   }


   printf("End of print\n");

   for (int i = 0; i < test_size; i++){

      if (array_list[i] != nullptr){
         head->free_safe(array_list[i]);
      }

      printf("%d:\n", i);
      global_ptr::print_heap(head);
     
   }

   global_ptr::print_heap(head);

   return;

}


__global__ void multi_thread_malloc_and_free(global_ptr * head, uint64_t num_threads, uint64_t ** nodes){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_threads) return;

   //grab 200 bytes at a time!
   uint64_t * test = nullptr;

   while (test == nullptr){

      test = (uint64_t *) head->malloc_safe(32);

      //printf("Stalling in malloc loop\n");

   }

   //test = (uint64_t *) head->malloc(8);



   if (test != nullptr){ 

      head->free_safe(test);
   }

   //nodes[tid] = test;


   // test[0] = 512;



   // printf("%llu Malloc done, written as %llu\n\n", tid, test[0]);

   // head->free(test);

   printf("%llu Free done\n\n", tid);


}


__global__ void print_heap_kernel(global_ptr * head, uint64_t num_threads, uint64_t ** nodes){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0 ) return;

   global_ptr::print_heap(head);

   //printf("Allocated Nodes\n");

   // for (int i=0; i< num_threads; i++){

   //    if (nodes[i] != nullptr){
   //           global_ptr * node = global_ptr::get_header_from_address(nodes[i]);
   //           node->printnode();
   //       }

   // }


}

__host__ void print_heap(global_ptr * head, uint64_t num_threads, uint64_t ** nodes){

   print_heap_kernel<<<1,1>>>(head, num_threads, nodes);

}


__host__ void test_mallocs(global_ptr * head, uint64_t num_mallocs){

   printf("Starting test with %llu threads\n", num_mallocs);

   uint64_t ** nodes;

   cudaMalloc((void **)&nodes, num_mallocs*sizeof(uint64_t));

   multi_thread_malloc_and_free<<<(num_mallocs -1)/512 + 1, 512>>>(head, num_mallocs, nodes);

   cudaDeviceSynchronize();

   print_heap(head, num_mallocs, nodes);

   cudaDeviceSynchronize();

   cudaFree(nodes);

}

int main(int argc, char** argv) {






   uint64_t bytes_in_use = 2000;

  

   global_ptr * heap = global_ptr::init_heap(bytes_in_use);

   cudaDeviceSynchronize();

   printf("Heap init Done\n");

   cudaDeviceSynchronize();
   single_thread_malloc_and_free_tests<<<1,1>>>(heap);

   cudaDeviceSynchronize();


   printf("Starting Malloc tests\n\n\n");

   test_mallocs(heap, 1);

   test_mallocs(heap, 10);

   test_mallocs(heap, 30);

   test_mallocs(heap, 50);

   test_mallocs(heap, 60);


   test_mallocs(heap, 100);

   test_mallocs(heap, 1000);

   //print_heap(heap);

   cudaDeviceSynchronize();

   global_ptr::free_heap(heap);

   cudaDeviceSynchronize();




   printf("Starting alignment tests\n");

   heap = global_ptr::init_heap(bytes_in_use);

   cudaDeviceSynchronize();

   test_aligned_alloc_and_free<<<1,1>>>(heap);

   cudaDeviceSynchronize();

   global_ptr::free_heap(heap);



}
