#ifndef CYCLE_COUNTING
#define CYCLE_COUNTING


#include <cuda.h>
#include <cuda_runtime_api.h>


#include "stdio.h"
#include "assert.h"


//A series of inclusions for building a poggers hash table

#define COMPRESS_VALUE 1024


//CMS cycle counting - in the tests set #COUNTING_CYCLES 1
//These global counters count total cycles over a cms init->free
//and are a global variable cause copying over all the functions would be ass


//cycles taken in overall kernel
__device__ uint64_t kernel_counter;

//cycles taken with sub_allocator
__device__ uint64_t sub_allocator_counter;

//cycles taken working with the free list
//this should be small
__device__ uint64_t free_list_counter;

//cycles taken in the stack object
__device__ uint64_t stack_counter;

__device__ uint64_t sub_allocator_regular_counter;


__device__ uint64_t kernel_traversals;
__device__ uint64_t sub_allocator_main_traversals;
__device__ uint64_t sub_allocator_alt_traversals;
__device__ uint64_t stack_traversals;
__device__ uint64_t heap_traversals;


__device__ uint64_t stack_init_counter;
__device__ uint64_t stack_init_traversals;


__global__ void reset_cycles_kernel(){

    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

    if (tid != 0) return;

    printf("Booting cycle counters.\n");
    kernel_counter = 0;
    sub_allocator_counter = 0;
    free_list_counter = 0;
    stack_counter = 0;
	sub_allocator_regular_counter = 0;


    kernel_traversals = 0;
	sub_allocator_main_traversals = 0;
	sub_allocator_alt_traversals = 0;
	stack_traversals = 0;
	heap_traversals = 0;

   stack_init_counter = 0;
   stack_init_traversals = 0;

}


__host__ void poggers_reset_cycles(){

	reset_cycles_kernel<<<1,1>>>();

}

__global__ void display_cycles_kernel(){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;
   printf("Cycle counters:\n");
   printf("Est. Cycles in kernel: %llu, %llu\n", kernel_counter, kernel_traversals);
   printf("Est. cycles in sub_allocator: %llu, %llu\n", sub_allocator_regular_counter, sub_allocator_main_traversals);
   printf("Est. cycles in sub_allocator getting new stack / shift: %llu, %llu\n", sub_allocator_counter, sub_allocator_alt_traversals);
   printf("Est. cyles in stacks: %llu, %llu\n", stack_counter, stack_traversals);
   printf("Est. cycles in heap: %llu, %llu\n", free_list_counter, heap_traversals);
   printf("Est. cycles booting stack: %llu, %llu\n", stack_init_counter, stack_init_traversals);

   double kernel_avg = ((double) kernel_counter )/ kernel_traversals;

   double sub_main_avg = ((double) sub_allocator_regular_counter )/ sub_allocator_main_traversals;

   double sub_alt_avg = ((double) sub_allocator_counter )/ sub_allocator_alt_traversals;
   
   double stack_avg = ((double) stack_counter )/ stack_traversals;

   double heap_avg = ((double) free_list_counter )/ heap_traversals;

   double stack_init_avg = ((double) stack_init_counter) / stack_init_traversals;

   printf("Kernel average %f\n", kernel_avg );
  	
   printf("Sub main avg %f\n", sub_main_avg);
   printf("sub alt avg %f\n", sub_alt_avg);

   printf("Stack average %f\n", stack_avg);
   printf("heap avg %f\n", heap_avg);

   printf("Stack boot avg %f\n", stack_init_avg);

}


__host__ void poggers_display_cycles(){

	display_cycles_kernel<<<1,1>>>();

}


#endif //GPU_BLOCK_