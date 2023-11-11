/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */





#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace gallatin::allocators;


#if GALLATIN_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 256
#endif


__global__ void malloc_multiple(uint64_t num_allocs){


   //uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= num_allocs) return;


   uint64_t * malloc = (uint64_t *) global_malloc(16);

   if (malloc == nullptr){

      asm volatile("trap;");
      return;
   }
   
   global_free(malloc);

}



//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
//works on actual pointers instead of uint64_t
//The correctness check is done by treating each allocation as a uint64_t and writing the tid
// if TID is not what is expected, we know that a double malloc has occurred.
__host__ void gallatin_test_allocs_pointer(uint64_t num_bytes, uint64_t num_allocs){

   init_global_allocator(num_bytes, 42, true, true);

   malloc_multiple<<<(num_allocs-1)/TEST_BLOCK_SIZE+1,TEST_BLOCK_SIZE>>>(num_allocs);

   cudaDeviceSynchronize();

   printf("Done\n");
  
   free_global_allocator();

   cudaDeviceSynchronize();


}


int main(int argc, char** argv) {

   uint64_t num_segments;

   uint64_t num_allocs;
   
   uint64_t size;

   if (argc < 2){
      num_segments = 100;
   } else {
      num_segments = std::stoull(argv[1]);
   }

   if (argc < 3){
      num_allocs = 1000;
   } else {
      num_allocs = std::stoull(argv[2]);
   }

   gallatin_test_allocs_pointer(num_segments*16*1024*1024, num_allocs);



   cudaDeviceReset();
   return 0;

}
