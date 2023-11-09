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


__global__ void insert_one_size(uint64_t num_inserts, uint64_t size, uint64_t ** bitarray, uint64_t * misses){


   //uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= num_inserts) return;


   uint64_t * malloc = (uint64_t *) global_malloc(size);

   if (malloc == nullptr){
      atomicAdd((unsigned long long int *)misses, 1ULL);

      bitarray[tid] = malloc;
      return;
   }


   uint64_t old = atomicExch((unsigned long long int *)&bitarray[tid], (unsigned long long int) malloc);

   // if (old != 0){
   //    printf("Two threads swapping to same addr\n");
   // }

   //bitarray[tid] = malloc;

   malloc[0] = tid;

   __threadfence();

   // if (bitarray[tid][0] != tid){
   //    printf("Err detected\n");
   // }

}

__device__ void write_non_poison(char * ptr, uint64_t n_bytes){

   for (uint64_t i = 0; i < n_bytes; i++){
      //write some junk.
      ptr[i] = (char) ((i*i+1)%256);
   }

} 


__device__ char * malloc_and_write(uint64_t size){

   //allocate with poison
   char * alloc = (char *) global_malloc_poison(size);

   if (alloc == nullptr) printf("Failed to malloc size %llu\n", size);

   write_non_poison(alloc, size);


   return alloc;


}

__global__ void test_poison_kernel(uint64_t size){


   //uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   uint64_t tid = gallatin::utils::get_tid();

   if (tid != 0) return;

   //allocate with poison
   char * alloc = malloc_and_write(size);

   //should pass
   global_free_poison(alloc);

   printf("Should be no issues so far.\n");



   alloc = malloc_and_write(size);

   uint64_t * alt_ptr = (uint64_t *) alloc;

   alt_ptr[-2] = 17ULL;

   global_free_poison(alloc);

   printf("Should be poison trigger 1\n");

   alloc = malloc_and_write(size);

   alt_ptr = (uint64_t *) alloc;

   alt_ptr[-1] = 17ULL;

   global_free_poison(alloc);

   printf("Should be poison trigger 2\n");

   alloc = malloc_and_write(size);

   alloc[size] = 'c';

   global_free_poison(alloc);

   printf("Should be poison trigger 3\n");



   alloc = malloc_and_write(size);

   alt_ptr = (uint64_t *) alloc;

   alt_ptr[-1] = 17ULL;

   alt_ptr[-2] = 17ULL;

   global_free_poison(alloc);

   //triggers 1 as write offset is incorrect
   printf("Should be poison trigger 1\n");


   alloc = malloc_and_write(size);

   alt_ptr = (uint64_t *) alloc;

   alt_ptr[-1] = 48ULL;

   alt_ptr[-2] = 48ULL;

   global_free_poison(alloc);

   //triggers 1 as write offset is incorrect
   printf("Should be poison trigger 3\n");


   alloc = malloc_and_write(size);

   alt_ptr = (uint64_t *) alloc;

   alt_ptr[-1] = 48ULL;

   alt_ptr[-2] = 48ULL;

   global_free_poison(alloc);

   //triggers 1 as write offset is incorrect
   printf("Should be poison trigger 3\n");





   __threadfence();


}


//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
//works on actual pointers instead of uint64_t
//The correctness check is done by treating each allocation as a uint64_t and writing the tid
// if TID is not what is expected, we know that a double malloc has occurred.
__host__ void gallatin_test_allocs_poison(uint64_t num_bytes, uint64_t size){


   init_global_allocator(num_bytes, 41);

   test_poison_kernel<<<1,1>>>(size);

   free_global_allocator();

   cudaDeviceSynchronize();


}


int main(int argc, char** argv) {

   //not necessary to determine
   uint64_t num_segments = 1000;


   for (uint64_t size = 4; size <= 131072; size*=2){

      std::cout << "testing poison on size " << size << std::endl; 

      gallatin_test_allocs_poison(num_segments*16*1024*1024, size);

   }

   



   cudaDeviceReset();
   return 0;

}
