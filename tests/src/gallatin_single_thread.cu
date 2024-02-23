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
   #define TEST_BLOCK_SIZE 512
#endif



__global__ void alloc_one_size_pointer(uint64_t num_allocs, uint64_t size, uint64_t ** bitarray, uint64_t * misses){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid !=0) return;


   for (uint64_t i=0; i < num_allocs; i++){

      uint64_t * malloc = (uint64_t *) global_malloc(size);


      if (malloc == nullptr){
         atomicAdd((unsigned long long int *)misses, 1ULL);
      }

      bitarray[i] = malloc;

      malloc[0] = i;

      __threadfence();


   }

}


__global__ void free_one_size_pointer(uint64_t num_allocs, uint64_t size, uint64_t ** bitarray){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;


   for (uint64_t i=0; i < num_allocs; i++){

      uint64_t * malloc = bitarray[i];

      if (malloc == nullptr) continue;


      if (malloc[0] != i){
         printf("Double malloc on index %llu: read address is %llu\n", i, malloc[0]);
      }

      global_free(malloc);

      __threadfence();

   } 


}


//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
//works on actual pointers instead of uint64_t
//The correctness check is done by treating each allocation as a uint64_t and writing the tid
// if TID is not what is expected, we know that a double malloc has occurred.
__host__ void gallatin_test_single_thread(uint64_t num_allocs, uint64_t num_rounds, uint64_t size){


   uint64_t num_bytes = 16ULL*1024*1024*1000;

   gallatin::utils::timer boot_timing;

   //uint64_t num_segments = gallatin::utils::get_max_chunks<mem_segment_size>(num_bytes);

   //betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

   init_global_allocator(num_bytes, 42);

   //generate bitarry
   //space reserved is one 
   uint64_t ** bits;
   cudaMalloc((void **)&bits, sizeof(uint64_t *)*num_allocs);

   cudaMemset(bits, 0, sizeof(uint64_t *)*num_allocs);


   uint64_t * misses;
   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   cudaDeviceSynchronize();

   misses[0] = 0;




   std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;

   for (int i = 0; i < num_rounds; i++){

      printf("Starting Round %d/%d\n", i, num_rounds);

      gallatin::utils::timer kernel_timing;
      alloc_one_size_pointer<<<1,1>>>(num_allocs, size, bits, misses);
      kernel_timing.sync_end();

      gallatin::utils::timer free_timing;
      free_one_size_pointer<<<1,1>>>(num_allocs, size, bits);
      free_timing.sync_end();

      kernel_timing.print_throughput("Malloced", num_allocs);

      free_timing.print_throughput("Freed", num_allocs);

      printf("Missed: %llu\n", misses[0]);

      cudaDeviceSynchronize();

      misses[0] = 0;

      print_global_stats();

   }

   cudaFree(misses);

   cudaFree(bits);

   free_global_allocator();

}


int main(int argc, char** argv) {

   uint64_t num_allocs;

   int num_rounds;
   
   uint64_t size;

   if (argc < 2){
      num_allocs = 100;
   } else {
      num_allocs = std::stoull(argv[1]);
   }


   if (argc < 3){
      num_rounds = 1;
   } else {
      num_rounds = std::stoull(argv[2]);
   }


   if (argc < 4){
      size = 16;
   } else {
      size = std::stoull(argv[3]);
   }


   gallatin_test_single_thread(num_allocs, num_rounds, size);


   cudaDeviceReset();
   return 0;

}
