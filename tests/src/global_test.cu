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


#if BETA_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 512
#endif


__global__ void alloc_one_size_pointer(uint64_t num_allocs, uint64_t size, uint64_t ** bitarray, uint64_t * misses){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;


   uint64_t * malloc = (uint64_t *) global_malloc(size);


   if (malloc == nullptr){
      atomicAdd((unsigned long long int *)misses, 1ULL);

      bitarray[tid] = malloc;
      return;
   }



   bitarray[tid] = malloc;

   malloc[0] = tid;

   __threadfence();


}


__global__ void free_one_size_pointer(uint64_t num_allocs, uint64_t size, uint64_t ** bitarray){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;


   uint64_t * malloc = bitarray[tid];

   if (malloc == nullptr) return;


   if (malloc[0] != tid){
      printf("Double malloc on index %lu: read address is %lu\n", tid, malloc[0]);
   }

   global_free(malloc);

   __threadfence();


}


//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
//works on actual pointers instead of uint64_t
//The correctness check is done by treating each allocation as a uint64_t and writing the tid
// if TID is not what is expected, we know that a double malloc has occurred.
__host__ void gallatin_test_allocs_pointer(uint64_t num_bytes, int num_rounds, uint64_t size){


   gallatin::utils::timer boot_timing;

   uint64_t mem_segment_size = 16ULL*1024*1024;

   uint64_t num_segments = gallatin::utils::get_max_chunks<16ULL*1024*1024>(num_bytes);

   uint64_t max_allocs_per_segment = mem_segment_size/16;

   uint64_t allocs_per_segment_size = mem_segment_size/size;

   if (allocs_per_segment_size >= max_allocs_per_segment) allocs_per_segment_size = max_allocs_per_segment;

   uint64_t num_allocs = allocs_per_segment_size*num_segments;

   printf("Starting test with %lu segments, %lu allocs per segment\n", num_segments, max_allocs_per_segment);
   printf("Actual allocs per segment %lu total allocs %lu\n", allocs_per_segment_size, num_allocs);

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


   uint64_t total_misses = 0;




   std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;

   for (int i = 0; i < num_rounds; i++){

      printf("Starting Round %d/%d\n", i, num_rounds);

      gallatin::utils::timer kernel_timing;
      alloc_one_size_pointer<<<(num_allocs-1)/TEST_BLOCK_SIZE+1,TEST_BLOCK_SIZE>>>(.9*num_allocs, size, bits, misses);
      kernel_timing.sync_end();

      gallatin::utils::timer free_timing;
      free_one_size_pointer<<<(num_allocs-1)/TEST_BLOCK_SIZE+1,TEST_BLOCK_SIZE>>>(.9*num_allocs, size, bits);
      free_timing.sync_end();

      kernel_timing.print_throughput("Malloced", .9*num_allocs);

      free_timing.print_throughput("Freed", .9*num_allocs);

      printf("Missed: %lu\n", misses[0]);

      cudaDeviceSynchronize();

      total_misses += misses[0];

      misses[0] = 0;

      print_global_stats();


   }

   printf("Total missed across %d runs: %lu\n", num_rounds, total_misses);

   print_global_stats();

   cudaFree(misses);

   cudaFree(bits);

   free_global_allocator();


}



int main(int argc, char** argv) {

   uint64_t num_segments;

   int num_rounds = 1;
   
   uint64_t size;

   if (argc < 2){
      num_segments = 100;
   } else {
      num_segments = std::stoull(argv[1]);
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

   gallatin_test_allocs_pointer(num_segments*16*1024*1024, num_rounds, size);



   cudaDeviceReset();
   return 0;

}
