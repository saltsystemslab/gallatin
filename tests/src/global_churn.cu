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

   uint64_t max_num_allocs = max_allocs_per_segment*num_segments;


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


   }

   printf("Total missed across %d runs: %lu\n", num_rounds, total_misses);

   print_global_stats();

   cudaFree(misses);

   cudaFree(bits);

   free_global_allocator();


}


//Catches the error with the tree ids.
__global__ void pointer_churn_kernel(uint64_t num_allocs, int num_rounds, uint64_t min_size, uint64_t max_size, uint64_t * misses){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;


   uint64_t hash = tid;

   gallatin::hashers::murmurHasher;


   //each loop, pick a random size and allocate from it.
   for (int i = 0; i < num_rounds; i++){

      hash = gallatin::hashers::MurmurHash64A(&hash, sizeof(uint64_t), i);

      uint64_t my_alloc_size = hash % max_size;

      if (my_alloc_size < min_size) my_alloc_size = min_size;

      __threadfence();

      uint64_t * allocation = (uint64_t *) global_malloc(my_alloc_size);

      uint64_t counter = 0;

      while (allocation == nullptr && counter < 10){

         __threadfence();

         allocation = (uint64_t *) global_malloc(my_alloc_size);

         counter+=1;
      }

      if (allocation == nullptr){
         atomicAdd((unsigned long long int *)misses, 1ULL);
         continue;
      }

      uint64_t old = atomicExch((unsigned long long int *)allocation, tid);

      if (old != 0ULL){
         printf("Double malloc: %lu and %lu share allocation\n", old, tid);
      }

      uint64_t current = atomicExch((unsigned long long int *)allocation, 0ULL);

      if (current != tid){
         printf("Double malloc on return: %lu and %lu share\n", current, tid);
      }


      global_free((void *) allocation);


   }

}

//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
__host__ void gallatin_pointer_churn(uint64_t num_bytes, uint64_t num_allocs, int num_rounds, uint64_t min_size, uint64_t max_size){


   gallatin::utils::timer boot_timing;

   const uint64_t mem_segment_size = 16ULL*1024*1024;

   uint64_t num_segments = gallatin::utils::get_max_chunks<mem_segment_size>(num_bytes);

   printf("Starting test with %lu segments, %lu threads per round for %d rounds in kernel\n", num_segments,  num_allocs, num_rounds);


   init_global_allocator(num_bytes, 111);


   //generate bitarrary - this covers the worst-case for alloc ranges.
   uint64_t num_bytes_bitarray = sizeof(uint64_t)*((num_allocs -1)/64+1);



   uint64_t * misses;
   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   cudaDeviceSynchronize();

   misses[0] = 0;




   std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;

   gallatin::utils::timer kernel_timing;
   pointer_churn_kernel<<<(num_allocs-1)/TEST_BLOCK_SIZE+1, TEST_BLOCK_SIZE>>>(num_allocs, num_rounds, max_size, min_size, misses);
   kernel_timing.sync_end();

   kernel_timing.print_throughput("Malloc/freed", num_allocs*num_rounds);
   printf("Missed: %llu/%llu: %f\n", misses[0], num_allocs*num_rounds, 1.0*(misses[0])/(num_allocs*num_rounds));


   print_global_stats();

   cudaFree(misses);



   free_global_allocator();

   cudaDeviceSynchronize();

}

int main(int argc, char** argv) {

   uint64_t num_segments;

   uint64_t num_threads;

   int num_rounds = 1;
   
   uint64_t max_size;

   uint64_t min_size;

   if (argc < 2){
      num_segments = 1000;
   } else {
      num_segments = std::stoull(argv[1]);
   }

   if (argc < 3){
      num_threads = 1000000;
   } else {
      num_threads = std::stoull(argv[2]);
   }

   if (argc < 4){
      num_rounds = 1;
   } else {
      num_rounds = std::stoull(argv[3]);
   }


   if (argc < 5){
      min_size = 16;
   } else {
      min_size = std::stoull(argv[4]);
   }

   if (argc < 6){
      max_size = 4096;
   } else {
      max_size = std::stoull(argv[5]);
   }




   gallatin_pointer_churn(num_segments*16*1024*1024, num_threads, num_rounds, min_size, max_size);


   cudaDeviceReset();
   return 0;

}
