/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */





#include <gallatin/allocators/gallatin.cuh>

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



//Catches the error with the tree ids.
template <typename allocator>
__global__ void pointer_churn_kernel(allocator * gallatin, uint64_t num_allocs, int num_rounds, uint64_t min_size, uint64_t max_size, uint64_t * misses){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;


   uint64_t hash = tid;


   //each loop, pick a random size and allocate from it.
   for (int i = 0; i < num_rounds; i++){

      hash = gallatin::hashers::MurmurHash64A(&hash, sizeof(uint64_t), i);

      uint64_t my_alloc_size = hash % max_size;

      if (my_alloc_size < min_size) my_alloc_size = min_size;

      __threadfence();

      uint64_t * allocation = (uint64_t *) gallatin->malloc(my_alloc_size);

      uint64_t counter = 0;

      while (allocation == nullptr && counter < 10){

         __threadfence();

         allocation = (uint64_t *) gallatin->malloc(my_alloc_size);

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


     gallatin->free((void *) allocation);


   }

}

//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
__host__ void gallatin_pointer_churn(uint64_t num_bytes, uint64_t num_allocs, int num_rounds, uint64_t min_size, uint64_t max_size){


   gallatin::utils::timer boot_timing;

   const uint64_t mem_segment_size = 16ULL*1024*1024;

   uint64_t num_segments = gallatin::utils::get_max_chunks<mem_segment_size>(num_bytes);

   printf("Starting test with %lu segments, %lu threads per round for %d rounds in kernel\n", num_segments,  num_allocs, num_rounds);


   size_t free, total;
   cudaMemGetInfo( &free, &total );

   uint64_t space_needed = num_segments*16ULL*1024*1024 + sizeof(uint64_t *)*num_allocs;

   if (space_needed >= free){

      printf("Test requires %llu bytes of space, only %llu free on device\n", space_needed, free);
      throw std::invalid_argument("Not enough space on GPU");


   }

   using gallatin_type = gallatin::allocators::Gallatin<16ULL*1024*1024, 16ULL, 4096ULL>;

   //init_global_allocator(num_bytes, 111);

   gallatin_type * allocator = gallatin_type::generate_on_device(num_bytes, 111);

   //generate bitarrary - this covers the worst-case for alloc ranges.
   uint64_t num_bytes_bitarray = sizeof(uint64_t)*((num_allocs -1)/64+1);



   uint64_t * misses;
   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   cudaDeviceSynchronize();

   misses[0] = 0;


   std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;

   gallatin::utils::timer kernel_timing;
   pointer_churn_kernel<gallatin_type><<<(num_allocs-1)/TEST_BLOCK_SIZE+1, TEST_BLOCK_SIZE>>>(allocator, num_allocs, num_rounds, max_size, min_size, misses);
   kernel_timing.sync_end();

   kernel_timing.print_throughput("Malloc/freed", num_allocs*num_rounds);
   printf("Missed: %llu/%llu: %f\n", misses[0], num_allocs*num_rounds, 1.0*(misses[0])/(num_allocs*num_rounds));

   allocator->print_info();


   cudaFree(misses);


   gallatin_type::free_on_device(allocator);


   //free_global_allocator();

   cudaDeviceSynchronize();

}

int main(int argc, char** argv) {

   uint64_t num_segments;

   uint64_t num_threads;

   int num_rounds = 1;
   
   uint64_t max_size;

   uint64_t min_size;

   // if (argc < 2){
   //    num_segments = 1000;
   // } else {
   //    num_segments = std::stoull(argv[1]);
   // }

   // if (argc < 3){
   //    num_threads = 1000000;
   // } else {
   //    num_threads = std::stoull(argv[2]);
   // }

   // if (argc < 4){
   //    num_rounds = 1;
   // } else {
   //    num_rounds = std::stoull(argv[3]);
   // }


   // if (argc < 5){
   //    min_size = 16;
   // } else {
   //    min_size = std::stoull(argv[4]);
   // }

   // if (argc < 6){
   //    max_size = 4096;
   // } else {
   //    max_size = std::stoull(argv[5]);
   // }


   if (argc < 6){
      printf("Test pulls allocaitons of a random size between min-max size in a loop, checks for double malloc\n");
      printf("Usage: ./tests/gallatin_churn [num_segments] [num_threads] [num_rounds] [min allocation size] [max allocation size]\n");
      return 0;
   }

   num_segments = std::stoull(argv[1]);
   num_threads = std::stoull(argv[2]);
   num_rounds = std::stoull(argv[3]);
   min_size = std::stoull(argv[4]);
   max_size = std::stoull(argv[5]);

   gallatin_pointer_churn(num_segments*16*1024*1024, num_threads, num_rounds, min_size, max_size);


   cudaDeviceReset();
   return 0;

}
