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

#define SHOW_PROGRESS 0

#define COUNTING_CYCLES 1

#include <poggers/allocators/cms.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

//#define stack_bytes 262144




//#define stack_bytes 4194304

#define MEGABYTE 1024*1024

#define GIGABYTE 1024*MEGABYTE

#define stack_bytes 4*MEGABYTE



template <typename allocator>
__global__ void malloc_benchmark(allocator * cms, uint64_t max_mallocs, uint64_t bytes_to_malloc){

   uint64_t items_per_thread = 1;


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= max_mallocs) return;

   #if SHOW_PROGRESS
   if (tid % 100000 == 0){
      printf("%llu\n", tid);
   }
   #endif


   for (uint64_t i = 0; i < items_per_thread; i++){

      cms->cms_malloc(bytes_to_malloc);

   }




}


template <typename allocator>
__global__ void malloc_free_benchmark(allocator * cms, uint64_t max_mallocs, uint64_t bytes_to_malloc){

   const uint64_t items_per_thread = 1;

   void * storage_array[items_per_thread];

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= max_mallocs) return;

   #if SHOW_PROGRESS
   if (tid % 100000 == 0){
      printf("%llu\n", tid);
   }
   #endif


   for (uint64_t i = 0; i < items_per_thread; i++){

      storage_array[i] = cms->cms_malloc(bytes_to_malloc);

   }


   for (uint64_t i = 0; i < items_per_thread; i++){

      cms->cms_free(storage_array[i]);

   }


}


template<typename allocator>
__host__ void benchmark_allocator(uint64_t num_bytes, uint64_t max_mallocs, uint64_t bytes_to_malloc){

   const uint64_t block_size = 512;

   printf("Running test on ");
   allocator::print_info();

   allocator * my_allocator = allocator::init(num_bytes);

   cudaDeviceSynchronize();

   auto cms_start = std::chrono::high_resolution_clock::now();

   malloc_benchmark<allocator><<<(max_mallocs - 1)/block_size +1, block_size>>>(my_allocator, max_mallocs, bytes_to_malloc);

   cudaDeviceSynchronize();

   auto cms_end = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> cms_diff = cms_end-cms_start;

   std::cout << "cms Malloced " << max_mallocs << " in " << cms_diff.count() << " seconds\n";

   printf("%f allocs per second\n", ((double) max_mallocs)/ cms_diff.count());

   my_allocator->host_report();

   allocator::free_cms_allocator(my_allocator);

   cudaDeviceSynchronize();

   printf("\n\n");


}

int main(int argc, char** argv) {


   //allocate 
   //const uint64_t meg = 1024*1024;

   using shibboleth = poggers::allocators::shibboleth<MEGABYTE*4, 150, 5>;

   printf("10,000,000\n");
   benchmark_allocator<shibboleth>(8ULL*GIGABYTE, 10000000ULL, 16);

   printf("25,000,000\n");
   benchmark_allocator<shibboleth>(8ULL*GIGABYTE, 25000000ULL, 16);


   printf("50,000,000\n");
   benchmark_allocator<shibboleth>(8ULL*GIGABYTE, 50000000ULL, 16);

   printf("100,000,000\n");
   benchmark_allocator<shibboleth>(8ULL*GIGABYTE, 100000000ULL, 16);

   printf("4 byte allocs\n");
   printf("10,000,000\n");
   benchmark_allocator<shibboleth>(8ULL*GIGABYTE, 10000000ULL, 4);

   printf("25,000,000\n");
   benchmark_allocator<shibboleth>(8ULL*GIGABYTE, 25000000ULL, 4);


   printf("50,000,000\n");
   benchmark_allocator<shibboleth>(8ULL*GIGABYTE, 50000000ULL, 4);

   printf("100,000,000\n");
   benchmark_allocator<shibboleth>(8ULL*GIGABYTE, 100000000ULL, 4);



   return 0;




}
