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


template<typename allocator>
__host__ void benchmark_allocator(uint64_t num_bytes, uint64_t max_mallocs, uint64_t bytes_to_malloc){

   const uint64_t block_size = 512;

   printf("Running test on ");
   allocator::print_info();

   allocator * my_allocator = allocator::init(num_bytes);

   cudaDeviceSynchronize();

   printf("Starting\n");

   auto cms_start = std::chrono::high_resolution_clock::now();

   malloc_benchmark<allocator><<<(max_mallocs - 1)/block_size +1, block_size>>>(my_allocator, max_mallocs, bytes_to_malloc);

   cudaDeviceSynchronize();

   auto cms_end = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> cms_diff = cms_end-cms_start;

   std::cout << "cms Malloced " << max_mallocs << " in " << cms_diff.count() << " seconds\n";

   my_allocator->host_report();

   allocator::free_cms_allocator(my_allocator);

   cudaDeviceSynchronize();

   printf("\n\n");


}

int main(int argc, char** argv) {


   //allocate 
   //const uint64_t meg = 1024*1024;

    using shibboleth_1meg = poggers::allocators::shibboleth<MEGABYTE, 150, 8>;

   benchmark_allocator<shibboleth_1meg>(4ULL*GIGABYTE, 100000000ULL, 16);

   using shibboleth_2meg = poggers::allocators::shibboleth<MEGABYTE*2, 150, 8>;

   benchmark_allocator<shibboleth_2meg>(4ULL*GIGABYTE, 100000000ULL, 16);

   using shibboleth = poggers::allocators::shibboleth<MEGABYTE*4, 150, 8>;

   benchmark_allocator<shibboleth>(4ULL*GIGABYTE, 100000000ULL, 16);

   using shibboleth_64 = poggers::allocators::shibboleth<MEGABYTE*4, 64, 8>;

   benchmark_allocator<shibboleth_64>(4ULL*GIGABYTE, 100000000ULL, 16); 


   using shibboleth_64_half = poggers::allocators::shibboleth<512ULL*1024, 64, 8>;

   benchmark_allocator<shibboleth_64_half>(4ULL*GIGABYTE, 100000000ULL, 16); 


   using shibboleth_256 = poggers::allocators::shibboleth<512ULL*1024, 256, 8>;

   benchmark_allocator<shibboleth_256>(4ULL*GIGABYTE, 100000000ULL, 16); 




   return 0;




}
