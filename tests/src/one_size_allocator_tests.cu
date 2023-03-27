/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/one_size_allocator.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>


#define NUM_ALLOCS  3125000

using namespace poggers::allocators;


__global__ void malloc_free_kernel(one_size_allocator * allocator ,uint64_t num_threads, int num_rounds, uint64_t * misses){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_threads) return;

   for (int i = 0; i < num_rounds; i++){

      void * allocation = allocator->malloc();

      //uint64_t allocation = allocator->get_offset();

      //if (allocation == )


      if (allocation == nullptr){

      //if (allocation == veb_tree::fail()){

         atomicAdd((unsigned long long int *) misses, 1ULL);

      } else {

         // uint64_t * test_alloc = (uint64_t * ) allocation;

         // test_alloc[0] = 35;

         allocator->free(allocation);

      }

   }


}




__host__ void time_allocation_cyclic(uint64_t num_allocs, uint64_t size_per_alloc, uint64_t num_threads, int num_rounds){


   one_size_allocator * allocator = one_size_allocator::generate_on_device(num_allocs, size_per_alloc, 23);

   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   cudaDeviceSynchronize();

   misses[0] = 0;

   cudaDeviceSynchronize();

   auto insert_start = std::chrono::high_resolution_clock::now();

   malloc_free_kernel<<<(num_threads-1)/512+1, 512>>>(allocator, num_threads, num_rounds, misses);

   cudaDeviceSynchronize();

   auto insert_end = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> elapsed_seconds = insert_end - insert_start;


   std::cout << "Inserted " <<  num_threads*num_rounds << " in " << elapsed_seconds.count() << " seconds, throughput: " << std::fixed << 1.0*num_threads*num_rounds/elapsed_seconds.count() << std::endl;
  
   std::cout << "Misses: " << misses[0] << "/" << num_threads*num_rounds << "\n";

   cudaFree(misses);

   one_size_allocator::free_on_device(allocator);

}


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   time_allocation_cyclic(31250000ULL, 128, 100000ULL, 10);

   time_allocation_cyclic(31250000ULL, 128, 1000000ULL, 10);

   time_allocation_cyclic(31250000ULL, 128, 10000000ULL, 10);

   time_allocation_cyclic(31250000ULL, 128, 20000000ULL, 10);

   time_allocation_cyclic(31250000ULL, 128, 30000000ULL, 10);

   time_allocation_cyclic(31250000ULL, 128, 10000000ULL, 100);

   // printf("Bigger?\n");


   // time_allocation_cyclic(1000000000ULL, 16, 100000ULL, 10);

   // time_allocation_cyclic(1000000000ULL, 16, 1000000ULL, 10);

   // time_allocation_cyclic(1000000000ULL, 16, 10000000ULL, 10);

   // time_allocation_cyclic(1000000000ULL, 16, 999000000ULL, 10);

 
   cudaDeviceReset();
   return 0;

}
