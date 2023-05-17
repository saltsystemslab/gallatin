/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/ext_veb_nosize.cuh>
#include <poggers/allocators/alloc_memory_table.cuh>
#include <poggers/allocators/betta.cuh>
#include <bits/stdc++.h>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>


using namespace std::chrono;


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

double elapsed(high_resolution_clock::time_point t1, high_resolution_clock::time_point t2) {
   return (duration_cast<duration<double> >(t2 - t1)).count();
}

using namespace poggers::allocators;

__global__ void assert_unique_mallocs(uint64_t * allocs, uint64_t num_allocs){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;

   //uint64_t my_block = (uint64_t) blocks[tid];

   uint64_t alloc = allocs[tid];


   for (uint64_t i=tid+1; i < num_allocs; i++){

      uint64_t ext_alloc = (uint64_t) allocs[i];

      if (alloc == ext_alloc){
         printf("Collision on %llu and %llu: %llu. Block %llu\n",tid, i, alloc, alloc/4096);
      }

   }


}



template <typename betta_type>
__global__ void test_malloc_kernel(betta_type * betta, uint64_t alloc_size, uint64_t num_allocs){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >=num_allocs) return;


   uint64_t allocation = betta->malloc(alloc_size);

   if (allocation == 0){
      printf("%llu could not alloc\n", tid);
   }

   //saved_allocs[tid] = (uint64_t) allocation;

}


template <typename betta_type>
__global__ void test_malloc_saved_kernel(betta_type * betta, uint64_t * saved_allocs, uint64_t alloc_size, uint64_t num_allocs){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >=num_allocs) return;


   uint64_t allocation = betta->malloc(alloc_size);

   if (allocation == 0){
      printf("%llu could not alloc\n", tid);
   } else {


      uint64_t high = allocation/64;

      uint64_t low = allocation % 64;

      if (atomicOr((unsigned long long int *)&saved_allocs[high], SET_BIT_MASK(low)) & SET_BIT_MASK(low)){
         printf("Duplicate alloc on %llu\n", allocation);
      }

   }
   //saved_allocs[tid] = (uint64_t) allocation;

}


template <typename betta_type>
__global__ void test_malloc_saved_kernel_single(betta_type * betta, uint64_t * saved_allocs, uint64_t alloc_size, uint64_t num_allocs){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;


   for (uint64_t i = 0; i < num_allocs; i++){


      uint64_t allocation = betta->malloc(alloc_size);

      if (allocation == 0){
         printf("%llu could not alloc\n", tid);
      }

      saved_allocs[i] = allocation;

   }

}

template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void beta_random_mallocs(uint64_t num_bytes, uint64_t alloc_size, uint64_t num_allocs){


      using betta_type = poggers::allocators::betta_allocator<mem_segment_size, smallest, largest>;

      betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

      cudaDeviceSynchronize();


      test_malloc_kernel<betta_type><<<(num_allocs-1)/512+1,512>>>(allocator, alloc_size, num_allocs);

      cudaDeviceSynchronize();

      allocator->print_info();

      cudaDeviceSynchronize();

      betta_type::free_on_device(allocator);

}


template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void beta_random_mallocs_save(uint64_t num_bytes, uint64_t alloc_size, uint64_t num_allocs){


      using betta_type = poggers::allocators::betta_allocator<mem_segment_size, smallest, largest>;

      betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

      uint64_t * allocs;

      cudaMalloc((void **)&allocs, sizeof(uint64_t)*((num_bytes/smallest-1)/64+1));

      cudaDeviceSynchronize();

      auto malloc_start = high_resolution_clock::now();

      test_malloc_saved_kernel<betta_type><<<(num_allocs-1)/512+1,512>>>(allocator, allocs, alloc_size, num_allocs);

      cudaDeviceSynchronize();

      auto malloc_end = high_resolution_clock::now();

      double time_taken = elapsed(malloc_start, malloc_end);

      printf("Done with malloc\n");

      std::cout << std::fixed << "Cycle took " << time_taken << " for " << num_allocs << " allocations, " << 1.0*num_allocs/time_taken << " per second." << std::endl;

      cudaDeviceSynchronize();

      //assert_unique_mallocs<<<(num_allocs-1)/512+1,512>>>(allocs, num_allocs);

      cudaDeviceSynchronize();

      allocator->print_info();

      cudaDeviceSynchronize();

      cudaFree((allocs));

      betta_type::free_on_device(allocator);

}


template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void beta_random_mallocs_save_single(uint64_t num_bytes, uint64_t alloc_size, uint64_t num_allocs){


      using betta_type = poggers::allocators::betta_allocator<mem_segment_size, smallest, largest>;

      betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

      uint64_t * allocs;

      cudaMalloc((void **)&allocs, sizeof(uint64_t)*num_allocs);

      cudaDeviceSynchronize();


      test_malloc_saved_kernel_single<betta_type><<<1, 1>>>(allocator, allocs, alloc_size, num_allocs);

      cudaDeviceSynchronize();

      printf("Done with malloc\n");

      cudaDeviceSynchronize();

      assert_unique_mallocs<<<(num_allocs-1)/512+1,512>>>(allocs, num_allocs);

      cudaDeviceSynchronize();

      allocator->print_info();

      cudaDeviceSynchronize();

      cudaFree((allocs));

      betta_type::free_on_device(allocator);

}



//Timing split into sections.
template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void beta_section_timing(uint64_t num_bytes){

   using betta_type = poggers::allocators::betta_allocator<mem_segment_size, smallest, largest>;

   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

   cudaDeviceSynchronize();

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   register_all_segments<betta_type><<<(num_segments-1)/512+1,512>>>(allocator, num_segments);

   printf("Ext sees %llu segments\n", num_segments);
   cudaDeviceSynchronize();

   poggers::utils::print_mem_in_use();


   cudaDeviceSynchronize();

   //malloc_all_blocks_single_thread<betta_type><<<1,1>>>(allocator, num_segments, 256);
   //malloc_all_blocks<betta_type><<<(num_segments*128-1)/512+1,512>>>(allocator, num_segments*128);

   malloc_all_blocks_betta<betta_type><<<(num_segments*256-1)/512+1,512>>>(allocator, num_segments, 256);

   cudaDeviceSynchronize();

   peek<betta_type><<<1,1>>>(allocator);

   cudaDeviceSynchronize();

   betta_type::free_on_device(allocator);

}


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   // boot_ext_tree<8ULL*1024*1024, 16ULL>();
 
   // boot_ext_tree<8ULL*1024*1024, 4096ULL>();


   // boot_alloc_table<8ULL*1024*1024, 16ULL>();


   //boot_betta_malloc_free<16ULL*1024*1024, 16ULL, 64ULL>(30ULL*1000*1000*1000);

   //single thread works - so issue is in memory consistency...
   beta_random_mallocs_save<16ULL*1024*1024, 16ULL, 16ULL>(2000ULL*16*1024*1024, 16, 6000000);


   //beta_random_mallocs<16ULL*1024*1024, 16ULL, 128ULL>(2000ULL*16*1024*1024, 64, 10000000);

   //betta_alloc_random<16ULL*1024*1024, 16ULL, 128ULL>(2000ULL*16*1024*1024, 100000);

   cudaDeviceReset();
   return 0;

}
