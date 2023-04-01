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


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace poggers::allocators;

template <typename betta_type>
__global__ void test_malloc_kernel(betta_type * betta, uint64_t alloc_size, uint64_t num_allocs){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >=num_allocs) return;


   void * allocation = betta->malloc(alloc_size);

   if (allocation == nullptr){
      printf("%llu could not alloc\n", tid);
   }

   //saved_allocs[tid] = (uint64_t) allocation;

}

template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void beta_random_mallocs(uint64_t num_bytes, uint64_t alloc_size, uint64_t num_allocs){


      using betta_type = poggers::allocators::betta_allocator<mem_segment_size, smallest, largest>;

      betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

      cudaDeviceSynchronize();


      test_malloc_kernel<betta_type><<<(num_allocs-1)/512+1,512>>>(allocator, alloc_size, num_allocs);

      cudaDeviceSynchronize();

      betta_type::free_on_device(allocator);

}


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   // boot_ext_tree<8ULL*1024*1024, 16ULL>();
 
   // boot_ext_tree<8ULL*1024*1024, 4096ULL>();


   // boot_alloc_table<8ULL*1024*1024, 16ULL>();


   //boot_betta_malloc_free<16ULL*1024*1024, 16ULL, 64ULL>(30ULL*1000*1000*1000);

   //not quite working - get some misses
   beta_random_mallocs<16ULL*1024*1024, 16ULL, 128ULL>(2000ULL*16*1024*1024, 64, 10);

   //betta_alloc_random<16ULL*1024*1024, 16ULL, 128ULL>(2000ULL*16*1024*1024, 100000);

   cudaDeviceReset();
   return 0;

}
