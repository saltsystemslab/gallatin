/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <poggers/counter_blocks/beta.cuh>

#include <poggers/beta/timer.cuh>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

#include <fstream>
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <random>


using namespace beta::allocators;


template<typename T>
__host__ __device__ __forceinline__ T divup(T a, T b)
{
   return (a + b - 1) / b;
}

// ##############################################################################################################################################
//
template<typename T, typename O>
constexpr __host__ __device__ __forceinline__ T divup(T a, O b)
{
   return (a + b - 1) / b;
}

// ##############################################################################################################################################
//
template<typename T>
constexpr __host__ __device__ __forceinline__ T alignment(const T size, size_t alignment)
{
   return divup<T, size_t>(size, alignment) * alignment;
}


// __global__ void test_kernel(veb_tree * tree, uint64_t num_removes, int num_iterations){


//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    if (tid >= num_removes)return;


//       //printf("Tid %lu\n", tid);


//    for (int i=0; i< num_iterations; i++){


//       if (!tree->remove(tid)){
//          printf("BUG\n");
//       }

//       tree->insert(tid);

//    }


#if BETA_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 256
#endif



template <typename MemoryManagerType>
__global__ void d_testAllocation(MemoryManagerType * mm, int** verification_ptr, int num_threads, unsigned int * allocation_sizes)
{
   uint64_t tid = poggers::utils::get_tid();


   if(tid >= num_threads)
      return;

   verification_ptr[tid] = reinterpret_cast<int*>(mm->malloc(allocation_sizes[tid]));
}

template <typename MemoryManagerType>
__global__ void d_testFree(MemoryManagerType * mm, int** verification_ptr, int num_threads)
{

   uint64_t tid = poggers::utils::get_tid();


   if(tid >= num_threads)
      return;

   mm->free(verification_ptr[tid]);
}

__global__ void d_testWriteToMemory(int** verification_ptr, int num_threads, unsigned int * allocation_sizes)
{
   uint64_t tid = poggers::utils::get_tid();

   if(tid >= num_threads)
      return;
   
   auto ptr = verification_ptr[tid];

   //this triggers... why?
   // if (ptr == nullptr){
   //    printf("got nullptr\n");
   // } 

   auto allocation_size = allocation_sizes[tid];

   for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
   {
      ptr[i] = tid;
   }
}

__global__ void d_testReadFromMemory(int** verification_ptr, int num_threads, unsigned int * allocation_sizes)
{
   uint64_t tid = poggers::utils::get_tid();


   if(tid >= num_threads)
      return;

   auto allocation_size = allocation_sizes[tid];
   
   auto ptr = verification_ptr[tid];

   for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
   {
      if(ptr[i] != tid)
      {
         printf("%d | We got a wrong value here! %d vs %d\n", tid, ptr[i], tid);
         __trap();
      }
   }
}


//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void beta_mixed_frag(uint64_t num_bytes, uint64_t num_threads, unsigned int max_size, uint64_t num_rounds){


   beta::utils::timer boot_timing;

   using betta_type = beta::allocators::beta_allocator<mem_segment_size, smallest, largest>;

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   uint64_t max_allocs_per_segment = mem_segment_size/smallest;

   uint64_t num_allocs = max_allocs_per_segment*num_segments;

   printf("Starting test with %lu segments, %lu allocs per segment, %lu threads in kernel\n", num_segments, max_allocs_per_segment, num_threads);


   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);


   unsigned int alloc_size_low = 16;

   if (max_size < alloc_size_low){
      max_size = alloc_size_low;
   }

   max_size = alignment(max_size, sizeof(int));
   alloc_size_low = alignment(alloc_size_low, sizeof(int));


   auto range = max_size - alloc_size_low;
   auto offset = alloc_size_low;


   int** d_memory{nullptr};
   GPUErrorCheck(cudaMalloc(&d_memory, sizeof(int*) * num_threads));


   //addition: holder for alloc size
   unsigned int * d_allocation_sizes{nullptr};

   std::vector<unsigned int> allocation_sizes(num_threads);

   GPUErrorCheck(cudaMalloc(&d_allocation_sizes, sizeof(unsigned int) * num_threads));





   std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;


   auto gridSize = (num_threads-1)/TEST_BLOCK_SIZE+1;
   auto blockSize = TEST_BLOCK_SIZE;



   for (int i = 0; i < num_rounds; i++){

      //std::cout << "#" << std::flush;
      std::cout << i << std::endl;

      std::mt19937 gen(i);
      std::uniform_real_distribution<> dis(0.0, 1.0);
      srand(i);

      for (auto i = 0; i < num_threads; i++){
         allocation_sizes[i] = alignment(offset + dis(gen) * range, sizeof(int));
      }

      GPUErrorCheck(cudaMemcpy(d_allocation_sizes, allocation_sizes.data(), sizeof(unsigned int) * num_threads, cudaMemcpyHostToDevice));

      d_testAllocation <betta_type> <<<gridSize, blockSize>>>(allocator, d_memory, num_threads, d_allocation_sizes);
      GPUErrorCheck(cudaDeviceSynchronize());

      allocator->print_info();

      d_testWriteToMemory<<<gridSize, blockSize>>>(d_memory, num_threads, d_allocation_sizes);

      GPUErrorCheck(cudaDeviceSynchronize());

      d_testReadFromMemory<<<gridSize, blockSize>>>(d_memory, num_threads, d_allocation_sizes);

      GPUErrorCheck(cudaDeviceSynchronize());

      d_testFree <betta_type> <<<gridSize, blockSize>>>(allocator, d_memory, num_threads);
      GPUErrorCheck(cudaDeviceSynchronize());

      printf("End of round\n");

      allocator->print_info();



   }

   std::cout << std::endl << std::flush;

   GPUErrorCheck(cudaFree(d_memory));
   GPUErrorCheck(cudaFree(d_allocation_sizes));


   betta_type::free_on_device(allocator);

   cudaDeviceSynchronize();

}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   uint64_t num_threads;

   unsigned int max_size;

   int num_rounds = 1;


   if (argc < 2){
      num_threads = 1000000;
   } else {
      num_threads = std::stoull(argv[1]);
   }

   if (argc < 3){
      max_size = 16;
   } else {
      max_size = std::stoull(argv[2]);
   }


   if (argc < 4){
      num_rounds = 50;
   } else {
      num_rounds = std::stoull(argv[3]);
   }

   beta_mixed_frag<16ULL*1024*1024, 16ULL, 4096ULL>(8ULL*1024*1024*1024, num_threads, max_size, num_rounds);

   //beta_full_churn<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments, num_rounds);


   //beta_pointer_churn<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments, num_rounds);


   //beta_churn_no_free<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments);



   cudaDeviceReset();
   return 0;

}
