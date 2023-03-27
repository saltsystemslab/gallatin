/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/simple_merged_allocator.cuh>

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



template <typename allocator>
__global__ void test_single_thread_malloc_only(allocator * alloc, uint64_t num_allocs){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   for (uint64_t i = 0; i < num_allocs; i++){

      void * allocation = alloc->malloc();

      if (allocation == nullptr){printf("malloc Error\n"); }

      else {


      if (!alloc->free(allocation)) printf("Free Error\n");


      }
      //printf("i/offset: %llu / %llu\n", i, test_val);

   }

}


template <typename allocator>
__global__ void test_multi_thread_malloc_only(allocator * alloc, uint64_t num_allocs){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;

   void * allocation = alloc->malloc();


   if (allocation != nullptr){

      alloc->free(allocation);
   } else { 

      printf("Fail!\n");

   }

   //printf("i/offset: %llu / %llu\n", tid, test_val);

}


template <typename allocator>
__global__ void test_multi_thread_rounds(allocator * alloc, uint64_t num_allocs, uint64_t num_rounds){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   //if (threadIdx.x == 0) printf("Progessing %llu\n", tid);

   if (tid >= num_allocs) return;


   for (uint64_t i =0; i < num_rounds; i++){


      void * allocation = alloc->malloc();


      if (allocation != nullptr){

         alloc->free(allocation);

      } else { 

         printf("Fail!\n");
      }


   }

   //printf("Finished with %llu\n", tid);

   //printf("i/offset: %llu / %llu\n", tid, test_val);

}


template <typename allocator>
__host__ void test_multi_thread_alloc(uint64_t num_allocs){



   allocator * alloc = allocator::generate_on_device();


   cudaDeviceSynchronize();

   test_multi_thread_malloc_only<allocator><<<(num_allocs-1)/512+1,512>>>(alloc, num_allocs);

   cudaDeviceSynchronize();

   allocator::free_on_device(alloc);


   printf("Done with multi %llu\n", num_allocs);
}


template <typename allocator>
__host__ void test_multi_thread_alloc_rounds(uint64_t num_allocs, uint64_t num_rounds){



   allocator * alloc = allocator::generate_on_device();


   cudaDeviceSynchronize();

   auto rounds_start = std::chrono::high_resolution_clock::now();

   test_multi_thread_rounds<allocator><<<(num_allocs-1)/512+1,512>>>(alloc, num_allocs, num_rounds);

   cudaDeviceSynchronize();

   auto rounds_end = std::chrono::high_resolution_clock::now();




   std::chrono::duration<double> rounds_diff = rounds_end-rounds_start;

   allocator::free_on_device(alloc);

   printf("Done with multi rounds %llu %llu\n", num_allocs, num_rounds);

   std::cout << "Inserted " << num_allocs*num_rounds << " in " << rounds_diff.count() << " seconds\n";

   printf("Malloc/Free pair throughput: %f \n", 1.0*num_allocs*num_rounds/rounds_diff.count());
 

}



template <typename allocator>
__host__ void test_single_thread_alloc(uint64_t num_allocs){



   allocator * alloc = allocator::generate_on_device();


   cudaDeviceSynchronize();

   test_single_thread_malloc_only<allocator><<<1,1>>>(alloc, num_allocs);

   cudaDeviceSynchronize();

   allocator::free_on_device(alloc);
}


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {



   // printf("Depth 32: %d 33: %d; 1024: %d; 32768: %d\n", determine_depth<32>::depth, determine_depth<33>::depth, determine_depth<1024>::depth, determine_depth<32768>::depth);

   // printf(" %d %d %d %d\n", determine_num_allocations<determine_depth<32>::depth>::count, determine_num_allocations<determine_depth<33>::depth>::count, determine_num_allocations<determine_depth<1024>::depth>::count, determine_num_allocations<determine_depth<32768>::depth>::count);

   using allocator = simple_allocator_one_size<128>;

   test_single_thread_alloc<allocator>(1);

   test_single_thread_alloc<allocator>(32);

   // //using allocator_1 = bitbuddy_allocator<1024,1>;


   // //test_multi_thread_alloc<allocator_1>(1024);

   // //test_single_thread_alloc<allocator_1>(1024);

   test_multi_thread_alloc_rounds<allocator>(1024, 1);


   test_multi_thread_alloc_rounds<allocator>(32768, 1);

   test_multi_thread_alloc_rounds<allocator>(33554432, 1);




   // //test_multi_thread_alloc<allocator_2>(32768);

   // test_multi_thread_alloc_rounds<allocator>(32768, 10);

   // test_multi_thread_alloc_rounds<allocator>(1048576, 10);

   // test_multi_thread_alloc_rounds<allocator>(33554432, 10);

   


   return 0;

}
