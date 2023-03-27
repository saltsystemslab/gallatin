/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/four_bit_bitbuddy.cuh>

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


//helper to assert uniqueness
__global__ void assert_unique(uint64_t * unique_ids, uint64_t num_allocs){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;

   uint64_t my_allocation = unique_ids[tid];


   if (my_allocation == (~0ULL)){


      printf("FAIL to malloc\n");
      asm("trap;");
   }

   for (uint64_t i =0; i < num_allocs; i++){


      if (i != tid && my_allocation == unique_ids[i]){
         asm("trap;");
      }
   }

}


//Unit Test 1
// cuda kernel
// with an exact thread match, can we successfully allocate all blocks in a layer?
__global__ void one_thread_test(templated_bitbuddy_four<1> * alloc, uint64_t num_threads, uint64_t * unique_ids){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;


   for (int i =0; i< num_threads; i++){



   uint64_t my_alloc = alloc->malloc_offset(1);

   //void * my_alloc = nullptr;

  


      if (my_alloc == (~0ULL)){
         printf("Problem! in Malloc!\n");
         asm("trap;");

      }

      else {

         //alloc->assert_correct_setup(my_alloc);

         unique_ids[i] = (uint64_t) my_alloc;

      }

      printf("Done with %i/%llu\n", i, num_threads);

   }


   for (uint64_t i=0; i< num_threads; i++){


      alloc->free(1);
   }

}

//comprehensive tests for single-threaded bitbuddy
//used to verify internal components without memory contention.
__host__ bool comp_single_threaded(){


   uint64_t * grabbed_allocs;


   cudaMalloc((void **)&grabbed_allocs, sizeof(uint64_t)*32);
   
   templated_bitbuddy_four<1> * alloc = templated_bitbuddy_four<1>::generate_on_device();

   cudaDeviceSynchronize();


   one_thread_test<<<1,1>>>(alloc, 31, grabbed_allocs);

   cudaDeviceSynchronize();

   assert_unique<<<1,31>>>(grabbed_allocs, 31);

   cudaDeviceSynchronize();

   //buddy_alloc_free_test_1<<<1,1>>>(alloc, 31, grabbed_allocs);

   cudaDeviceSynchronize();


   templated_bitbuddy_four<1>::free_on_device(alloc);

   cudaDeviceSynchronize();

   cudaFree(grabbed_allocs);


   return true;



}




//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   // if (!test_one_thread()){
   //    printf("Test one thread: [FAIL]\n");
   // } else {
   //    printf("Test one thread: [PASS]\n");
   // }

   comp_single_threaded();

 

   return 0;

}
