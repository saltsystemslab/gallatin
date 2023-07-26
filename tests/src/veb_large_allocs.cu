/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/counter_blocks/veb.cuh>
#include <poggers/allocators/alloc_utils.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>


using namespace beta::allocators;


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







__global__ void veb_alloc_kernel(veb_tree * test_allocator){


   uint64_t tid = poggers::utils::get_tid();

   if (tid != 0) return;

   uint64_t first = test_allocator->gather_multiple(2);

   uint64_t second = test_allocator->gather_multiple(15);

   uint64_t third = test_allocator->gather_multiple(128);

   printf("Acquired indices %llu %llu %llu\n", first, second, third);



   test_allocator->return_multiple(third, 128);

   test_allocator->return_multiple(first, 2);

   test_allocator->return_multiple(second, 15);



}


int main(int argc, char** argv) {

   uint64_t num_items = 2048;

   veb_tree * test_allocator = veb_tree::generate_on_device(num_items, 42);


   cudaDeviceSynchronize();

   veb_alloc_kernel<<<1,1>>>(test_allocator);

   cudaDeviceSynchronize();

   veb_tree::free_on_device(test_allocator);

   cudaDeviceSynchronize();
 
   cudaDeviceReset();
   return 0;

}
