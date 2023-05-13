/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




//#include <poggers/allocators/slab_one_size.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace std::chrono;


#include <cooperative_groups.h>


#include <poggers/beta/slab_one_size.cuh>

#include <poggers/beta/timer.cuh>

namespace cg = cooperative_groups;

using namespace beta::allocators;


__global__ void warp_lock_tests(){

   

   
}


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   // for (int i =0; i< 20; i++){
   //    boot_slab_one_size();
   // }
   
   //test_num_malloc_frees<4>(1000, 1);


   //test_num_malloc_frees(10000, 100);

   //test_num_malloc_frees(10000, 10);

   //test_num_malloc_frees<4>(1000000000, 10);

   //test_num_malloc_frees<4>(10000000, 10);

   //test_num_malloc_frees(100000000, 10);

   //test_num_malloc_frees_bitarr<4>(100000000, 10);

   uint64_t num_threads;

   uint64_t num_allocs;
   

   if (argc < 2){
      num_threads = 10000000;
      num_allocs = 100;
   } else if (argc < 3) {
      num_threads = std::stoull(argv[1]);
      num_allocs = 100;
   } else {

      num_threads = std::stoull(argv[1]);
      num_allocs = std::stoull(argv[2]);

   }

   //test_num_malloc_frees_bitarr<4>(100000000, 10);

   test_churn<4>(num_threads, num_allocs);



 
   cudaDeviceReset();
   return 0;

}
