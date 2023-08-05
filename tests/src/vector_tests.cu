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

#include <poggers/data_structs/vector.cuh>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace beta::allocators;


#if BETA_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 256
#endif



template <typename allocator>
__global__ void init_vector_test(allocator * alloc){


   using vector_type = gallatin::data_structs::vector<uint64_t, allocator>;


   uint64_t tid = poggers::utils::get_tid();

   if (tid != 0) return;


   vector_type test_vector(4, alloc);


   for (uint64_t i=0; i < 9; i++){

      test_vector.insert(i);

   }


   test_vector.free_vector();
  

}


//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void vector_test_one(uint64_t num_bytes){


   beta::utils::timer boot_timing;

   using betta_type = beta::allocators::beta_allocator<mem_segment_size, smallest, largest>;


   betta_type * allocator = betta_type::generate_on_device(num_bytes, 111);

   std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;

   beta::utils::timer kernel_timing;
   init_vector_test<betta_type><<<1,1>>>(allocator);
   kernel_timing.sync_end();

   allocator->print_info();

   betta_type::free_on_device(allocator);

   cudaDeviceSynchronize();

}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   //one_boot_betta_test_all_sizes<16ULL*1024*1024, 16ULL, 16ULL>(num_segments*16*1024*1024);  


   //beta_test_allocs_correctness<16ULL*1024*1024, 16ULL, 4096ULL>(num_segments*16*1024*1024, num_rounds, size);


   vector_test_one<16ULL*1024*1024, 16ULL, 4096ULL>(16ULL*1024*1024*1024);

   //beta_full_churn<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments, num_rounds);


   //beta_pointer_churn<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments, num_rounds);


   //beta_churn_no_free<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments);



   cudaDeviceReset();
   return 0;

}
