/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



//This tests the fixed vector type, which allows for vector operations within
// a set range of sizes.
//Fixing he size of the vector alows for faster operations
// due to the stability of the vector components.

#include <gallatin/data_structs/fixed_vector.cuh>
#include <gallatin/allocators/timer.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace gallatin::data_structs;
using namespace gallatin::allocators;


template <uint64_t min, uint64_t max>
__global__ void single_vector_test(){


   //using vector_type = gallatin::data_structs::fixed_vector<uint64_t, min, max>;



   uint64_t tid = gallatin::utils::get_tid();

   if (tid != 0) return;


   gallatin::data_structs::fixed_vector<uint64_t, min, max> test_vector;


   for (uint64_t i=0; i < max; i++){

      test_vector.insert(i);

   }



   printf("Done with insert, verifying\n");

   for (uint64_t i =0; i < max; i++){

      if (test_vector[i] != i) printf("Failed to set index %lu\n", i);
   }

   printf("Test finished\n");

   test_vector.free_vector();
  

}


template <uint64_t min, uint64_t max, bool on_host>
__global__ void multi_vector_test(gallatin::data_structs::fixed_vector<uint64_t, min, max, on_host> * test_vector){


   //using vector_type = gallatin::data_structs::fixed_vector<uint64_t, min, max>;



   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= max) return;

   //printf("Started %lu\n", tid);

   uint64_t index = test_vector->insert(tid);

   if (index == ~0ULL){
      printf("Failed insert\n");
   } else {
      uint64_t output = test_vector[0][index];

      if (output != tid){
         printf("Index %lu does not match: %lu instead of %lu\n", index, output, tid);
      }
   }

   //printf("ended %lu\n", tid);

}

template <uint64_t min, uint64_t max>
__host__ void init_and_test_vector(){

   //init_global_allocator(30ULL*1024*1024*1024, 42);


   cudaDeviceSynchronize();
   

   gallatin::utils::timer vector_timing;

   single_vector_test<min, max><<<1,1>>>();

   vector_timing.sync_end();

   vector_timing.print_throughput("vector enqueued", max);


   //free_global_allocator();

}

template <typename vector_type>
__global__ void verify_vector(vector_type * vector, uint64_t * bits, uint64_t nitems){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems)return;

   uint64_t my_item = vector[0][tid];

   uint64_t high = my_item/64;
   uint64_t low = my_item % 64;


   if (atomicOr((unsigned long long int *)&bits[high], SET_BIT_MASK(low)) & SET_BIT_MASK(low)){
      printf("Double set for item %llu with index %lu\n", my_item, tid);
   }

}

template <uint64_t min, uint64_t max, bool on_host>
__host__ void init_and_test_multi(){


   using vector_type = gallatin::data_structs::fixed_vector<uint64_t, min, max, on_host>;

   vector_type * dev_vector = vector_type::get_device_vector();

   cudaDeviceSynchronize();

   gallatin::utils::timer vector_timing;

   multi_vector_test<min, max, on_host><<<(max-1)/256+1,256>>>(dev_vector);

   vector_timing.sync_end();

   vector_timing.print_throughput("vector enqueued", max);


   uint64_t * bits;

   uint64_t num_uints = (max-1)/64+1;

   cudaMallocManaged((void **)&bits, sizeof(uint64_t)*num_uints);

   cudaMemset(bits, 0, sizeof(uint64_t)*num_uints);

   verify_vector<vector_type><<<(max-1)/256+1,256>>>(dev_vector, bits, max);

   cudaDeviceSynchronize();

   cudaFree(bits);

   vector_type::free_device_vector(dev_vector);



}


template <uint64_t min, uint64_t max, bool on_host>
__host__ void init_and_test_output(){


   using vector_type = gallatin::data_structs::fixed_vector<uint64_t, min, max, on_host>;

   vector_type * dev_vector = vector_type::get_device_vector();

   cudaDeviceSynchronize();

   gallatin::utils::timer vector_timing;

   multi_vector_test<min, max, on_host><<<(max-1)/256+1,256>>>(dev_vector);

   vector_timing.sync_end();

   vector_timing.print_throughput("vector enqueued", max);

   gallatin::utils::timer vector_export;
   auto vector_output = vector_type::export_to_host(dev_vector);
   //vector_type::free_device_vector(dev_vector);

   vector_export.sync_end();

   vector_export.print_throughput("vector exported", max);



   printf("First = %lu, last = %lu\n", vector_output[0], vector_output[vector_output.size()-1]);
   vector_type::free_device_vector(dev_vector);


}


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   //one_boot_betta_test_all_sizes<16ULL*1024*1024, 16ULL, 16ULL>(num_segments*16*1024*1024);  


   //beta_test_allocs_correctness<16ULL*1024*1024, 16ULL, 4096ULL>(num_segments*16*1024*1024, num_rounds, size);

   //init_global_allocator(30ULL*1024*1024*1024, 42);
   init_global_allocator_combined(20ULL*1024*1024*1024, 20ULL*1024*1024*1024, 42);

   //init_and_test_vector<16ULL, 16384ULL>();

   init_and_test_multi<16ULL, 16384ULL, false>();
   init_and_test_multi<16ULL, 65336ULL, false>();

   init_and_test_multi<16384ULL, 65336ULL, false>();
   init_and_test_multi<16384ULL, 65536ULL, false>();
   
   init_and_test_multi<16ULL, 1048576ULL, false>();
   init_and_test_multi<32ULL, 1048576ULL, false>();
   init_and_test_multi<64ULL, 1048576ULL, false>();
   init_and_test_multi<128ULL, 1048576ULL, false>();
   init_and_test_multi<256ULL, 1048576ULL, false>();
   init_and_test_multi<16ULL, 1073741824ULL, false>();
   init_and_test_multi<1048576ULL, 1073741824ULL, false>();

   init_and_test_multi<16384ULL, 65336ULL, true>();
   init_and_test_multi<16384ULL, 65536ULL, true>();
   
   init_and_test_multi<16ULL, 1048576ULL, true>();
   init_and_test_multi<32ULL, 1048576ULL, true>();
   init_and_test_multi<64ULL, 1048576ULL, true>();
   init_and_test_multi<128ULL, 1048576ULL, true>();
   init_and_test_multi<256ULL, 1048576ULL, true>();
   init_and_test_multi<16ULL, 1073741824ULL, true>();
   init_and_test_multi<1048576ULL, 1073741824ULL, true>();


   //init_and_test_multi<16ULL, 64ULL>();
   // init_and_test_multi<16ULL, 65336ULL>();


   init_and_test_output<1048576ULL, 1073741824ULL, false>();
   init_and_test_output<1048576ULL, 1073741824ULL, true>();


   free_global_allocator_combined();

   //beta_full_churn<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments, num_rounds);


   //beta_pointer_churn<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments, num_rounds);


   //beta_churn_no_free<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments);



   cudaDeviceReset();
   return 0;

}
