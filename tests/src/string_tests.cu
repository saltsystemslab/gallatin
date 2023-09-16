/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




   
#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/data_structs/custring.cuh>
#include <gallatin/allocators/timer.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace gallatin::allocators;


#if BETA_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 512
#endif


__global__ void single_string_test(uint64_t n_sums){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid != 0 )return;

   gallatin::data_structs::custring test_string("Data ");

   for (uint i = 0; i < 1000; i++){

      gallatin::data_structs::custring alt_string(i);

      alt_string.print_string_device();

      test_string = test_string + alt_string;

   }

   test_string.print_string_device();
   // for (uint64_t i = 0; i < n_sums; i++){

   //    if (i != n_sums -1){

   //       gallatin::data_structs::custring alt_string = gallatin::data_structs::custring(i) + ",";
   //       test_string = test_string + alt_string;
   //    } else {
   //       test_string = test_string + gallatin::data_structs::custring(i) + ".";
   //    }
      
   //    test_string.print_string_device();
   // }

}

//test the copy constructor when not reflexive?
__global__ void single_string_test_copy(){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid != 0 )return;

   gallatin::data_structs::custring test_string("FOO");

   gallatin::data_structs::custring alt_string("BAR");
     
   //this works   
   //gallatin::data_structs::custring mixed = test_string + alt_string;

   test_string = test_string+alt_string;
   test_string.print_string_device();



}

__global__ void string_test_floats(uint64_t n_sums){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid != 0 )return;

   gallatin::data_structs::custring test_string("Data ");

   for (uint64_t i = 0; i < n_sums; i++){

      if (i != n_sums -1){
         test_string = test_string + gallatin::data_structs::custring(1.0*i/n_sums) + ", ";
      } else {
         test_string = test_string + gallatin::data_structs::custring(1.0*i/n_sums);
      }
      
     
   }

   test_string.print_string_device();
   test_string.print_info();


}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   gallatin::allocators::init_global_allocator(8ULL*1024*1024*1024, 42);


   cudaDeviceSynchronize();

   single_string_test<<<1,1>>>(1);

   //single_string_test_copy<<<1,1>>>();
   cudaDeviceSynchronize();

   string_test_floats<<<1,1>>>(1000);

   cudaDeviceSynchronize();

   gallatin::allocators::free_global_allocator();

   cudaDeviceReset();
   return 0;

}
