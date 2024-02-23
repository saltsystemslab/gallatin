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


__global__ void test_string_combination(){

   uint64_t tid = gallatin::utils::get_tid();

   //gallatin::data_structs::make_string("This is a test with tid: ", tid, "/", 10U, "\n");


   auto s1 = gallatin::data_structs::make_string_device("This is a test with tid: ", 12345.12345, " ", 12345.12345*-1, " ", 1000/100, " ", 1000.0/100);

      //, ", Next is gonna be: ", tid+1, " along with double: ", .5, " and negative ", -5, " vs positive: ", 5*25, "\n");

   s1.print_string_device();

   auto my_string = gallatin::data_structs::make_string_device("This is a test with tid: ", tid, ", Next is gonna be: ", tid+1, " along with double: ", .5, " and negative ", -5, " vs positive: ", 5*25, "\n");

   my_string.print_string_device();

}


__global__ void test_host_string_combination(){

   uint64_t tid = gallatin::utils::get_tid();

   //gallatin::data_structs::make_string("This is a test with tid: ", tid, "/", 10U, "\n");


   auto s1 = gallatin::data_structs::make_string_host("This is a test with tid: ", 12345.12345, " ", 12345.12345*-1, " ", 1000/100, " ", 1000.0/100);

      //, ", Next is gonna be: ", tid+1, " along with double: ", .5, " and negative ", -5, " vs positive: ", 5*25, "\n");

   s1.print_string_device();

   auto my_string = gallatin::data_structs::make_string_host("This is a test with tid: ", tid, ", Next is gonna be: ", tid+1, " along with double: ", .5, " and other double ", .005, " and negative ", -5, " vs positive: ", 5*25, "\n");

   auto sample_string = gallatin::data_structs::make_string_host(12345.12345);


   my_string.print_string_device();

}


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   gallatin::allocators::init_global_allocator_combined(8ULL*1024*1024*1024,8ULL*1024*1024*1024, 42);


   cudaDeviceSynchronize();

   //single_string_test<<<1,1>>>(1);

   //single_string_test_copy<<<1,1>>>();
   cudaDeviceSynchronize();

   //string_test_floats<<<1,1>>>(1000);

   test_string_combination<<<1,1>>>();

   cudaDeviceSynchronize();

   test_host_string_combination<<<1,1>>>();

   cudaDeviceSynchronize();

   gallatin::allocators::free_global_allocator_combined();

   cudaDeviceReset();
   return 0;

}
