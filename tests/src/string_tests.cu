/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/counter_blocks/beta.cuh>

#include <poggers/counter_blocks/timer.cuh>

#include <poggers/data_structs/custring.cuh>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace beta::allocators;

using namespace gallatin::data_structs;


//enqueue test kernel loads nitems into the queue, with every item unique based on TID
//then dequeue tests correctness by mapping to bitarry.
template <typename Allocator> 
__global__ void string_boot_test_kernel(Allocator * allocator){

   uint64_t tid = poggers::utils::get_tid();

   if (tid != 0) return;

   using str_type = custring<Allocator>;

   str_type string1("test", allocator);

   string1.print_info();

   str_type string2(1234ULL, allocator);

   string2.print_info();

   
}


__host__ void string_boot_test(){

   using gallatin_allocator = beta::allocators::beta_allocator<16ULL*1024*1024, 16ULL, 4096ULL>;

   //boot with 20 Gigs
   gallatin_allocator * alloc = gallatin_allocator::generate_on_device(20ULL*1024*1024*1024, 111);

   string_boot_test_kernel<gallatin_allocator><<<1,1>>>(alloc);

   cudaDeviceSynchronize();

   gallatin_allocator::free_on_device(alloc);

}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   string_boot_test();

   cudaDeviceReset();
   return 0;

}
