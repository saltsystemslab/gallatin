/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




   
#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/data_structs/callocable.cuh>
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


using namespace gallatin::data_structs;

template <typename callocable_type>
__global__ void get_callocable_typeinit(callocable_type * callocable_wrapper, uint64_t nitems){

   //callocable_type * calloc_ptr = (callocable_type *) global_malloc(size(callocable_type));

   uint64_t tid = gallatin::utils::get_tid();

   if (tid != 0) return;

   callocable_wrapper[0] = callocable_type(nitems);


}

//simulate malloc of calloced type.
template <typename callocable_type, typename T>
__global__ void preload_calloced_type(callocable_type * callocable_wrapper, uint64_t nitems){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems) return;

   callocable_wrapper->debug_set_memory(tid, T(tid));

}


template<typename callocable_type>
__global__ void calloc_query_kernel(callocable_type * callocable_wrapper, uint64_t nitems){


   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems) return;

   auto my_item = callocable_wrapper[0][tid];

   if (my_item != 0){
      printf("Failed to unset address %lu\n", tid);
   }
}


template <typename callocable_type>
__global__ void free_calloced_memory(callocable_type * callocable_wrapper){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid != 0) return;

   callocable_wrapper->free_memory();


}


template <typename T, int stride>
__host__ void calloc_test(uint64_t nitems, int num_rounds){


   using callocable_type = callocable<T, stride>;


   callocable_type * callocable_wrapper;

   cudaMallocManaged((void **)&callocable_wrapper, sizeof(callocable_type));



   cudaDeviceSynchronize();

   gallatin::utils::timer init_timing;

   get_callocable_typeinit<<<1,1>>>(callocable_wrapper, nitems);

   std::cout << "Init in " << init_timing.sync_end() << " seconds" << std::endl;

   preload_calloced_type<callocable_type, T><<<(nitems-1)/256+1,256>>>(callocable_wrapper, nitems);


   cudaDeviceSynchronize();


   for (int i = 0; i< num_rounds; i++){

      gallatin::utils::timer query_timing;


      calloc_query_kernel<callocable_type><<<(nitems-1)/256+1,256>>>(callocable_wrapper, nitems);


      query_timing.sync_end();


      query_timing.print_throughput("Checked calloced memory", nitems);



   }
   
   free_calloced_memory<callocable_type><<<1,1>>>(callocable_wrapper);



}


template <typename T, int stride>
__host__ void calloc_battery(){

   printf("Stride %d\n", stride);

   calloc_test<T, stride>(10000, 5);

   cudaDeviceSynchronize();

   calloc_test<T, stride>(100000, 5);

   cudaDeviceSynchronize();

   calloc_test<T, stride>(1000000, 5);

   cudaDeviceSynchronize();

   calloc_test<T, stride>(10000000, 5);

   cudaDeviceSynchronize();

   calloc_test<T, stride>(100000000, 5);

   cudaDeviceSynchronize();


}


template <typename T>
__global__ void query_array_kernel(T * array, uint64_t nitems){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems) return;

   if (array[tid] != 0){
      printf("Failed to memset index %llu\n", tid);
   }

}

template <typename T>
void array_test(uint64_t nitems, int num_rounds){

   T * array;

   gallatin::utils::timer init_timing;
   cudaMalloc((void **)&array, sizeof(T)*nitems);

   cudaMemset(array, 0, sizeof(T)*nitems);

   std::cout << "Init array in " << init_timing.sync_end() << std::endl;


   for (int i = 0; i < num_rounds; i++){


      gallatin::utils::timer query_timing;


      query_array_kernel<T><<<(nitems-1)/256+1,256>>>(array, nitems);


      query_timing.sync_end();


      query_timing.print_throughput("queried Array memory", nitems);



   }

   cudaDeviceSynchronize();

   cudaFree(array);

}

template <typename T>
__host__ void array_battery(){

   printf("Array tests\n");

   array_test<T>(10000, 5);

   cudaDeviceSynchronize();

   array_test<T>(100000, 5);

   cudaDeviceSynchronize();

   array_test<T>(1000000, 5);

   cudaDeviceSynchronize();

   array_test<T>(10000000, 5);

   cudaDeviceSynchronize();

   array_test<T>(100000000, 5);

   cudaDeviceSynchronize();

}

//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   gallatin::allocators::init_global_allocator(8ULL*1024*1024*1024, 42);


   cudaDeviceSynchronize();

   //single_string_test<<<1,1>>>(1);

   //single_string_test_copy<<<1,1>>>();
   cudaDeviceSynchronize();

   //string_test_floats<<<1,1>>>(1000);

   calloc_battery<uint64_t,1>();
   calloc_battery<uint64_t,4>();
   calloc_battery<uint64_t,8>();
   calloc_battery<uint64_t,16>();

   cudaDeviceSynchronize();

   array_battery<uint64_t>();

   cudaDeviceSynchronize();

   gallatin::allocators::free_global_allocator();

   cudaDeviceReset();
   return 0;

}
