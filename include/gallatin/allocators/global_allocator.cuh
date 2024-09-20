#ifndef GALLATIN_GLOBAL_ALLOCATOR
#define GALLATIN_GLOBAL_ALLOCATOR

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without l> imitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so,
//  subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial
//  portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY,
//  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
//  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.


/*** ABOUT

This is a wrapper for the Gallatin allocator that creates a global variable in scope

This allows threads to reference Gallatin without passing a pointer to the kernel.
*/


#include <gallatin/allocators/gallatin.cuh>

namespace gallatin {

namespace allocators {


using global_allocator_type = gallatin::allocators::Gallatin<16ULL*1024*1024, 16ULL, 4096ULL, 8, 8>;

__device__ global_allocator_type * global_gallatin;

__device__ global_allocator_type * global_host_gallatin;


__host__ void init_global_allocator(uint64_t num_bytes, uint64_t seed, bool print_info=true, bool running_calloc=false){

  global_allocator_type * local_copy = global_allocator_type::generate_on_device(num_bytes, seed, print_info, running_calloc);

  cudaMemcpyToSymbol(global_gallatin, &local_copy, sizeof(global_allocator_type *));

  cudaDeviceSynchronize();

}


__host__ void free_global_allocator(){


  global_allocator_type * local_copy;

  cudaMemcpyFromSymbol(&local_copy, global_gallatin, sizeof(global_allocator_type *));

  cudaDeviceSynchronize();

  global_allocator_type::free_on_device(local_copy);

}

__device__ void * global_malloc(uint64_t num_bytes){

  return global_gallatin->malloc(num_bytes);

}

__device__ void global_free(void * ptr){

  global_gallatin->free(ptr);

}



__host__ void print_global_stats(){

  global_allocator_type * local_copy;

  cudaMemcpyFromSymbol(&local_copy, global_gallatin, sizeof(global_allocator_type *));

  cudaDeviceSynchronize();

  local_copy->print_info();


}

//host_init
// __host__ void init_global_allocator_host(uint64_t num_bytes, uint64_t seed, bool print_info=true, bool running_calloc=false){

//   global_allocator_type * local_copy = global_allocator_type::generate_on_device_host(num_bytes, seed, print_info, running_calloc);

//   cudaMemcpyToSymbol(global_host_gallatin, &local_copy, sizeof(global_allocator_type *));

//   cudaDeviceSynchronize();

// }


// __host__ void free_global_allocator_host(){


//   global_allocator_type * local_copy;

//   cudaMemcpyFromSymbol(&local_copy, global_host_gallatin, sizeof(global_allocator_type *));

//   cudaDeviceSynchronize();

//   global_allocator_type::free_on_device(local_copy);

// }

// __device__ void * global_malloc_host(uint64_t num_bytes){

//   return global_host_gallatin->malloc(num_bytes);

// }

// __device__ void global_free_host(void * ptr){

//   global_host_gallatin->free(ptr);

// }



// __host__ void print_global_stats_host(){

//   global_allocator_type * local_copy;

//   cudaMemcpyFromSymbol(&local_copy, global_host_gallatin, sizeof(global_allocator_type *));

//   cudaDeviceSynchronize();

//   local_copy->print_info();


// }
//end host version


//mixed malloc init
// __host__ void init_global_allocator_combined(uint64_t num_bytes, uint64_t host_bytes, uint64_t seed, bool print_info=true, bool running_calloc=false){

//   init_global_allocator(num_bytes, seed, print_info, running_calloc);
//   init_global_allocator_host(host_bytes, seed, print_info, running_calloc);


// }


// __host__ void free_global_allocator_combined(){

//   free_global_allocator();
//   free_global_allocator_host();

// }

// __device__ void * global_malloc_combined(uint64_t num_bytes, bool on_host=false){

//   if (on_host){
//     return global_malloc_host(num_bytes);
//   } else {
//     return global_malloc(num_bytes);
//   }

// }

// __device__ void global_free_combined(void * ptr, bool on_host=false){

//   if (on_host){
//     return global_free_host(ptr);
//   } else {
//     return global_free(ptr);
//   }

// }


//fused ops attempt to malloc on device
//then fall back to host on failure
//this allows for a data structure to expand cleanly to host
// this does NOT perform caching - pointers are stable until free is called.
// __device__ void * global_malloc_fused(uint64_t num_bytes){

//   void * alloc = global_malloc(num_bytes);

//   if (alloc == nullptr){
//     return global_malloc_host(num_bytes);
//   }

//   return alloc;

// }

// __device__ void global_free_fused(void * ptr){

//   if (global_gallatin->owns_allocation(ptr)){
//     global_free(ptr);
//   } else {
//     global_free_host(ptr);
//   }


// }






// __host__ void print_global_stats_combined(){

//   printf("Device Allocator:\n");

//   print_global_stats();

//   printf("Host Allocator:\n");

//   print_global_stats_host();


// }


}  // namespace allocators

}  // namespace gallatin

#endif  // End of gallatin