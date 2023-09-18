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


using global_allocator_type = gallatin::allocators::Gallatin<16ULL*1024*1024, 16ULL, 4096ULL>;

__device__ global_allocator_type * global_gallatin;


__host__ void init_global_allocator(uint64_t num_bytes, uint64_t seed, bool print_info=true){

  global_allocator_type * local_copy = global_allocator_type::generate_on_device(num_bytes, seed, print_info);

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


}  // namespace allocators

}  // namespace gallatin

#endif  // End of gallatin