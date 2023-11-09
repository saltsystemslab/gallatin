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


//cudastream wait?
// __device__ void * global_calloc(uint64_t num_bytes){

//   void * alloc = global_malloc(num_bytes);


//   if (num_bytes > 4096){

//     cudaStream_t my_stream;
//     cudaStreamCreateWithFlags(&my_stream, cudaStreamNonBlocking);


//     cudaStreamDestroy(my_stream);
 



//   } else {
//     memset(alloc, 0, num_bytes);
//   }

  

//   return alloc;


// }


__device__ void global_free(void * ptr){

  global_gallatin->free(ptr);

}


__global__ void global_free_kernel(void * allocation, uint64_t num_bytes, uint64_t bytes_per_thread){


  uint64_t max_threads = (num_bytes-1)/bytes_per_thread+1;

  uint64_t tid = gallatin::utils::get_tid();


  if (tid >= max_threads) return;

  memset(allocation + tid*bytes_per_thread, 0, bytes_per_thread);


}

__device__ void global_free_clear(void * ptr){


  //need to determine size


}


__host__ void print_global_stats(){

  global_allocator_type * local_copy;

  cudaMemcpyFromSymbol(&local_copy, global_gallatin, sizeof(global_allocator_type *));

  cudaDeviceSynchronize();

  local_copy->print_info();


}

//writes poison directly before and after the region.
//TODO - add check and fill in extra with poison.
__device__ void * global_malloc_poison(uint64_t bytes_needed){

  //need to promote this to power of 2.
  //if (bytes_needed < 16) bytes_needed = 16;

  //if (bytes_needed % 16 != 0) bytes_needed += (16 - bytes_needed%16);


  //this contains any extra padding
  uint64_t extra_bytes = global_gallatin->get_allocated_size(bytes_needed+32) - (bytes_needed + 32);


  void * alloced_memory = global_gallatin->malloc(bytes_needed+32);

  uint64_t * poison_start = (uint64_t *) alloced_memory;

  uint64_t memory_as_bytes = (uint64_t) alloced_memory;

  uint64_t * poison_end = (uint64_t *) (memory_as_bytes + 16+bytes_needed+extra_bytes);

  char * memory_end = (char *) (memory_as_bytes + 16 + bytes_needed);

  for (uint64_t i = 0; i < extra_bytes; i++){

    memory_end[i] = (char) i;

  }

  atomicExch((unsigned long long int *)&poison_start[0], bytes_needed);

  atomicExch((unsigned long long int *)&poison_start[1], extra_bytes);

  atomicExch((unsigned long long int *)&poison_end[0], bytes_needed);

  atomicExch((unsigned long long int *)&poison_end[1], extra_bytes);

  return (void *) (memory_as_bytes + 16);

}

__device__ bool global_check_poison(void * allocation){

  if (allocation == nullptr){
    printf("Poison violated 0\n");
    return false;
  } 

  uint64_t alloc_as_bytes = (uint64_t) allocation;

  uint64_t * poison_start = (uint64_t *) (alloc_as_bytes-16);

  uint64_t bytes_needed = poison_start[0];

  uint64_t extra_bytes = poison_start[1];


  if ((bytes_needed + extra_bytes) % 16 != 0){
        printf("Poison violated 1\n");
        return false;
  }
  //poison check 1 - 
  // if (bytes_needed < 16 || bytes_needed % 16 != 0){
  //   printf("Poison violated 1\n");
  //   return false;
  // } 


  //get pointer to poison end and check region

  char * check_region = (char *) (alloc_as_bytes + bytes_needed);

  //bytes needed -32 is alloc size.
  uint64_t * poison_end = (uint64_t *) (alloc_as_bytes + bytes_needed + extra_bytes);


  if (poison_end[0] != bytes_needed){
     printf("Poison violated 2\n");
     return false;
  }

  if (poison_end[1] != extra_bytes){
    printf("Poison violated 3\n");
    return;
  } 



  for (uint64_t i = 0; i < extra_bytes; i++){


    char compare_val = (char) i;
    if (check_region[i] != compare_val){

      printf("Poison 4 at %llu bytes after allocation\n", extra_bytes);
      return false;

    } 

  }

  return true;



}


__device__ void global_free_poison(void * allocation){

  global_check_poison(allocation);

  uint64_t alloc_as_bytes = (uint64_t) allocation;

  uint64_t * poison_start = (uint64_t *) (alloc_as_bytes-16);
  
  global_free((void *) poison_start);

}


}  // namespace allocators

}  // namespace gallatin

#endif  // End of gallatin