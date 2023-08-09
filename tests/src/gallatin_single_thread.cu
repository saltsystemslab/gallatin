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


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace beta::allocators;


// __global__ void test_kernel(veb_tree * tree, uint64_t num_removes, int num_iterations){


//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    if (tid >= num_removes)return;


//       //printf("Tid %lu\n", tid);


//    for (int i=0; i< num_iterations; i++){


//       if (!tree->remove(tid)){
//          printf("BUG\n");
//       }

//       tree->insert(tid);

//    }


#if BETA_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 256
#endif



template<typename allocator_type>
__global__ void alloc_one_size_pointer(allocator_type * allocator, uint64_t num_allocs, uint64_t size, uint64_t ** bitarray, uint64_t * misses){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid !=0) return;


   for (uint64_t i=0; i < num_allocs; i++){

      uint64_t * malloc = (uint64_t *) allocator->malloc(size);


      if (malloc == nullptr){
         atomicAdd((unsigned long long int *)misses, 1ULL);
      }

      bitarray[i] = malloc;

      malloc[0] = i;

      __threadfence();


   }

}


template<typename allocator_type>
__global__ void free_one_size_pointer(allocator_type * allocator, uint64_t num_allocs, uint64_t size, uint64_t ** bitarray){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;


   for (uint64_t i=0; i < num_allocs; i++){

      uint64_t * malloc = bitarray[i];

      if (malloc == nullptr) continue;


      if (malloc[0] != i){
         printf("Double malloc on index %llu: read address is %llu\n", i, malloc[0]);
      }

      allocator->free(malloc);

      __threadfence();

   } 


}


//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
//works on actual pointers instead of uint64_t
//The correctness check is done by treating each allocation as a uint64_t and writing the tid
// if TID is not what is expected, we know that a double malloc has occurred.
template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void gallatin_test_single_thread(uint64_t num_allocs, uint64_t num_rounds, uint64_t size){


   uint64_t num_bytes = 16ULL*1024*1024*1000;

   beta::utils::timer boot_timing;

   using betta_type = beta::allocators::beta_allocator<mem_segment_size, smallest, largest>;

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);


   //generate bitarry
   //space reserved is one 
   uint64_t ** bits;
   cudaMalloc((void **)&bits, sizeof(uint64_t *)*num_allocs);

   cudaMemset(bits, 0, sizeof(uint64_t *)*num_allocs);


   uint64_t * misses;
   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   cudaDeviceSynchronize();

   misses[0] = 0;




   std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;

   for (int i = 0; i < num_rounds; i++){

      printf("Starting Round %d/%d\n", i, num_rounds);

      beta::utils::timer kernel_timing;
      alloc_one_size_pointer<betta_type><<<1,1>>>(allocator, num_allocs, size, bits, misses);
      kernel_timing.sync_end();

      beta::utils::timer free_timing;
      free_one_size_pointer<betta_type><<<1,1>>>(allocator, num_allocs, size, bits);
      free_timing.sync_end();

      kernel_timing.print_throughput("Malloced", num_allocs);

      free_timing.print_throughput("Freed", num_allocs);

      printf("Missed: %llu\n", misses[0]);

      cudaDeviceSynchronize();

      misses[0] = 0;

      allocator->print_info();

   }

   cudaFree(misses);

   cudaFree(bits);

   betta_type::free_on_device(allocator);


}


int main(int argc, char** argv) {

   uint64_t num_allocs;

   int num_rounds;
   
   uint64_t size;

   if (argc < 2){
      num_allocs = 100;
   } else {
      num_allocs = std::stoull(argv[1]);
   }


   if (argc < 3){
      num_rounds = 1;
   } else {
      num_rounds = std::stoull(argv[2]);
   }


   if (argc < 4){
      size = 16;
   } else {
      size = std::stoull(argv[3]);
   }


   gallatin_test_single_thread<16ULL*1024*1024, 16ULL, 4096ULL>(num_allocs, num_rounds, size);


   cudaDeviceReset();
   return 0;

}
