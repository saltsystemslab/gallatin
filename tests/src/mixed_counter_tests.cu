/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/counter_blocks/mixed_counter.cuh>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/counter_blocks/timer.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace gallatin::utils;

using timer = beta::utils::timer;

//test the performance of the mixed counter and verify the atomic precondition: 
// Should only give out a set of unique indices while letting a set CAP come through.

__global__ void mixed_counter_tests(mixed_counter * shared_counter, uint64_t num_threads, uint64_t * bitarray, uint64_t live_cap){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_threads) return;

   
   while (true){

      uint64_t index = shared_counter->count_and_increment(live_cap);

      //valid
      if (index != ~0ULL){

         uint64_t high = index / 64;

         uint64_t low = index % 64;

         auto bitmask = SET_BIT_MASK(low);

         //printf("Index %llu\n", index);

         uint64_t bits = atomicOr((unsigned long long int *) &bitarray[high], (unsigned long long int) bitmask);

         if (bits & bitmask){
            printf("Double offset bug in address %llu\n", index);
         }

         //at the end, drop lower counter while adding to queue_offset
         shared_counter->release();

         return;

      }

   }



}


int main(int argc, char** argv) {

   uint64_t num_threads;

   uint64_t live_cap;

   if (argc < 2){
      num_threads = 10000;
   } else {
      num_threads = std::stoull(argv[1]);
   }

   if (argc < 3){
      live_cap = 32;
   } else {
      live_cap = std::stoull(argv[2]);
   }

   mixed_counter * counter;

   cudaMallocManaged((void **)&counter, sizeof(counter));

   counter->init();

   uint64_t num_bytes_bitarray = sizeof(uint64_t) * ((num_threads -1)/64+1);

   uint64_t * bits;

   cudaMalloc((void **)&bits, num_bytes_bitarray);

   cudaMemset(bits, 0, num_bytes_bitarray);



   cudaDeviceSynchronize();


   timer acq_timer;
   mixed_counter_tests<<<(num_threads-1)/512+1,512>>>(counter, num_threads, bits, live_cap);

   acq_timer.sync_end();

   acq_timer.print_throughput("Locked", num_threads);


   cudaFree(bits);
   cudaFree(counter);
 
   cudaDeviceReset();
   return 0;

}
