/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <poggers/beta/block.cuh>
#include <poggers/beta/thread_storage.cuh>
#include <poggers/beta/alloc_with_locks.cuh>

#include <poggers/beta/allocator_context.cuh>

#include <poggers/beta/timer.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/beta/timer.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>

#define BLOCK_SIZE 512

using namespace beta::allocators;

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void read_random_uints_throughput(uint64_t * uints, uint64_t num_uints, uint64_t num_rounds, uint64_t * counters){

   uint64_t tid = poggers::utils::get_tid();

   uint64_t my_hash = tid;

   if (tid >= num_uints) return;

   poggers::hashers::murmurHasher<uint64_t> my_hasher;

   my_hasher.init(13);

   for (int i = 0; i < num_rounds; i++){

      my_hash = my_hasher.hash(my_hash);

      uint64_t read = uints[my_hash % num_uints];

      if (read != 0){
         printf("Stuff\n");
      }


   }


}


__global__ void read_random_uints_latency(uint64_t * uints, uint64_t num_uints, uint64_t num_rounds, uint64_t * counters){

   uint64_t tid = poggers::utils::get_tid();

   uint64_t my_hash = tid;

   if (tid >= num_uints) return;

   poggers::hashers::murmurHasher<uint64_t> my_hasher;

   my_hasher.init(13);

   for (int i = 0; i < num_rounds; i++){

      my_hash = my_hasher.hash(my_hash);

      uint64_t clock_start = poggers::utils::get_clock_time();

      uint64_t read = uints[my_hash % num_uints];

      uint64_t clock_end = poggers::utils::get_clock_time();

      uint64_t cycles = clock_end - clock_start;

      atomicAdd((unsigned long long int *)&counters[0], (unsigned long long int ) cycles);

      atomicMax((unsigned long long int *)&counters[1], (unsigned long long int ) cycles);

      atomicMin((unsigned long long int *)&counters[2], (unsigned long long int ) cycles);

      if (read != 0){
         printf("Stuff\n");
      }

   }


}


__global__ void read_ordered_uints_throughput(uint64_t * uints, uint64_t num_uints, uint64_t * counters){

   uint64_t tid = poggers::utils::get_tid();


   if (tid >= num_uints) return;

   uint64_t read = uints[tid];

   if (read != 0){
         printf("Stuff\n");
   }



}

__global__ void read_ordered_uints_latency(uint64_t * uints, uint64_t num_uints, uint64_t * counters){

   uint64_t tid = poggers::utils::get_tid();



   if (tid >= num_uints) return;

   uint64_t clock_start = poggers::utils::get_clock_time();

   uint64_t read = uints[tid];

   uint64_t clock_end = poggers::utils::get_clock_time();

   uint64_t cycles = clock_end - clock_start;

   atomicAdd((unsigned long long int *)&counters[0], (unsigned long long int ) cycles);

   atomicMax((unsigned long long int *)&counters[1], (unsigned long long int ) cycles);

   atomicMin((unsigned long long int *)&counters[2], (unsigned long long int ) cycles);


   if (read != 0){
         printf("Stuff\n");
   }

}

//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {





   uint64_t num_uints;

   uint64_t num_rounds;

   bool measure_latency = false;
   

   if (argc < 2){
      num_uints = 40ULL*1024*1024*1024/64;
   } else {
      num_uints = std::stoull(argv[1]);
   }


   if (argc < 3){
      num_rounds = 10;
   } else {
      num_rounds = std::stoull(argv[2]);
   }

   if (argc < 4){
      measure_latency = false;
   } else {
      measure_latency = true;
   }


   
   uint64_t * bits;

   cudaMalloc((void **)&bits, sizeof(uint64_t)*num_uints);

   cudaMemset(bits, 0, sizeof(uint64_t)*num_uints);


   uint64_t * counters;

   cudaMallocManaged((void **)&counters, sizeof(uint64_t)*3);

   cudaDeviceSynchronize();

   counters[0] = 0;
   counters[1] = 0;
   counters[2] = ~0ULL;

   cudaDeviceSynchronize();

   if (measure_latency){
      printf("Latency test with %llu uints, %llu rounds\n", num_uints, num_rounds);
   } else {
      printf("Throughput test with %llu uints, %llu rounds\n", num_uints, num_rounds);
   }
   


   beta::utils::timer throughput_timer;
   if (measure_latency){
      read_random_uints_latency<<<(num_uints-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(bits, num_uints, num_rounds, counters);
   } else {
      read_random_uints_throughput<<<(num_uints-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(bits, num_uints, num_rounds, counters);
   }

   throughput_timer.sync_end();

   throughput_timer.print_throughput("Accessed", num_uints*num_rounds);

   if (measure_latency){
      printf("Avg: %llu Max: %llu, min: %llu\n", counters[0]/(num_uints*num_rounds), counters[1], counters[2]);
   }



   cudaDeviceSynchronize();


   counters[0] = 0;
   counters[1] = 0;
   counters[2] = ~0ULL;

   cudaDeviceSynchronize();

   beta::utils::timer ordered_timer;


   if (measure_latency){
       read_ordered_uints_latency<<<(num_uints-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(bits, num_uints, counters);
   } else {
       read_ordered_uints_throughput<<<(num_uints-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(bits, num_uints, counters);
   }
   ordered_timer.sync_end();

   throughput_timer.print_throughput("Accessed Ordered", num_uints);

   if (measure_latency){
      printf("Avg: %llu Max: %llu, min: %llu\n", counters[0]/(num_uints), counters[1], counters[2]);
   }




 

   cudaDeviceReset();
   return 0;

}
