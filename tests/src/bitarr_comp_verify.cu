/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */


#define DEBUG_PRINTS 0


#include <poggers/allocators/uint64_bitarray.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;




using namespace poggers::allocators;

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void one_thread_init(alloc_bitarr * allocations, storage_bitmap * storage){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0)return;

   allocations[0].init();

   storage[0].init();
   return;

}

__global__ void one_team_test_bitarr(alloc_bitarr * allocations, storage_bitmap * storage){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= 32) return;

   void * allocation;
   uint64_t remainder;
   void * remainder_offset;
   bool is_leader;

   if (tid == 0){
      printf("Starting\n");
   }

   cg::coalesced_group full_team = cg::coalesced_threads();

   allocations[0].bit_malloc(allocation, remainder, remainder_offset, is_leader);

   if (is_leader){
      printf("%llu has %d remaining, they look like %llx.\n", tid, __popcll(remainder), remainder);
      bool result = storage[0].attach_buffer(remainder_offset, remainder);
      printf("%d: New storage %llx, old storage %llx\n", result, storage[0].manager_bits, allocations[0].manager_bits);

   }

   full_team.sync();

   if (tid >= 8) return;

   //small team entering bit malloc

   cg::coalesced_group small_team = cg::coalesced_threads();

   for (int i = 0; i < 4; i++){

      void * second_alloc;

      bool result = storage[0].bit_malloc(second_alloc);

      if (!result){
         printf("%llu failed iteration %d\n", tid, i);
      }

      small_team.sync();

   }




}


__global__ void init_all_bitmaps(alloc_bitarr * allocators, storage_bitmap * storages, uint64_t num_allocators){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >=num_allocators) return;

   allocators[tid].init();
   storages[tid].init();

}

__global__ void full_multi_team_test(alloc_bitarr * allocators, storage_bitmap * storages, uint64_t num_allocators){

   uint64_t blockID = blockIdx.x;

   if (blockID >= num_allocators) return;

   void * my_allocation;
   bool found = alloc_with_locks(my_allocation, &allocators[blockID], &storages[blockID]);

   if (!found){
      uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;
      printf("%llu/%llu not found!\n", tid, blockID);
   }

}




__global__ void full_team_test_bitarr(alloc_bitarr * allocator, storage_bitmap * storage){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   void * allocation;

   bool found = alloc_with_locks(allocation, allocator, storage);

   if (!found){
      printf("%llu could not find!\n", tid);
   }

}

__host__ void build_bitarr_test_split(uint64_t num_threads){



   alloc_bitarr * allocations;

   cudaMalloc((void **)&allocations, sizeof(alloc_bitarr));


   storage_bitmap * storage;

   cudaMalloc((void **)&storage, sizeof(storage_bitmap));

   one_thread_init<<<1,1>>>(allocations, storage);

   full_team_test_bitarr<<<1,num_threads>>>(allocations, storage);

   cudaDeviceSynchronize();


   cudaFree(storage);
   cudaFree(allocations);

   printf("Passed test %llu\n", num_threads);




}

__host__ void multi_team_host_wrapper(uint64_t num_blocks){



   alloc_bitarr * allocations;

   cudaMalloc((void **)&allocations, sizeof(alloc_bitarr)*num_blocks);


   storage_bitmap * storage;

   cudaMalloc((void **)&storage, sizeof(storage_bitmap)*num_blocks);


   init_all_bitmaps<<<(num_blocks-1)/512+1,512>>>(allocations, storage, num_blocks);


   cudaDeviceSynchronize();

   auto insert_start = std::chrono::high_resolution_clock::now();

   full_multi_team_test<<<num_blocks, 512>>>(allocations, storage, num_blocks);

   cudaDeviceSynchronize();

   auto insert_end = std::chrono::high_resolution_clock::now();


   cudaFree(storage);
   cudaFree(allocations);

   std::chrono::duration<double> time_diff = insert_end-insert_start;


   printf("%llu with Inserts: %f\n", num_blocks, 1.0*num_blocks*512/time_diff.count());
   

   //printf("Passed test %llu, took \n", num_blocks);




}

int main(int argc, char** argv) {

   // build_bitarr_test_split(32);

   // build_bitarr_test_split(64);

   // build_bitarr_test_split(128);

   // build_bitarr_test_split(256);

   // build_bitarr_test_split(512);

   for (int i = 0; i < 25; i++){

      multi_team_host_wrapper((1ULL) << i);
      
   }
   // build_bitarr_test_split(1024);

   // build_bitarr_test_split(2048);

	return 0;

}
