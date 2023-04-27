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

#include <stdio.h>
#include <iostream>
#include <assert.h>



//since blocks can be split to different thread storages, 2048 is the only safe val
//64 pull guaranteed.
#define ALLOCS_PER_BLOCK 2048



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

using namespace beta::allocators;


__global__ void alloc_all_blocks(block * blocks, uint64_t num_allocs){


   uint64_t tid = poggers::utils::get_tid();

   if (tid >= num_allocs) return;


   uint64_t blockID = tid/ALLOCS_PER_BLOCK;


   cg::coalesced_group my_team = cg::coalesced_threads();


   uint64_t remainder;

   uint64_t malloc = blocks[blockID].block_malloc(my_team, remainder);




}


__global__ void alloc_all_blocks_storage(block * blocks, pinned_thread_storage * thread_storages, uint64_t num_allocs){

   uint64_t tid = poggers::utils::get_tid();

   if (tid >= num_allocs) return;

   auto team_warp_lock = thread_storages->get_warp_lock();

   auto my_thread_storage = thread_storages->get_thread_storage();

   uint64_t my_block = tid/ALLOCS_PER_BLOCK;

   //cg::coalesced_group my_team = cg::coalesced_threads();

   uint64_t malloc = alloc_with_locks(team_warp_lock, my_block, &blocks[my_block], my_thread_storage);

   if (malloc == ~0ULL) printf("Allocation error\n");


   //printf("Tid %llu done\n", tid);


}

__global__ void alloc_all_blocks_local(block * blocks, pinned_thread_storage * thread_storages, uint64_t num_allocs){

   __shared__ warp_lock team_warp_lock;

   team_warp_lock.init();

   __syncthreads();

   uint64_t tid = poggers::utils::get_tid();

   if (tid >= num_allocs) return;

   //auto team_warp_lock = thread_storages->get_warp_lock();

   auto my_thread_storage = thread_storages->get_thread_storage();

   uint64_t my_block = tid/ALLOCS_PER_BLOCK;

   //cg::coalesced_group my_team = cg::coalesced_threads();

   uint64_t malloc = alloc_with_locks(&team_warp_lock, my_block, &blocks[my_block], my_thread_storage);

   if (malloc == ~0ULL) printf("Allocation error\n");


   //printf("Tid %llu done\n", tid);


}


__global__ void init_blocks(block * blocks, uint64_t num_blocks){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_blocks) return;

   blocks[tid].init();

}


__host__ void init_and_test_blocks(uint64_t num_blocks){

   block * blocks;

   uint64_t num_allocs = num_blocks*ALLOCS_PER_BLOCK;

   cudaMalloc((void **)&blocks, sizeof(block)*num_blocks);

   init_blocks<<<(num_blocks-1)/256+1, 256>>>(blocks, num_blocks);

   alloc_all_blocks<<<(num_allocs-1)/256+1, 256>>>(blocks, num_allocs);

}


__host__ void init_and_test_blocks_storage(uint64_t num_blocks){

   block * blocks;

   uint64_t num_allocs = num_blocks*ALLOCS_PER_BLOCK;

   cudaMalloc((void **)&blocks, sizeof(block)*num_blocks);

   init_blocks<<<(num_blocks-1)/256+1, 256>>>(blocks, num_blocks);

   auto thread_storage = pinned_thread_storage::generate_on_device();

   beta::utils::timer block_timing;
   alloc_all_blocks_storage<<<(num_allocs-1)/256+1, 256>>>(blocks, thread_storage, num_allocs);
   auto duration = block_timing.sync_end();

   std::cout << "Alloced " << num_allocs << " in " << duration << " seconds, throughput " << std::fixed << 1.0*num_allocs/duration << std::endl;   


}

__host__ void init_and_test_blocks_lock_local(uint64_t num_blocks){

   block * blocks;

   uint64_t num_allocs = num_blocks*ALLOCS_PER_BLOCK;

   cudaMalloc((void **)&blocks, sizeof(block)*num_blocks);

   init_blocks<<<(num_blocks-1)/256+1, 256>>>(blocks, num_blocks);

   auto thread_storage = pinned_thread_storage::generate_on_device();

   beta::utils::timer block_timing;
   alloc_all_blocks_local<<<(num_allocs-1)/256+1, 256>>>(blocks, thread_storage, num_allocs);
   auto duration = block_timing.sync_end();

   std::cout << "Alloced " << num_allocs << " in " << duration << " seconds, throughput " << std::fixed << 1.0*num_allocs/duration << std::endl;   


}


//Functions above test throughput.
//lets test accuracy! Boot up a bitarray and set it


__global__ void test_block_correctness_local(block * blocks, pinned_thread_storage * thread_storages, uint64_t * bitarray, uint64_t num_allocs){

   __shared__ warp_lock team_warp_lock;

   team_warp_lock.init();

   __syncthreads();

   uint64_t tid = poggers::utils::get_tid();

   if (tid >= num_allocs) return;

   //auto team_warp_lock = thread_storages->get_warp_lock();

   auto my_thread_storage = thread_storages->get_thread_storage();

   uint64_t my_block = tid/ALLOCS_PER_BLOCK;

   //cg::coalesced_group my_team = cg::coalesced_threads();

   uint64_t malloc = alloc_with_locks(&team_warp_lock, my_block, &blocks[my_block], my_thread_storage);

   if (malloc == ~0ULL){
      printf("Allocation error\n");
      return;
   }


   uint64_t high = malloc/64;

   uint64_t low = malloc % 64;

   auto bitmask = SET_BIT_MASK(low);

   uint64_t bits = atomicOr((unsigned long long int *) &bitarray[high], (unsigned long long int) bitmask);

   if (bits & bitmask){
      printf("Double alloc! tid: %llu block %llu, %llu\n", tid, my_block, malloc);
   }

   //printf("Tid %llu done\n", tid);


}


__global__ void test_block_correctness_context(block * blocks, pinned_thread_storage * thread_storages, uint64_t * bitarray, uint64_t num_allocs){

   __shared__ context local_context;

   local_context.open_context(thread_storages);

   //scoped_context temp_context(&local_context);

   __syncthreads();


   //printf("Entering scope\n");


   uint64_t tid = poggers::utils::get_tid();

   if (tid >= num_allocs) return;

   //auto team_warp_lock = thread_storages->get_warp_lock();

   //auto my_thread_storage = thread_storages->get_thread_storage();

   uint64_t my_block = tid/ALLOCS_PER_BLOCK;

   //cg::coalesced_group my_team = cg::coalesced_threads();

   uint64_t malloc = alloc_with_locks(local_context.get_local_lock(), my_block, &blocks[my_block], local_context.get_local_storage());

   if (malloc == ~0ULL){
      printf("Allocation error\n");

      //local_context.close_context(thread_storages->get_thread_storage());
      return;
   }


   uint64_t high = malloc/64;

   uint64_t low = malloc % 64;

   auto bitmask = SET_BIT_MASK(low);

   uint64_t bits = atomicOr((unsigned long long int *) &bitarray[high], (unsigned long long int) bitmask);

   if (bits & bitmask){
      printf("Double alloc! tid: %llu block %llu, %llu\n", tid, my_block, malloc);
   }

   //printf("Tid %llu done\n", tid);

   //local_context.close_context(thread_storages->get_thread_storage());


}

__host__ void test_correctness_local(uint64_t num_blocks){

   block * blocks;

   uint64_t num_allocs = num_blocks*ALLOCS_PER_BLOCK;

   cudaMalloc((void **)&blocks, sizeof(block)*num_blocks);

   init_blocks<<<(num_blocks-1)/256+1, 256>>>(blocks, num_blocks);

   auto thread_storage = pinned_thread_storage::generate_on_device();

   //4096 bits per block.
   uint64_t * bitarray;
   cudaMalloc((void **)&bitarray, sizeof(uint64_t)*num_blocks*64);

   cudaMemset(bitarray, 0ULL, sizeof(uint64_t)*num_blocks*64);


   cudaDeviceSynchronize();


   beta::utils::timer block_timing;
   test_block_correctness_local<<<(num_allocs-1)/256+1, 256>>>(blocks, thread_storage, bitarray, num_allocs);
   auto duration = block_timing.sync_end();

   std::cout << "Alloced " << num_allocs << " in " << duration << " seconds, throughput " << std::fixed << 1.0*num_allocs/duration << std::endl;   


}


//test correctness when using allocator contexts
__host__ void test_correctness_context(uint64_t num_blocks){

   block * blocks;

   uint64_t num_allocs = num_blocks*ALLOCS_PER_BLOCK;

   cudaMalloc((void **)&blocks, sizeof(block)*num_blocks);

   init_blocks<<<(num_blocks-1)/256+1, 256>>>(blocks, num_blocks);

   auto thread_storage = pinned_thread_storage::generate_on_device();

   //4096 bits per block.
   uint64_t * bitarray;
   cudaMalloc((void **)&bitarray, sizeof(uint64_t)*num_blocks*64);

   cudaMemset(bitarray, 0ULL, sizeof(uint64_t)*num_blocks*64);


   cudaDeviceSynchronize();


   beta::utils::timer block_timing;
   test_block_correctness_context<<<(num_allocs-1)/256+1, 256>>>(blocks, thread_storage, bitarray, num_allocs);
   auto duration = block_timing.sync_end();

   std::cout << "Alloced " << num_allocs << " in " << duration << " seconds, throughput " << std::fixed << 1.0*num_allocs/duration << std::endl;   


}


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   uint64_t num_blocks;
   

   if (argc < 2){
      num_blocks = 100;
   } else {
      num_blocks = std::stoull(argv[1]);
   }

   
   printf("Storage lock test with %llu blocks\n", num_blocks);
   init_and_test_blocks_storage(num_blocks);


   printf("local lock test with %llu blocks\n", num_blocks);
   init_and_test_blocks_lock_local(num_blocks);


   printf("Correctness test\n");
   test_correctness_local(num_blocks);

   printf("Local test\n");
   test_correctness_context(num_blocks);

   cudaDeviceReset();
   return 0;

}
