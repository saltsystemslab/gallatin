/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <poggers/counter_blocks/block.cuh>
#include <poggers/beta/timer.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>



//since blocks can be split to different thread storages, 2048 is the only safe val
//64 pull guaranteed.
#define ALLOCS_PER_BLOCK 4096



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

//Functions above test throughput.
//lets test accuracy! Boot up a bitarray and set it

__global__ void init_blocks(Block * blocks, uint64_t num_blocks){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_blocks) return;

   blocks[tid].init();

}


__global__ void test_block_correctness_local(Block * blocks, uint64_t * bitarray, uint64_t num_allocs){


   uint64_t tid = poggers::utils::get_tid();

   if (tid >= num_allocs) return;

   uint64_t my_block = tid/ALLOCS_PER_BLOCK;

   cg::coalesced_group my_team = cg::coalesced_threads();

   uint64_t malloc = blocks[my_block].block_malloc(my_team);

   if (malloc == ~0ULL){
      printf("Allocation error\n");
      return;
   }

   malloc += my_block*4096;


   uint64_t high = malloc/64;

   uint64_t low = malloc % 64;

   auto bitmask = SET_BIT_MASK(low);

   uint64_t bits = atomicOr((unsigned long long int *) &bitarray[high], (unsigned long long int) bitmask);

   if (bits & bitmask){
      printf("Double alloc! tid: %llu block %llu, %llu\n", tid, my_block, malloc);
   }



   //printf("Tid %llu done\n", tid);


}

__host__ void test_correctness_local(uint64_t num_blocks){

   Block * blocks;

   uint64_t num_allocs = num_blocks*ALLOCS_PER_BLOCK;

   cudaMalloc((void **)&blocks, sizeof(Block)*num_blocks);

   init_blocks<<<(num_blocks-1)/256+1, 256>>>(blocks, num_blocks);

   //4096 bits per block.
   uint64_t * bitarray;
   cudaMalloc((void **)&bitarray, sizeof(uint64_t)*num_blocks*64);

   cudaMemset(bitarray, 0ULL, sizeof(uint64_t)*num_blocks*64);


   cudaDeviceSynchronize();


   beta::utils::timer block_timing;
   test_block_correctness_local<<<(num_allocs-1)/256+1, 256>>>(blocks, bitarray, num_allocs);
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



   printf("Correctness test\n");
   test_correctness_local(num_blocks);

   cudaDeviceReset();
   return 0;

}
