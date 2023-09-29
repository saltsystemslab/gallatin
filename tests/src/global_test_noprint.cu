/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */





#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace gallatin::allocators;


#if GALLATIN_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 256
#endif


__global__ void alloc_one_size_pointer(uint64_t num_allocs, uint64_t size, uint64_t ** bitarray, uint64_t * misses){


   //introduce some randomness

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   //uint64_t tid = gallatin::utils::get_tid();

   if (tid >= num_allocs) return;

   //printf("Tid %lu counter %lu\n", tid, counter);

   uint64_t * malloc = (uint64_t *) global_malloc(size);

   if (malloc == nullptr){
      atomicAdd((unsigned long long int *)misses, 1ULL);

      bitarray[tid] = malloc;
      return;
   }


   uint64_t old = atomicExch((unsigned long long int *)&bitarray[tid], (unsigned long long int) malloc);

   // if (old != 0){
   //    printf("Two threads swapping to same addr\n");
   // }

   //bitarray[tid] = malloc;

   malloc[0] = tid;

   __threadfence();

   // if (bitarray[tid][0] != tid){
   //    printf("Err detected\n");
   // }

}


__global__ void free_one_size_pointer(uint64_t num_allocs, uint64_t size, uint64_t ** bitarray, uint64_t * block_counts){


   //uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= num_allocs) return;


   uint64_t * malloc = bitarray[tid];

   if (malloc == nullptr) return;


   uint64_t segment = global_gallatin->table->get_segment_from_ptr((void *)malloc);

   uint16_t tree_id = global_gallatin->table->read_tree_id(segment);

   //get the block

   uint64_t offset = global_gallatin->allocation_to_offset((void *)malloc, tree_id);

   uint64_t block_id = offset/4096;

   uint64_t old_count = atomicAdd((unsigned long long int *)&block_counts[block_id], 1ULL);

   // if (old_count >= 4096){
   //    printf("Large count: %llu\n", old_count);
   // }


   if (malloc[0] != tid){


      uint64_t alt_address = malloc[0];

      //printf("Addr: %llx vs %llx\n", (uint64_t) malloc, (uint64_t) bitarray[alt_address]);


      //global_gallatin->check_alloc_valid((void *)malloc);

      uint64_t miss_amount;
      if (tid >= malloc[0]){
         miss_amount = tid-malloc[0];
      } else {
         miss_amount = malloc[0] - tid;
      }

      Block * block_ptr = global_gallatin->table->get_block_from_global_block_id(block_id);

      Block * block_before = global_gallatin->table->get_block_from_global_block_id(block_id-1);

      printf("Block %llu stats: malloc %u free %u - block before %u - %u\n", block_id, block_ptr->malloc_counter, block_ptr->free_counter, block_before->malloc_counter, block_before->free_counter);

      printf("Double malloc %lu vs %lu - diff is %lu, tree %u\n", tid, malloc[0], miss_amount, tree_id);
      return;
   }

   //global_free(malloc);

   __threadfence();


}


__global__ void check_blocks(uint64_t * blocks, uint64_t nblocks){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nblocks) return;

   uint64_t my_block_count = blocks[tid];

   if (my_block_count > 4096){

      Block * my_block = global_gallatin->table->get_block_from_global_block_id(tid);

      Block * prev_block = global_gallatin->table->get_block_from_global_block_id(tid-1);

      Block * next_block = global_gallatin->table->get_block_from_global_block_id(tid+1);


      printf("Block %lu count %lu: <malloc %u free %u> previous %lu next %lu\n", tid, my_block_count, my_block->malloc_counter, my_block->free_counter, blocks[tid-1], blocks[tid+1]);
   }


}


//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
//works on actual pointers instead of uint64_t
//The correctness check is done by treating each allocation as a uint64_t and writing the tid
// if TID is not what is expected, we know that a double malloc has occurred.
__host__ void gallatin_test_allocs_pointer(uint64_t num_bytes, int num_rounds, uint64_t size){


   gallatin::utils::timer boot_timing;

   uint64_t mem_segment_size = 16ULL*1024*1024;

   uint64_t num_segments = gallatin::utils::get_max_chunks<16ULL*1024*1024>(num_bytes);

   uint64_t max_allocs_per_segment = mem_segment_size/16;

   uint64_t allocs_per_segment_size = mem_segment_size/size;

   if (allocs_per_segment_size >= max_allocs_per_segment) allocs_per_segment_size = max_allocs_per_segment;

   uint64_t num_allocs = allocs_per_segment_size*num_segments;

   uint64_t blocks_per_segment = max_allocs_per_segment/4096;

   uint64_t total_num_blocks = blocks_per_segment*num_segments;

   // printf("Starting test with %lu segments, %lu allocs per segment\n", num_segments, max_allocs_per_segment);
   // printf("Actual allocs per segment %lu total allocs %lu\n", allocs_per_segment_size, num_allocs);

   init_global_allocator(num_bytes, 42, false);


   //generate bitarry
   //space reserved is one 
   uint64_t ** bits;
   GPUErrorCheck(cudaMalloc((void **)&bits, sizeof(uint64_t *)*num_allocs));

   GPUErrorCheck(cudaMemset(bits, 0, sizeof(uint64_t *)*num_allocs));


   uint64_t * misses;
   GPUErrorCheck(cudaMallocManaged((void **)&misses, sizeof(uint64_t)));


   uint64_t * block_counts;

   GPUErrorCheck(cudaMalloc((void **)&block_counts, sizeof(uint64_t)*total_num_blocks));

   GPUErrorCheck(cudaMemset(block_counts, 0, sizeof(uint64_t)*total_num_blocks));

   GPUErrorCheck(cudaDeviceSynchronize());

   misses[0] = 0;


   uint64_t total_misses = 0;




   //std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;

   for (int i = 0; i < num_rounds; i++){

      //printf("Starting Round %d/%d\n", i, num_rounds);

      gallatin::utils::timer kernel_timing;
      alloc_one_size_pointer<<<(num_allocs-1)/TEST_BLOCK_SIZE+1,TEST_BLOCK_SIZE>>>(.9*num_allocs, size, bits, misses);
      kernel_timing.sync_end();

      gallatin::utils::timer free_timing;
      free_one_size_pointer<<<(num_allocs-1)/TEST_BLOCK_SIZE+1,TEST_BLOCK_SIZE>>>(.9*num_allocs, size, bits, block_counts);
      free_timing.sync_end();


      check_blocks<<<(total_num_blocks-1)/256+1, 256>>>(block_counts, total_num_blocks);

      cudaDeviceSynchronize();

      //kernel_timing.print_throughput("Malloced", .9*num_allocs);

      //free_timing.print_throughput("Freed", .9*num_allocs);

      printf("Missed: %lu\n", misses[0]);

      cudaDeviceSynchronize();

      total_misses += misses[0];

      misses[0] = 0;

      //print_global_stats();


   }

   //printf("Total missed across %d runs: %lu/%lu\n", num_rounds, total_misses, num_allocs*num_rounds);

   //print_global_stats();

   cudaFree(misses);

   cudaFree(bits);

   free_global_allocator();


}



int main(int argc, char** argv) {

   uint64_t num_segments;

   int num_rounds = 1;
   
   uint64_t size;

   if (argc < 2){
      num_segments = 100;
   } else {
      num_segments = std::stoull(argv[1]);
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

   gallatin_test_allocs_pointer(num_segments*16*1024*1024, num_rounds, size);



   cudaDeviceReset();
   return 0;

}
