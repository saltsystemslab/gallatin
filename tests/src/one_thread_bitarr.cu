/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/bitarray.cuh>

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


__global__ void malloc_test_kernel_prefetching(bitarr_grouped<4> * global_bitarray, storage_bitmap<4> * local_bitmaps, uint64_t max_mallocs, uint64_t mallocs_per_thread){


   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid >=max_mallocs) return;


   int my_core = poggers::utils::get_smid();

   //printf("%d\n", my_core);

   storage_bitmap<4> * my_bitmap = &local_bitmaps[my_core];

   //storage_bitmap<4> * my_bitmap = storage_bitmap<4>::get_my_bitmap(local_bitmaps);


   uint64_t blockID = blockIdx.x;

   // if (threadIdx.x == 0){
   //    printf("%llu\n", blockID);
   // }


   for (int i =0; i < mallocs_per_thread; i++){



      void * my_allocation = my_bitmap->malloc_from_existing();
      //void * my_allocation = nullptr;

      if (my_allocation) continue;


      // __syncthreads();

      // assert(my_bitmap->check_attachment() == 0ULL);

      // __syncthreads();

      //doesn't need
      //my_allocation = nullptr;
      bool should_preempt = false;
      uint64_t * ext_address = nullptr;
      uint64_t remaining_metadata = 0;



      //grouped threads is the local group!
      //need the bigga one
      cg::coalesced_group grouped_threads = global_bitarray[blockID].metadata_malloc(my_allocation, should_preempt, ext_address, remaining_metadata);


      // __syncthreads();

      // assert(my_bitmap->check_attachment() == 0ULL);

      // __syncthreads();




   if (grouped_threads.thread_rank() == 0 && remaining_metadata != 0ULL){

         //printf("Size: %d, popcount: %d\n", grouped_threads.size(), __popcll(remaining_metadata));

         my_bitmap->attach_buffer(ext_address, remaining_metadata);

      }

   }


   if (tid == 0){
      printf("%llx %llx\n", global_bitarray[blockID].check_attachment(), my_bitmap->check_attachment());
   }

}

__global__ void malloc_test_kernel_split_local(bitarr_grouped<4> * global_bitarray, storage_bitmap<4> * local_bitmaps, uint64_t max_mallocs, uint64_t mallocs_per_thread){


   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid >=max_mallocs) return;


   int my_core = poggers::utils::get_smid();

   //printf("%d\n", my_core);



   //storage_bitmap<4> * my_bitmap = storage_bitmap<4>::get_my_bitmap(local_bitmaps);


   uint64_t blockID = blockIdx.x;

   storage_bitmap<4> * my_bitmap = &local_bitmaps[blockID];
   // if (threadIdx.x == 0){
   //    printf("%llu\n", blockID);
   // }


   for (int i =0; i < mallocs_per_thread; i++){



      void * my_allocation = my_bitmap->malloc_from_existing();
      //void * my_allocation = nullptr;

      if (my_allocation) continue;


      // __syncthreads();

      // assert(my_bitmap->check_attachment() == 0ULL);

      // __syncthreads();

      //doesn't need
      //my_allocation = nullptr;
      bool should_preempt = false;
      uint64_t * ext_address = nullptr;
      uint64_t remaining_metadata = 0;



      //grouped threads is the local group!
      //need the bigga one
      cg::coalesced_group grouped_threads = global_bitarray[blockID].metadata_malloc(my_allocation, should_preempt, ext_address, remaining_metadata);


      // __syncthreads();

      // assert(my_bitmap->check_attachment() == 0ULL);

      // __syncthreads();




   if (grouped_threads.thread_rank() == 0 && remaining_metadata != 0ULL){

         //printf("Size: %d, popcount: %d\n", grouped_threads.size(), __popcll(remaining_metadata));

         my_bitmap->attach_buffer(ext_address, remaining_metadata);

      }

   }


   if (tid == 0){
      printf("%llx %llx\n", global_bitarray[blockID].check_attachment(), my_bitmap->check_attachment());
   }

}

__global__ void malloc_test_correctness_kernel(bitarr_grouped<4> * global_bitarray, uint64_t * counters, uint64_t max_mallocs, uint64_t mallocs_per_thread){


   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid >= max_mallocs) return;

   uint64_t blockID = blockIdx.x;

   uint64_t my_val = global_bitarray[blockID].malloc(blockID);

   uint64_t old_counter = atomicAdd((unsigned long long int *)&counters[my_val], 1ULL);

   assert(old_counter == 0);

}


__global__ void malloc_init_kernel(bitarr_grouped<4> * global_bitarray, uint64_t num_bit_arr){

   uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

   if (tid >= num_bit_arr) return;

   global_bitarray[tid].init();

}


__host__ void build_bitarr_test_prefetch(uint64_t max_mallocs, uint64_t mallocs_per_thread, uint64_t block_size){


   uint64_t max_blocks = (max_mallocs-1)/block_size+1;

   bitarr_grouped<4> * dev_bitarray;

   cudaMalloc((void **)& dev_bitarray, sizeof(bitarr_grouped<4>)*max_blocks);


   cudaDeviceSynchronize();

   assert(dev_bitarray != nullptr);

   malloc_init_kernel<<<(max_blocks -1)/512+1, 512>>>(dev_bitarray, max_blocks);

   cudaDeviceSynchronize();


   storage_bitmap<4> * local_bitmaps = storage_bitmap<4>::generate_buffers();

   printf("Done with init\n");

   auto bitarr_start = std::chrono::high_resolution_clock::now();



   malloc_test_kernel_prefetching<<<(max_mallocs -1)/block_size+1, block_size>>>(dev_bitarray, local_bitmaps, max_mallocs, mallocs_per_thread);

   cudaDeviceSynchronize();

   auto bitarr_end = std::chrono::high_resolution_clock::now();

   printf("Done with speed test\n");

   std::chrono::duration<double> bit_diff = bitarr_end - bitarr_start;

   std::cout << "bitarr Malloced " << max_mallocs*mallocs_per_thread << " in " << bit_diff.count() << " seconds, " << block_size << "max block size\n";

   printf("%f allocs per second\n", ((double) max_mallocs*mallocs_per_thread)/ bit_diff.count());


   cudaDeviceSynchronize();

   cudaFree(dev_bitarray);


   cudaDeviceSynchronize();

   uint64_t * max_counters;

   cudaMalloc((void ** )&max_counters, sizeof(uint64_t)*max_blocks*4096);

   assert(max_counters != nullptr);
   

   cudaMemset(max_counters, 0, sizeof(uint64_t)*max_blocks*4096);

   //and boot correctness test
   cudaMalloc((void **)& dev_bitarray, sizeof(bitarr_grouped<4>)*max_blocks);

   assert(dev_bitarray != nullptr);

   cudaDeviceSynchronize();

   malloc_init_kernel<<<(max_blocks -1)/512+1, 512>>>(dev_bitarray, max_blocks);






   cudaDeviceSynchronize();

   //malloc_test_correctness_kernel<<<(max_mallocs -1)/block_size+1, block_size>>>(dev_bitarray, max_counters, max_mallocs, mallocs_per_thread);


   cudaDeviceSynchronize();

   cudaFree(max_counters);

   cudaFree(dev_bitarray);





}

__host__ void build_bitarr_test_split(uint64_t max_mallocs, uint64_t mallocs_per_thread, uint64_t block_size){


   uint64_t max_blocks = (max_mallocs-1)/block_size+1;

   bitarr_grouped<4> * dev_bitarray;

   cudaMalloc((void **)& dev_bitarray, sizeof(bitarr_grouped<4>)*max_blocks);


   cudaDeviceSynchronize();

   assert(dev_bitarray != nullptr);

   malloc_init_kernel<<<(max_blocks -1)/512+1, 512>>>(dev_bitarray, max_blocks);

   cudaDeviceSynchronize();


   storage_bitmap<4> * local_bitmaps = storage_bitmap<4>::generate_buffers_blocks(max_blocks);

   printf("Done with init\n");

   auto bitarr_start = std::chrono::high_resolution_clock::now();



   malloc_test_kernel_split_local<<<(max_mallocs -1)/block_size+1, block_size>>>(dev_bitarray, local_bitmaps, max_mallocs, mallocs_per_thread);

   cudaDeviceSynchronize();

   auto bitarr_end = std::chrono::high_resolution_clock::now();

   printf("Done with speed test\n");

   std::chrono::duration<double> bit_diff = bitarr_end - bitarr_start;

   std::cout << "bitarr Malloced " << max_mallocs*mallocs_per_thread << " in " << bit_diff.count() << " seconds, " << block_size << "max block size\n";

   printf("%f allocs per second\n", ((double) max_mallocs*mallocs_per_thread)/ bit_diff.count());


   cudaDeviceSynchronize();

   cudaFree(dev_bitarray);


   cudaDeviceSynchronize();

   uint64_t * max_counters;

   cudaMalloc((void ** )&max_counters, sizeof(uint64_t)*max_blocks*4096);

   assert(max_counters != nullptr);
   

   cudaMemset(max_counters, 0, sizeof(uint64_t)*max_blocks*4096);

   //and boot correctness test
   cudaMalloc((void **)& dev_bitarray, sizeof(bitarr_grouped<4>)*max_blocks);

   assert(dev_bitarray != nullptr);

   cudaDeviceSynchronize();

   malloc_init_kernel<<<(max_blocks -1)/512+1, 512>>>(dev_bitarray, max_blocks);






   cudaDeviceSynchronize();

   //malloc_test_correctness_kernel<<<(max_mallocs -1)/block_size+1, block_size>>>(dev_bitarray, max_counters, max_mallocs, mallocs_per_thread);


   cudaDeviceSynchronize();

   cudaFree(max_counters);

   cudaFree(dev_bitarray);





}

int main(int argc, char** argv) {

   //1 mil
   build_bitarr_test_prefetch(512, 2, 512);

   //10 mil
   //build_bitarr_test_prefetch(10000000, 1, 512);

   // //100 mil
   // build_bitarr_test_prefetch(100000000, 1, 512);

   //    //1 mil
   // build_bitarr_test_prefetch(1000000, 1, 1024);

   // //10 mil
   // build_bitarr_test_prefetch(10000000, 1, 1024);

   // //100 mil
   // build_bitarr_test_prefetch(100000000, 1, 1024);


	return 0;

}
