/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/offset_slab.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>


#include <cooperative_groups.h>


#include <poggers/allocators/one_size_allocator.cuh>

namespace cg = cooperative_groups;


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace poggers::allocators;



__global__ void scan_kernel(uint64_t allocs, uint64_t * allocations){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid/32 >= allocs) return;

   cg::coalesced_group my_group = cg::coalesced_threads();

}


__global__ void assert_unique(uint64_t * items, uint64_t num_items){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_items) return;

   uint64_t item = items[tid];

   //0's are misses
   if (item == 0) return; 

   for (uint64_t i = 0; i < tid; i++){

      if (i == tid) continue;

      if (item == items[i]){
         printf("Conflict betwen %lu and %lu: %lu\n", tid, i, item);
      }

   }

}



__global__ void init_bitarr_kernel(offset_alloc_bitarr * bitarr, offset_storage_bitmap * storage){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   bitarr->init();

   bitarr->attach_allocation(0ULL);

   storage->init();

   return;

}


// __global__ void test_warp_kernel(offset_alloc_bitarr * bitarr, offset_storage_bitmap * storage){


//    cg::coalesced_group my_warp = cg::coalesced_threads();

//    assert(my_warp.size() == 32);

//    uint64_t my_offset;

//    uint64_t my_remainder;

//    if (bitarr->bit_malloc_v2(my_warp, my_offset, my_remainder)){


//       //printf("My address %lu\n", my_offset);

//       // if (my_warp.thread_rank() == 0){

//       //    printf("Warp %d has %d remaining\n", threadIdx.x/32, __popcll(my_remainder));

//       // }

//       assert(my_offset != ~0ULL);

//       bitarr->free_allocation_v2(my_offset);

//       if (my_warp.thread_rank() == 0){


//          uint64_t my_upper = my_offset/64;

//          storage->attach_buffer(my_upper, my_remainder);

//          // while (my_remainder != 0){

//          //    int leader = __ffsll(my_remainder)-1;

//          //    bitarr->free_allocation(my_upper*64+leader);

//          //    my_remainder ^= (1ULL << leader);

//          // }


//       }

//       my_warp.sync();

//       uint64_t ext_offset;


//       if (storage->bit_malloc_v2(my_warp, ext_offset)){
//          bitarr->free_allocation_v2(ext_offset);
//       } else {
//          printf("Failure on second malloc\n");
//       }


//    } else {

//       printf("Failure!\n");

//    }


//    __syncthreads();

//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    if (tid == 0){

//       if (bitarr->manager_bits != ~0ULL){
//           printf("Manager bits: %lx\n", bitarr->manager_bits.bits);
//       }
     
//    }

// }


__global__ void gather_unique_cyclic(offset_alloc_bitarr * bitarr, offset_storage_bitmap * storage, uint64_t * allocs, uint64_t * counter){


   uint64_t allocation = ~0ULL;

   while (allocation == ~0ULL){

      if (alloc_with_locks(allocation, bitarr, storage)){

         uint64_t value = atomicAdd((unsigned long long int *)counter, 1ULL);

         allocs[value] = allocation;
         return;

      } else {
         allocation = ~0ULL;
      }


   }


}


// __global__ void test_kernel_cyclic(offset_alloc_bitarr * bitarr, offset_storage_bitmap * storage, int num_rounds){


//    cg::coalesced_group my_warp = cg::coalesced_threads();

//    assert(my_warp.size() == 32);


//    for (int i = 0; i < num_rounds; i++){

//       uint64_t my_offset;

//       uint64_t my_remainder;

//       if (bitarr->bit_malloc_v2(my_warp, my_offset, my_remainder)){


//          //printf("My address %lu\n", my_offset);

//          // if (my_warp.thread_rank() == 0){

//          //    printf("Warp %d has %d remaining\n", threadIdx.x/32, __popcll(my_remainder));

//          // }

//          assert(my_offset != ~0ULL);

//          bitarr->free_allocation_v2(my_offset);

//          if (my_warp.thread_rank() == 0){


//             uint64_t my_upper = my_offset/64;

//             storage->attach_buffer(my_upper, my_remainder);

//             // while (my_remainder != 0){

//             //    int leader = __ffsll(my_remainder)-1;

//             //    bitarr->free_allocation(my_upper*64+leader);

//             //    my_remainder ^= (1ULL << leader);

//             // }


//          }

//          my_warp.sync();

//          uint64_t ext_offset;


//          if (storage->bit_malloc_v3(my_warp, ext_offset)){
//             bitarr->free_allocation_v2(ext_offset);
//          } else {
//             printf("Failure on second malloc\n");
//          }


//       } else {

//          printf("Failure!\n");

//       }


//       __syncthreads();

//       uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//       if (tid == 0){

//          if (bitarr->manager_bits != ~0ULL){
//              printf("Manager bits: %lx\n", bitarr->manager_bits.bits);
//          }
        


//         printf("End of round %d\n", i);
//       }

//    }

// }



__global__ void test_combined_cyclic_kernel(offset_alloc_bitarr * bitarr, offset_storage_bitmap * storage, int num_rounds, uint64_t * claim_fails, uint64_t * bad_allocs){


   for (int i = 0; i < num_rounds; i++){

      uint64_t allocation;
   

      if (!alloc_with_locks(allocation, bitarr, storage)){

         atomicAdd((unsigned long long int *)claim_fails, 1ULL);
         //printf("Failure to claim!\n");
      } else {

         if (allocation == ~0ULL){

            atomicAdd((unsigned long long int *)bad_allocs, 1ULL);
            printf("Bug - allocation is unset\n");

         } else {
            bitarr->free_allocation_v2(allocation);
         }

         //__syncthreads();
      }

      __syncthreads();

      // if (threadIdx.x == 0){
      //    printf("Done with iter %d\n", i);
      // }

      __syncthreads();

   }


   __syncthreads();

}



// __host__ void test_single_warp_bitarr(uint64_t num_warps){

//    printf("Starting test for %lu warps\n", num_warps);

//    offset_alloc_bitarr * bitarr;

//    cudaMalloc((void **)&bitarr, sizeof(offset_alloc_bitarr));

//    offset_storage_bitmap * storage;

//    cudaMalloc((void **)&storage, sizeof(offset_storage_bitmap));


//    init_bitarr_kernel<<<1,1>>>(bitarr, storage);

//    cudaDeviceSynchronize();


//    test_warp_kernel<<<1, 32*num_warps>>>(bitarr, storage);


//    cudaDeviceSynchronize();

//    cudaFree(bitarr);

//    printf("\n\n");



// }


// __host__ void test_single_warp_cyclic(uint64_t num_warps, int num_rounds){

//    printf("Starting test for %lu warps\n", num_warps);

//    offset_alloc_bitarr * bitarr;

//    cudaMalloc((void **)&bitarr, sizeof(offset_alloc_bitarr));

//    offset_storage_bitmap * storage;

//    cudaMalloc((void **)&storage, sizeof(offset_storage_bitmap));


//    init_bitarr_kernel<<<1,1>>>(bitarr, storage);

//    cudaDeviceSynchronize();

//    test_kernel_cyclic<<<1, 32*num_warps>>>(bitarr, storage, num_rounds);


//    cudaDeviceSynchronize();

//    cudaFree(bitarr);

//    printf("\n\n");



// }


__host__ void test_combined_cyclic(uint64_t num_warps, int num_rounds){


   uint64_t * misses;

   cudaMallocManaged((void **)& misses, sizeof(uint64_t)*2);

   cudaDeviceSynchronize();

   misses[0] = 0;
   misses[1] = 0;

   cudaDeviceSynchronize();

   printf("Starting test for %lu warps\n", num_warps);

   offset_alloc_bitarr * bitarr;

   cudaMalloc((void **)&bitarr, sizeof(offset_alloc_bitarr));

   offset_storage_bitmap * storage;

   cudaMalloc((void **)&storage, sizeof(offset_storage_bitmap));


   init_bitarr_kernel<<<1,1>>>(bitarr, storage);

   cudaDeviceSynchronize();

   test_combined_cyclic_kernel<<<1, 32*num_warps>>>(bitarr, storage, num_rounds, misses, misses + 1);


   cudaDeviceSynchronize();

   printf("Kernel done: Claim fails %llu and claim errors %llu for %llu allocations split into %d rounds\n", misses[0], misses[1], 32ULL*num_warps*num_rounds, num_rounds);

   cudaFree(misses);

   cudaFree(bitarr);

   printf("\n\n");



}


__host__ void test_unique(uint64_t num_warps){


   uint64_t * counter;

   cudaMalloc((void **)&counter, sizeof(uint64_t));

   cudaMemset(counter, 0, sizeof(uint64_t));

   uint64_t * allocs;

   cudaMallocManaged((void **)& allocs, sizeof(uint64_t)*32*num_warps);

   cudaDeviceSynchronize();

   printf("Starting test for %lu warps\n", num_warps);

   offset_alloc_bitarr * bitarr;

   cudaMalloc((void **)&bitarr, sizeof(offset_alloc_bitarr));

   offset_storage_bitmap * storage;

   cudaMalloc((void **)&storage, sizeof(offset_storage_bitmap));


   init_bitarr_kernel<<<1,1>>>(bitarr, storage);

   cudaDeviceSynchronize();

   //test_combined_cyclic_kernel<<<1, 32*num_warps>>>(bitarr, storage, num_rounds, misses, misses + 1);


   gather_unique_cyclic<<<1,32*num_warps>>>(bitarr, storage, allocs, counter);




   cudaDeviceSynchronize();

   assert_unique<<<1,32*num_warps>>>(allocs, 32*num_warps);

   cudaDeviceSynchronize();

   //printf("Kernel done: Claim fails %llu and claim errors %llu for %llu allocations split into %d rounds\n", misses[0], misses[1], 32ULL*num_warps*num_rounds, num_rounds);

   cudaFree(allocs);

   cudaFree(bitarr);

   printf("\n\n");



}



// __global__ void start_kernel(){

//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    __shared__ kernel_init_test;

// }

// __host__ void start_kernel_host(){

//    //doesn't work - maybe build layer to detect block start and stop?
//    printf("Host starting test\n");

//    start_kernel<<<1,1>>>();

//    cudaDeviceSynchronize();

//    return;

// }

template <int num_blocks>
__global__ void pinned_alloc_kernels(smid_pinned_container<num_blocks> * bitarr_containers, pinned_storage * storage_containers, uint64_t num_threads, int allocs_per_thread){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_threads) return;

   int allocs_claimed = 0;

   smid_pinned_storage<num_blocks> * my_storage = bitarr_containers->get_pinned_storage();

   offset_storage_bitmap * my_storage_bitmap = storage_containers->get_pinned_storage();

   while (allocs_claimed < allocs_per_thread){

      //printf("Starting round %d\n", allocs_claimed);

      offset_alloc_bitarr * bitarr = my_storage->get_primary();

      if (bitarr == nullptr) continue;

      uint64_t allocation;

      bool alloced = alloc_with_locks(allocation, bitarr, my_storage_bitmap);

      if (!alloced){


         int result = my_storage->pivot_primary(bitarr);

         //printf("Spinning on failure %d\n", result);

      } else {
          allocs_claimed+=1;
      }



     

   }

}


template <int num_blocks>
__global__ void team_pinned_alloc_kernels(smid_pinned_container<num_blocks> * bitarr_containers, pinned_storage * storage_containers, uint64_t num_threads, int allocs_per_thread){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_threads) return;

   int allocs_claimed = 0;

   smid_pinned_storage<num_blocks> * my_storage = bitarr_containers->get_pinned_storage();

   offset_storage_bitmap * my_storage_bitmap = storage_containers->get_pinned_storage();



   while (allocs_claimed < allocs_per_thread){

      auto team = cg::coalesced_threads();

      //printf("Starting round %d\n", allocs_claimed);

      offset_alloc_bitarr * bitarr = my_storage->get_primary();

      if (bitarr == nullptr){
         team.sync();

         continue;
      }

      uint64_t allocation;

      bool alloced = alloc_with_locks(allocation, bitarr, my_storage_bitmap);

      if (!alloced){


         int result = my_storage->pivot_primary(bitarr);

         //printf("Spinning on failure %d\n", result);

      } else {
          allocs_claimed+=1;
      }

      team.sync();

     

   }

}

template<int num_blocks>
__global__ void smid_warp_allocs(smid_pinned_storage<num_blocks> * storage, offset_storage_bitmap * storage_bitmap, int num_warps, int allocs_per_block){


   uint64_t tid =threadIdx.x+blockIdx.x*blockDim.x;

   if (tid == 0){
      printf("Each thread must alloc %d items\n", (num_blocks+1)*allocs_per_block/(32*num_warps));
   }

   int warp_id = tid/32;

   if (warp_id >= num_warps) return;

   uint64_t allocation;

   int allocs_acquired = 0;

   while (allocs_acquired < (num_blocks+1)*allocs_per_block/(32*num_warps)){



      cg::coalesced_group mix = cg::coalesced_threads();


      offset_alloc_bitarr * bitarr = storage->get_primary();

      if (bitarr == nullptr) continue;

      bool alloced = alloc_with_locks(allocation, bitarr, storage_bitmap);


      if (!alloced){

         //attempt to swap out



         int result = storage->pivot_primary(bitarr);

         //printf("Alloc failed %d result\n", result);


         // if (result != -1){
         //    //get new buffer
         // }

      } else {

         allocs_acquired+=1;


      }


      mix.sync();


   }



}

template <int num_blocks>
__global__ void load_smid_components(smid_pinned_storage<num_blocks> * storage, offset_alloc_bitarr * bitarrs, offset_storage_bitmap * storage_bitmap){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   storage[0].init();

   for (int i = 0; i < num_blocks+1; i++){

      bitarrs[i].init();

      bitarrs[i].attach_allocation(4096*i);

      storage[0].attach_new_buffer(i, &bitarrs[i]);
   }

   storage_bitmap->init();


}

template<int num_blocks>
__host__ void test_smid_component(int num_warps){

   smid_pinned_storage<num_blocks> * test_storage;

   cudaMalloc((void **)&test_storage, sizeof(smid_pinned_storage<num_blocks>));

   offset_alloc_bitarr * bitarrs;

   cudaMalloc((void **)&bitarrs, sizeof(offset_alloc_bitarr)*(num_blocks+1));


   offset_storage_bitmap * storage;

   cudaMalloc((void **)&storage, sizeof(offset_storage_bitmap));

   cudaDeviceSynchronize();


   load_smid_components<num_blocks><<<1,1>>>(test_storage, bitarrs, storage);

   cudaDeviceSynchronize();

   smid_warp_allocs<num_blocks><<<1,32*num_warps>>>(test_storage, storage, num_warps, 4096);

   cudaDeviceSynchronize(); 


   cudaFree(test_storage);

   cudaFree(bitarrs);


}



template <int num_blocks>
__host__ void test_pinned_components(uint64_t num_allocs, uint64_t ext_size, uint64_t num_threads, int allocs_per_thread){



   printf("Starting up test of %llu threads / %d allocs with allocators - each will serve %llu allocations\n", num_threads, allocs_per_thread, (num_allocs-1)/4096+1);

   one_size_allocator * block_allocator = one_size_allocator::generate_on_device((num_allocs-1)/4096+1, sizeof(offset_alloc_bitarr), 17);

   one_size_allocator * mem_allocator = one_size_allocator::generate_on_device((num_allocs-1)/4096+1, 4096*ext_size, 1324);


   smid_pinned_container<num_blocks> * malloc_containers = smid_pinned_container<num_blocks>::generate_on_device(block_allocator, mem_allocator);


   pinned_storage * storage_containers = pinned_storage::generate_on_device();

   cudaDeviceSynchronize();


   // (smid_pinned_container<num_blocks> * bitarr_containers, pinned_storage * storage_containers, uint64_t num_threads, int allocs_per_thread){


   auto malloc_start = std::chrono::high_resolution_clock::now();


   team_pinned_alloc_kernels<num_blocks><<<(num_threads-1)/512+1,512>>>(malloc_containers, storage_containers, num_threads, allocs_per_thread);

   cudaDeviceSynchronize();

   auto malloc_end = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> elapsed_seconds = malloc_end - malloc_start;

   uint64_t total_allocs = num_threads*allocs_per_thread;

   std::cout << "Malloced " <<  total_allocs << " in " << elapsed_seconds.count() << " seconds, throughput: " << std::fixed << 1.0*total_allocs/elapsed_seconds.count() << std::endl;
  

   smid_pinned_container<num_blocks>::free_on_device(malloc_containers);

   pinned_storage::free_on_device(storage_containers);

   one_size_allocator::free_on_device(block_allocator);

   one_size_allocator::free_on_device(mem_allocator);

   cudaDeviceSynchronize();



}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   // test_smid_component<0>(1);

   // test_smid_component<1>(1);


   // test_smid_component<8>(1);


   // printf("Starting test with 2 warps\n");

   // test_smid_component<0>(2);

   // test_smid_component<1>(2);

   // test_smid_component<63>(2);


   // test_smid_component<63>(8);


   // for (int i = 0; i< 20; i++){
   //    printf("Round %d\n", i);
   //    test_smid_component<63>(16);
   // }
   


   //test_pinned_components<0>(64000000, 1, 32, 1);


   test_pinned_components<15>(64000000, 1, 4*512*108, 10);

   //this causes an actual failure
   test_pinned_components<63>(64000000, 1, 20*512*108, 10);
   //test_single_warp_bitarr(1);

   // test_single_warp_bitarr(2);

   // test_single_warp_bitarr(4);

   // test_single_warp_bitarr(8);

   // test_single_warp_bitarr(16);

   //test_combined_cyclic(1, 2);

   //test_combined_cyclic(2, 10);

   //test_combined_cyclic(4, 20);

   //test_combined_cyclic(8, 100);

   //test_combined_cyclic(16, 1);

  // test_combined_cyclic(16, 1000);



   // pinned_storage * test_storage = pinned_storage::generate_on_device();

   // cudaDeviceSynchronize();

   // pinned_storage::free_on_device(test_storage);

   // cudaDeviceSynchronize();


   // test_unique(1);

   // test_unique(4);

   // test_unique(16);

 
   cudaDeviceReset();
   return 0;

}
