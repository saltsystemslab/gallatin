/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/slab_one_size.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace std::chrono;


#include <cooperative_groups.h>


#include <poggers/allocators/one_size_allocator.cuh>

namespace cg = cooperative_groups;

using namespace poggers::allocators;


double elapsed(high_resolution_clock::time_point t1, high_resolution_clock::time_point t2) {
   return (duration_cast<duration<double> >(t2 - t1)).count();
}



__global__ void malloc_tests(one_size_slab_allocator<15> * allocator, uint64_t max_mallocs){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= max_mallocs) return;

   void * allocation = allocator->malloc();


   return;

}


__host__ void boot_slab_one_size(){


   one_size_slab_allocator<15> * test_alloc = one_size_slab_allocator<15>::generate_on_device(64000000, 16);

   cudaDeviceSynchronize();

   malloc_tests<<<1, 256>>>(test_alloc, 10);

   cudaDeviceSynchronize();


   one_size_slab_allocator<15>::free_on_device(test_alloc);

   cudaDeviceSynchronize();

}

template <int num_blocks>
__global__ void allocate_into_array(one_size_slab_allocator<num_blocks> * allocator, uint64_t * array, uint64_t num_mallocs, uint64_t * misses){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_mallocs) return;

   void * allocation = allocator->malloc();

   if (allocation != nullptr){

      uint64_t offset = allocator->get_offset_from_ptr(allocation);

      if (offset >= allocator->get_largest_allocation_offset()){

         printf("allocation bug %llx > %llx\n", offset, num_mallocs+15000000);
      }

      char * cast = (char *) allocation;

      cast[0] = 't';
   
   } else {
      atomicAdd((unsigned long long int *) misses, 1ULL);
   }

   array[tid] = (uint64_t) allocation;

   //printf("Tid %llu\n", tid);


}


template <int num_blocks>
__global__ void allocate_into_array_bits(one_size_slab_allocator<num_blocks> * allocator, uint64_t * array, uint64_t num_mallocs, uint64_t * misses){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_mallocs) return;

   void * allocation = allocator->malloc_mark_unpinned(&misses[3]);

   if (allocation != nullptr){

      uint64_t offset = allocator->get_offset_from_ptr(allocation);

      if (offset >= allocator->get_largest_allocation_offset()){

         printf("allocation bug %llu > %llu, diff is %llu\n", offset, num_mallocs+15000000, offset - (num_mallocs+15000000));
      }

      char * cast = (char *) allocation;

      cast[0] = 't';


      int local_bits = offset % 64;
      uint64_t shrunken_offset = offset/64;

      uint64_t old_bits = atomicOr((unsigned long long int *)&array[shrunken_offset], (1ULL << local_bits));

      if (old_bits & (1ULL << local_bits)){
         //printf("Bit alloc array Bug with index %llu\n", offset);
         atomicAdd((unsigned long long int *)misses, 1ULL);
      }

   
   } else {
      atomicAdd((unsigned long long int *) misses, 1ULL);
   }

   //printf("Tid %llu\n", tid);


}


template <int num_blocks>
__global__ void free_from_array(one_size_slab_allocator<num_blocks> * allocator, uint64_t * array, uint64_t num_mallocs){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_mallocs) return;

   void * allocation = (void *) array[tid];

   if (allocation != nullptr){
      //allocator->free(allocation, num_mallocs+15000000);
      allocator->free(allocation);
   }



}

template <int num_blocks>
__global__ void free_from_array_bits(one_size_slab_allocator<num_blocks> * allocator, uint64_t * array, uint64_t max_offset_bit, uint64_t * misses){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= max_offset_bit) return;


   uint64_t lower_offset = tid/64;
   int offset_bit = tid % 64;

   bool valid = (array[lower_offset] & (1ULL << offset_bit));

   if (!valid) return;

   void * allocation = (void *) (allocator->offset_size*tid + allocator->extra_memory);

   if (allocator->get_offset_from_ptr(allocation) != tid){
      printf("Bug in free offset generation\n");
   }

   allocator->free_count_misses(allocation, misses);



}

template <int num_blocks>
__global__ void log_free_kernel(one_size_slab_allocator<num_blocks> * allocator, uint64_t * array, uint64_t max_offset_bit){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= max_offset_bit) return;

   uint64_t lower_offset = tid/64;
   int offset_bit = tid % 64;

   bool valid = (array[lower_offset] & (1ULL << offset_bit));

   if (!valid) return;

   void * allocation = (void *) (allocator->offset_size*tid + allocator->extra_memory);


   allocator->log_free();


}


template <int blocks>
__host__ void test_num_malloc_frees_bitarr(uint64_t num_mallocs, int num_rounds){

      //I think 4,000,000 is enough to saturate the wavefront.
   //108 SMs, each with 4096 items
   //pocket math says 500,000 is sufficient for an A100
   //times 16 jk gonna do 10,000,000 to be safe

   printf("Starting test with %llu threads and %d rounds\n", num_mallocs, num_rounds);


   uint64_t total_num_allocs = 15000000+num_mallocs+4096;

   uint64_t total_num_allocs_bits = (total_num_allocs-1)/64+1;


   one_size_slab_allocator<blocks> * test_alloc = one_size_slab_allocator<blocks>::generate_on_device(total_num_allocs, 1);


   uint64_t pre_fill = test_alloc->report_fill();

   uint64_t pre_max = test_alloc->report_max();

   printf("Initial fill ratio %llu/%llu %f \n", pre_fill, pre_max, 1.0*pre_fill/pre_max);


   uint64_t * array;

   cudaMalloc((void ** )&array, sizeof(uint64_t)*total_num_allocs_bits);

   cudaMemset(array, 0, sizeof(uint64_t)*total_num_allocs_bits);

   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*4);

   cudaDeviceSynchronize();



   for (int i=0; i< num_rounds; i++){

      misses[0] = 0;
      misses[1] = 0;
      misses[2] = 0;
      misses[3] = 0;

      cudaDeviceSynchronize();



      allocate_into_array_bits<blocks><<<(num_mallocs -1)/512+1, 512>>>(test_alloc, array, num_mallocs, misses);

      cudaDeviceSynchronize();

      uint64_t half_fill = test_alloc->report_fill();

      uint64_t half_max = test_alloc->report_max();

      printf("Halfway through iteration %d: %llu/%llu %f \n", i, half_fill, half_max, 1.0*half_fill/half_max);



      //log_free_kernel<blocks><<<(num_mallocs-1)/512+1,512


      free_from_array_bits<blocks><<<(total_num_allocs -1)/512+1, 512>>>(test_alloc, array, total_num_allocs, misses+1);

      cudaDeviceSynchronize();

      cudaMemset(array, 0, sizeof(uint64_t)*total_num_allocs_bits);

      cudaDeviceSynchronize();

      uint64_t fill = test_alloc->report_fill();
      uint64_t max = test_alloc->report_max();

      printf("Done with cycle %d, %llu/%llu: %f misses. %llu/%llu free\n", i, misses[0], num_mallocs, 1.0*misses[0]/num_mallocs, fill, max);
      printf("Misses in free: blocks unpinned %llu, blocks freed %llu, threads that failed %llu\n", misses[3], misses[1], misses[2]);


   }

   cudaFree(array);

   one_size_slab_allocator<blocks>::free_on_device(test_alloc);

   return;



}


template <int num_blocks>
__host__ void test_num_malloc_frees(uint64_t num_mallocs, int num_rounds){

   //I think 4,000,000 is enough to saturate the wavefront.
   //108 SMs, each with 4096 items
   //pocket math says 500,000 is sufficient for an A100
   //times 16 jk gonna do 10,000,000 to be safe

   high_resolution_clock::time_point malloc_start, malloc_end, free_start, free_end;

   printf("Starting test with %llu threads and %d rounds\n", num_mallocs, num_rounds);

   one_size_slab_allocator<num_blocks> * test_alloc = one_size_slab_allocator<num_blocks>::generate_on_device(15000000+num_mallocs, 1);


   uint64_t pre_fill = test_alloc->report_fill();

   uint64_t pre_max = test_alloc->report_max();

   printf("Initial fill ratio %llu/%llu %f \n", pre_fill, pre_max, 1.0*pre_fill/pre_max);


   uint64_t * array;

   cudaMalloc((void ** )&array, sizeof(uint64_t)*num_mallocs);

   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   cudaDeviceSynchronize();



   for (int i=0; i< num_rounds; i++){

      misses[0] = 0;

      cudaDeviceSynchronize();

      malloc_start = high_resolution_clock::now();

      allocate_into_array<num_blocks><<<(num_mallocs -1)/512+1, 512>>>(test_alloc, array, num_mallocs, misses);

      cudaDeviceSynchronize();

      malloc_end = high_resolution_clock::now();

      uint64_t half_fill = test_alloc->report_fill();

      uint64_t half_max = test_alloc->report_max();

      printf("Halfway through iteration %d: %llu/%llu %f \n", i, half_fill, half_max, 1.0*half_fill/half_max);

      cudaDeviceSynchronize();

      free_start = high_resolution_clock::now();

      free_from_array<num_blocks><<<(num_mallocs -1)/512+1, 512>>>(test_alloc, array, num_mallocs);

      cudaDeviceSynchronize();

      free_end = high_resolution_clock::now();

      uint64_t fill = test_alloc->report_fill();
      uint64_t max = test_alloc->report_max();

      printf("Done with cycle %d. %llu/%llu: %f misses. %llu/%llu free\n", i, misses[0], num_mallocs, 1.0*misses[0]/num_mallocs, fill, max);
      std::cout << "Cycle took " << elapsed(malloc_start, malloc_end) << " for malloc and " << elapsed(free_start, free_end) << " for frees.\n";


   }

   cudaFree(array);

   one_size_slab_allocator<num_blocks>::free_on_device(test_alloc);

   return;



}


__host__ void test_num_malloc_no_free(uint64_t num_mallocs, int num_rounds){

    printf("Starting test with %llu threads and %d rounds\n", num_mallocs, num_rounds);


   uint64_t * array;

   cudaMalloc((void ** )&array, sizeof(uint64_t)*num_mallocs);

   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   cudaDeviceSynchronize();


   for (int i=0; i< num_rounds; i++){


      misses[0] = 0;

      cudaDeviceSynchronize();

      one_size_slab_allocator<15> * test_alloc = one_size_slab_allocator<15>::generate_on_device(15000000+num_mallocs, 1);

      cudaDeviceSynchronize();

      allocate_into_array<15><<<(num_mallocs -1)/512+1, 512>>>(test_alloc, array, num_mallocs, misses);

      cudaDeviceSynchronize();

      //free_from_array<<<(num_mallocs -1)/512+1, 512>>>(test_alloc, array, num_mallocs);

      one_size_slab_allocator<15>::free_on_device(test_alloc);

      cudaDeviceSynchronize();

      printf("Done with cycle %d, misses %llu\n", i, misses[0]);


   }

   cudaFree(array);



   return;


}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   // for (int i =0; i< 20; i++){
   //    boot_slab_one_size();
   // }
   
   //test_num_malloc_frees(1000, 10000);


   //test_num_malloc_frees(10000, 100);

   //test_num_malloc_frees(10000, 10);

   test_num_malloc_frees<4>(1000000000, 10);

   //test_num_malloc_frees(1000000, 10);

   //test_num_malloc_frees(100000000, 10);

   //test_num_malloc_frees_bitarr<4>(100000000, 10);



 
   cudaDeviceReset();
   return 0;

}
