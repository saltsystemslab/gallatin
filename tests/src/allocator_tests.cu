/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/const_heap_pointer.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>


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

//the smallest allowed is two byte chunks
using test_type = const_heap_pointer<4, 10>;


using small_type = const_heap_pointer<4,2>;


#define BYTES_USED 131072

using large_type = bytes_given_wrapper<4, BYTES_USED>::heap_ptr;

using manager_type = manager<4, 8196>;

__global__ void setup_tiny_superblock(char * superblock_ptr){



   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid != 0) return;

   //only one thread running, init superblock

   printf("Tiny block setup done\n");

   small_type * heap_ptr = small_type::init_superblock(superblock_ptr);

   __threadfence();

   while (heap_ptr != nullptr){
      printf("Iterating through heap: %llu\n", heap_ptr - ((small_type *) superblock_ptr));

      heap_ptr = heap_ptr->non_atomic_get_next();
   }


   //and print off last node

   small_type * ext_heap = (small_type * )superblock_ptr;

   printf("Heap count %d\n", ext_heap->count_heap_valid());

   printf("Tiny block setup done\n");
   return;
}



__global__ void setup_superblock(char * superblock_ptr){


   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid != 0) return;

   //only one thread running, init superblock
   test_type * heap_ptr = test_type::init_superblock(superblock_ptr);

   while (heap_ptr != nullptr){
      printf("%llu\n", heap_ptr - ((test_type *) superblock_ptr));

      heap_ptr = heap_ptr->non_atomic_get_next();
   }


   //and print off last node
   //printf("%llu\n", heap_ptr - ((test_type *) superblock_ptr));


   // heap_ptr = (test_type *) superblock_ptr;

   // heap_ptr->assert_valid();
   return;

}


__global__ void test_atomic_allocates(test_type * heap_ptr){


   //who cares about the number of tid?

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   //printf("%llu starting\n", tid);

   uint * my_allocation = (uint *) heap_ptr->malloc();


   while (my_allocation == nullptr){

      //retry!
      my_allocation = (uint *) heap_ptr->malloc();


   }

   if (my_allocation == nullptr){
      //printf("Couldn't get node\n");
   } else {

      //printf("Allocation offset %llu\n", (test_type *) my_allocation - heap_ptr);

     //atm I think this breaks it

     //this does! why?
     my_allocation[0] = 105;

     heap_ptr->free(my_allocation);

   }



}

__global__ void one_thread_allocate(test_type * heap_ptr){

   //go through and malloc a bunch of them

   //heap_ptr->printstack(10);

   uint16_t * allocations[10];

   for (int i =0; i < 9; i++){


      allocations[i] = (uint16_t *) heap_ptr->malloc();

   }

   printf("Before frees:\n");
   // heap_ptr->printstack(0);


   for (int i = 0; i < 9; i++){

      //printf("Node %d\n", i+1);

      heap_ptr->free(allocations[i]);

      //heap_ptr->printstack(i+1);



      // if (!heap_ptr->assert_valid()){
      //    printf("De-Allocation of %d failed\n", i);
      // }
   }


   printf("All done\n");

}

__global__ void small_test_allocate(small_type * heap_ptr){

   //first allocation is 0
   //heap_ptr->printstack(2);

   //printf("Heap ptr offset %hu\n", heap_ptr->atomicload());
   //printf("Next Node offset %hu\n", heap_ptr->next()->atomicload());

   assert(heap_ptr != nullptr);

   assert(heap_ptr->count_heap_valid() == 2);


   return;

   uint16_t * allocation = (uint16_t *) heap_ptr->malloc();

   assert(allocation != nullptr);
   assert(allocation != 0);

   printf("Counted distance, should be 1: %llu\n", ( (small_type *) allocation) - heap_ptr);

   

   //printf("Heap ptr offset %hu\n", heap_ptr->atomicload());


   uint16_t * secondary_allocation = (uint16_t * ) heap_ptr->malloc();

   //return;

   assert(secondary_allocation == nullptr);

   heap_ptr->free(allocation);

}

__global__ void tiny_one_thread_allocate(test_type * heap_ptr, int val){

   //go through and malloc a bunch of them

   //heap_ptr->printstack(10);

   uint16_t * allocations[10];

   for (int i =0; i < val; i++){

      printf("Starting allocation of %d\n", i);

      allocations[i] = (uint16_t *) heap_ptr->malloc();

      assert(allocations[i] != nullptr);

      assert(heap_ptr->count_heap_valid == 9-i);

      // if (!heap_ptr->assert_valid()){
      //    printf("Allocation of %d failed\n", i);
      // }
   }

   // printf("Before frees:\n");
   // heap_ptr->printstack(0);


   for (int i = 0; i < val; i++){

      printf("Node %d\n", i+1);

      if (allocations[i] != nullptr){
          heap_ptr->free(allocations[i]);
      }

     

     // heap_ptr->printstack(i+1);



      // if (!heap_ptr->assert_valid()){
      //    printf("De-Allocation of %d failed\n", i);
      // }
   }

   assert(heap_ptr->count_heap_valid == 10);


   printf("All done\n");

}



__host__ void test_allocates_host(uint64_t num_threads, test_type * heap_ptr){

   test_atomic_allocates<<<1, num_threads>>>(heap_ptr);

   cudaDeviceSynchronize();

   printf("Done with %llu\n", num_threads);

   //count_superblock<<<1,1>>>(heap_ptr);

   cudaDeviceSynchronize();

}

__global__ void test_large_allocates(uint64_t num_threads, large_type * allocator){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_threads) return;

   int * my_allocation = (int *) allocator->malloc();

   while (my_allocation == nullptr){

      my_allocation = (int *) allocator->malloc();

   }

   allocator->free(my_allocation);

   return;



}

__host__ void test_large_allocator(uint64_t num_threads, large_type * allocator){

   cudaDeviceSynchronize();
   auto allocate_start = std::chrono::high_resolution_clock::now();
   test_large_allocates<<<(num_threads -1)/1024 + 1, 1024>>>(num_threads, allocator);
   cudaDeviceSynchronize();

   auto allocate_end = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> alloc_diff = allocate_end - allocate_start;


   cudaDeviceSynchronize();
   std::cout << "allocated " << num_threads << " in " << alloc_diff.count() << " seconds\n";

   printf("Items/bytes per second: %f / %f\n", 1.0*num_threads/alloc_diff.count(), 1.0*num_threads*4/alloc_diff.count());
  

}

__global__ void assert_correct_kernel(test_type * heap_ptr){

   assert(heap_ptr->count_heap_valid == 10);
}

__host__ void assert_heap_correct(test_type * heap_ptr){

   assert_correct_kernel<<<1,1>>>(heap_ptr);

   cudaDeviceSynchronize();
   printf("Asserted_correct\n");

}

int main(int argc, char** argv) {


   char * char_heap;

   cudaMalloc((void **)&char_heap, 40);

   setup_superblock<<<1,1>>>(char_heap);

   cudaDeviceSynchronize();


   printf("Heap setup done\n");


   test_type * heap_ptr = (test_type * ) char_heap;


   // count_superblock<<<1,1>>>(heap_ptr);

   // cudaDeviceSynchronize();


   for (int i =1; i < 10; i++){

      printf("Starting run sized %d\n", i);

      tiny_one_thread_allocate<<<1,1>>>(heap_ptr, i);

      cudaDeviceSynchronize();

   }


   // count_superblock<<<1,1>>>(heap_ptr);

   printf("Done with first swap\n");

   cudaDeviceSynchronize();

   tiny_one_thread_allocate<<<1,1>>>(heap_ptr, 8);

   cudaDeviceSynchronize();


   // count_superblock<<<1,1>>>(heap_ptr);

   // one_thread_allocate<<<1,1>>>(heap_ptr);

   // cudaDeviceSynchronize();

   fflush(stdout);
   printf("Starting on test_allocates_host\n");

   assert_heap_correct(heap_ptr);


   test_allocates_host(1, heap_ptr);

   test_allocates_host(100, heap_ptr);

   test_allocates_host(1000, heap_ptr);

   printf("host allocated done\n");

   cudaDeviceSynchronize();

   cudaFree(char_heap);

   char * tiny_heap;

   cudaMalloc((void **)&tiny_heap, 8);

   small_type * tiny_heap_ptr = (small_type *) tiny_heap;

   setup_tiny_superblock<<<1,1>>>(tiny_heap);

   cudaDeviceSynchronize();


   cudaFree(tiny_heap);

   //Initialize large block
   char * mega_heap;

   cudaMalloc((void **)&mega_heap, BYTES_USED);

   cudaDeviceSynchronize();


   printf("Heap allocated\n");
   fflush(stdout);
   auto allocate_start = std::chrono::high_resolution_clock::now();


   large_type * heap = large_type::host_init_superblock(mega_heap);


   auto allocate_end = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> alloc_diff = allocate_end - allocate_start;


   cudaDeviceSynchronize();
   std::cout << "prepped space in " << alloc_diff.count() << " seconds\n";
   fflush(stdout);


   test_large_allocator(1, heap);

   test_large_allocator(100, heap);

   test_large_allocator(1000, heap);

   test_large_allocator(10000, heap);

   test_large_allocator(100000, heap);

   //test_large_allocator(1000000, heap);

   cudaDeviceSynchronize();
   cudaFree(mega_heap);


   assert(sizeof(internal_bitfield) == 4);

   // void * manager_heap;

   // cudaMalloc((void **)&manager_heap, BYTES_USED);

   //manager<4, BYTES_USED> * test_manager;






	return 0;

}
