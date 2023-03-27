/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/bitbuddy.cuh>

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


//helper to assert uniqueness
__global__ void assert_unique(char ** unique_ids, uint64_t num_allocs){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;

   char * my_allocation = unique_ids[tid];


   if (my_allocation == nullptr){


      printf("FAIL to malloc\n");
      asm("trap;");
   }

   for (uint64_t i =0; i < num_allocs; i++){


      if (i != tid && my_allocation == unique_ids[i]){
         asm("trap;");
      }
   }

}


//Unit Test 1
// cuda kernel
// with an exact thread match, can we successfully allocate all blocks in a layer?
__global__ void buddy_alloc_malloc_test_1(bitbuddy_allocator * alloc, uint64_t num_threads, char ** unique_ids){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;


   for (int i =0; i< num_threads; i++){



   void * my_alloc = alloc->malloc(1);

   //void * my_alloc = nullptr;

  


      if (my_alloc == nullptr){
         printf("Problem! in Malloc!\n");
         asm("trap;");

      }

      else {

         //alloc->assert_correct_setup(my_alloc);

         unique_ids[i] = (char *) my_alloc;

      }

      printf("Done with %i/%llu\n", i, num_threads);

   }

}


__global__ void buddy_alloc_free_test_1(bitbuddy_allocator * alloc, uint64_t num_threads, char ** unique_ids){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;


   for (int i = 0; i < num_threads; i++){

      if (unique_ids[i] == nullptr) continue;

      alloc->free((void * ) unique_ids[i]); 

   }

}



__global__ void buddy_alloc_malloc_test_2(bitbuddy_allocator * alloc, uint64_t num_threads, char ** unique_ids){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_threads) return;



   void * my_alloc = alloc->malloc(1);

   //void * my_alloc = nullptr;



   if (my_alloc == nullptr){
      printf("Problem in Malloc! %llu\n", tid);
      //asm("trap;");

   }

   else {

      //alloc->assert_correct_setup(my_alloc);

      unique_ids[tid] = (char *) my_alloc;

   }

}



__global__ void single_malloc_test_count_misses(bitbuddy_allocator * alloc, uint64_t num_threads, uint64_t * misses){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   for (uint64_t i = 0; i < num_threads; i++){


       void * my_alloc = alloc->malloc(1);

       //void * my_alloc = nullptr;



      if (my_alloc == nullptr){
      
         atomicAdd((unsigned long long int *) misses, 1ULL);
         //asm("trap;");

      }


   }


}

__global__ void malloc_test_cyclic(bitbuddy_allocator * alloc, uint64_t num_threads, uint64_t num_rounds, uint64_t * misses){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_threads) return;



   for (uint64_t i =0; i < num_rounds; i++){

      void * my_alloc = alloc->malloc(1);

      //void * my_alloc = nullptr;



     if (my_alloc == nullptr) {


         void * second_alloc = alloc->malloc(1);

     //     alloc->free(second_alloc);

     }



      if (my_alloc == nullptr){



      
         atomicAdd((unsigned long long int *) misses, 1ULL);
         //asm("trap;");

      } else {

         alloc->free(my_alloc);
      }


      //printf("Progressing, %llu\n", tid);





   }

}

__global__ void malloc_test_count_misses(bitbuddy_allocator * alloc, uint64_t num_threads, uint64_t * misses){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_threads) return;



   void * my_alloc = alloc->malloc(1);

   //void * my_alloc = nullptr;



   if (my_alloc == nullptr){
   
      atomicAdd((unsigned long long int *) misses, 1ULL);
      //asm("trap;");

   }


}


__global__ void buddy_alloc_free_test_2(bitbuddy_allocator * alloc, uint64_t num_threads, char ** unique_ids){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_threads) return;


   if (unique_ids[tid] == nullptr) return;


   alloc->free((void * ) unique_ids[tid]); 

}



__host__ bool test_one_thread(){


   char ** grabbed_allocs;

   cudaMalloc((void **)&grabbed_allocs, sizeof(char *)*1);

   char * ext_memory;

   cudaMalloc((void **)&ext_memory, 31);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, 31, 1);

   cudaDeviceSynchronize();


   buddy_alloc_malloc_test_1<<<1,1>>>(alloc, 1, grabbed_allocs);

   cudaDeviceSynchronize();

   assert_unique<<<1,31>>>(grabbed_allocs, 1);

   cudaDeviceSynchronize();

   buddy_alloc_free_test_1<<<1,1>>>(alloc, 1, grabbed_allocs);

   cudaDeviceSynchronize();


   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   cudaFree(ext_memory);

   cudaFree(grabbed_allocs);


   return true;



}

__host__ bool test_one_thread_two(){


   char ** grabbed_allocs;

   cudaMalloc((void **)&grabbed_allocs, sizeof(char *)*2);

   char * ext_memory;

   cudaMalloc((void **)&ext_memory, 31);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, 31, 2);

   cudaDeviceSynchronize();


   buddy_alloc_malloc_test_1<<<1,1>>>(alloc, 2, grabbed_allocs);

   cudaDeviceSynchronize();

   assert_unique<<<1,31>>>(grabbed_allocs, 2);

   cudaDeviceSynchronize();

   buddy_alloc_free_test_1<<<1,1>>>(alloc, 2, grabbed_allocs);

   cudaDeviceSynchronize();


   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   cudaFree(ext_memory);

   cudaFree(grabbed_allocs);


   return true;



}

//simplicity with 31 bits
// should only occupy one line
// If it doesn't need to tune
__host__ bool test_1(){


   char ** grabbed_allocs;

   cudaMalloc((void **)&grabbed_allocs, sizeof(char *)*31);

   char * ext_memory;

   cudaMalloc((void **)&ext_memory, 31);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, 31, 1);

   cudaDeviceSynchronize();


   buddy_alloc_malloc_test_1<<<1,1>>>(alloc, 31, grabbed_allocs);

   cudaDeviceSynchronize();

   assert_unique<<<1,31>>>(grabbed_allocs, 31);

   cudaDeviceSynchronize();

   buddy_alloc_free_test_1<<<1,1>>>(alloc, 31, grabbed_allocs);

   cudaDeviceSynchronize();


   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   cudaFree(ext_memory);

   cudaFree(grabbed_allocs);


   return true;



}


__host__ bool test_2(){


   char ** grabbed_allocs;

   cudaMalloc((void **)&grabbed_allocs, sizeof(char *)*31);

   char * ext_memory;

   cudaMalloc((void **)&ext_memory, 31);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, 31, 1);

   cudaDeviceSynchronize();


   buddy_alloc_malloc_test_2<<<1,31>>>(alloc, 31, grabbed_allocs);

   cudaDeviceSynchronize();

   assert_unique<<<1,31>>>(grabbed_allocs, 31);

   cudaDeviceSynchronize();

   //buddy_alloc_free_test_1<<<1,1>>>(alloc, 31, grabbed_allocs);

   buddy_alloc_free_test_2<<<1,31>>>(alloc, 31, grabbed_allocs);

   cudaDeviceSynchronize();


   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   cudaFree(ext_memory);

   cudaFree(grabbed_allocs);


   return true;



}


__host__ bool test_num_allocs_one_thread(uint64_t num_allocs){



   char ** grabbed_allocs;

   cudaMalloc((void **)&grabbed_allocs, sizeof(char *)*num_allocs);

   char * ext_memory;

   cudaMalloc((void **)&ext_memory, num_allocs);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, num_allocs, 1);

   cudaDeviceSynchronize();


   buddy_alloc_malloc_test_1<<<1,1>>>(alloc, num_allocs, grabbed_allocs);

   cudaDeviceSynchronize();

   assert_unique<<<(num_allocs-1)/512+1,512>>>(grabbed_allocs, num_allocs);

   cudaDeviceSynchronize();

   //buddy_alloc_free_test_1<<<1,1>>>(alloc, 31, grabbed_allocs);

   buddy_alloc_free_test_1<<<1,1>>>(alloc, num_allocs, grabbed_allocs);

   cudaDeviceSynchronize();


   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   cudaFree(ext_memory);

   cudaFree(grabbed_allocs);


   return true;



}


__host__ bool test_multi_threaded_boundary_singleton(){



   char ** grabbed_allocs;

   char ** single_grabbed_allocs;

   cudaMalloc((void **)&grabbed_allocs, sizeof(char *)*2);

   cudaMalloc((void **)&single_grabbed_allocs, sizeof(char *)*31);

   char * ext_memory;

   cudaMalloc((void **)&ext_memory, 33);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, 33, 1);

   cudaDeviceSynchronize();

   buddy_alloc_malloc_test_1<<<1,1>>>(alloc, 31, single_grabbed_allocs);

   buddy_alloc_malloc_test_2<<<1,1>>>(alloc, 1, grabbed_allocs);
   buddy_alloc_malloc_test_2<<<1,1>>>(alloc, 1, grabbed_allocs);

   cudaDeviceSynchronize();

   assert_unique<<<1,2>>>(grabbed_allocs,1);

   cudaDeviceSynchronize();

   //buddy_alloc_free_test_1<<<1,1>>>(alloc, 31, grabbed_allocs);

   buddy_alloc_free_test_2<<<1,1>>>(alloc, 1, grabbed_allocs);

   cudaDeviceSynchronize();


   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   cudaFree(ext_memory);

   cudaFree(grabbed_allocs);

   cudaFree(single_grabbed_allocs);


   return true;



}


__host__ bool test_multi_threaded_boundary(){



   char ** grabbed_allocs;

   char ** single_grabbed_allocs;

   cudaMalloc((void **)&grabbed_allocs, sizeof(char *)*2);

   cudaMalloc((void **)&single_grabbed_allocs, sizeof(char *)*31);

   char * ext_memory;

   cudaMalloc((void **)&ext_memory, 33);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, 33, 1);

   cudaDeviceSynchronize();

   buddy_alloc_malloc_test_1<<<1,1>>>(alloc, 31, single_grabbed_allocs);

   cudaDeviceSynchronize();

   fflush(stdout);
   printf("Done with prev work\n");
   fflush(stdout);

   cudaDeviceSynchronize();

   buddy_alloc_malloc_test_2<<<1,2>>>(alloc, 2, grabbed_allocs);

   cudaDeviceSynchronize();

   assert_unique<<<1,2>>>(grabbed_allocs,2);

   cudaDeviceSynchronize();

   //buddy_alloc_free_test_1<<<1,1>>>(alloc, 31, grabbed_allocs);

   buddy_alloc_free_test_2<<<1,2>>>(alloc, 2, grabbed_allocs);

   cudaDeviceSynchronize();


   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   cudaFree(ext_memory);

   cudaFree(grabbed_allocs);

   cudaFree(single_grabbed_allocs);


   return true;



}


__host__ bool test_cycles(uint64_t num_allocs, uint64_t num_rounds){



   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   misses[0] = 0;


   char * ext_memory;

   cudaMalloc((void **)&ext_memory, num_allocs);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, num_allocs, 1);

   cudaDeviceSynchronize();


   //this gets 0.
   //single_malloc_test_count_misses<<<(num_allocs-1)/512+1,512>>>(alloc, num_allocs, misses);

   malloc_test_cyclic<<<(num_allocs-1)/512+1,512>>>(alloc, num_allocs, num_rounds, misses);

   cudaDeviceSynchronize();

   fflush(stdout);
   printf("Done with mallocs, missed: %llu/%llu\n", misses[0], num_allocs*num_rounds);

   cudaDeviceSynchronize();


   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   cudaFree(ext_memory);

   cudaFree(misses);


   return true;



}


__host__ bool test_num_misses(uint64_t num_allocs){



   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   misses[0] = 0;


   char * ext_memory;

   cudaMalloc((void **)&ext_memory, num_allocs);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, num_allocs, 1);

   cudaDeviceSynchronize();


   //this gets 0.
   //single_malloc_test_count_misses<<<(num_allocs-1)/512+1,512>>>(alloc, num_allocs, misses);

   malloc_test_count_misses<<<(num_allocs-1)/512+1,512>>>(alloc, num_allocs, misses);

   cudaDeviceSynchronize();

   fflush(stdout);
   printf("Done with mallocs, missed: %llu/%llu\n", misses[0], num_allocs);

   cudaDeviceSynchronize();


   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   cudaFree(ext_memory);

   cudaFree(misses);


   return true;



}


__host__ bool test_num_allocs(uint64_t num_allocs){



   char ** grabbed_allocs;

   cudaMalloc((void **)&grabbed_allocs, sizeof(char *)*num_allocs);

   char * ext_memory;

   cudaMalloc((void **)&ext_memory, num_allocs);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, num_allocs, 1);

   cudaDeviceSynchronize();


   buddy_alloc_malloc_test_2<<<(num_allocs-1)/512+1,512>>>(alloc, num_allocs, grabbed_allocs);

   cudaDeviceSynchronize();

   fflush(stdout);
   printf("Done with mallocs\n");

   cudaDeviceSynchronize();

   assert_unique<<<(num_allocs-1)/512+1,512>>>(grabbed_allocs, num_allocs);

   cudaDeviceSynchronize();


   fflush(stdout);
   printf("Done with check\n");

   cudaDeviceSynchronize();

   //buddy_alloc_free_test_1<<<1,1>>>(alloc, 31, grabbed_allocs);

   buddy_alloc_free_test_2<<<(num_allocs-1)/512+1,512>>>(alloc, num_allocs, grabbed_allocs);

   cudaDeviceSynchronize();


   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   cudaFree(ext_memory);

   cudaFree(grabbed_allocs);


   return true;



}



__host__ bool test_num_allocs_repeat(uint64_t num_allocs, uint64_t num_loops){



   char ** grabbed_allocs;

   cudaMalloc((void **)&grabbed_allocs, sizeof(char *)*num_allocs);

   char * ext_memory;

   cudaMalloc((void **)&ext_memory, num_allocs);
   
   bitbuddy_allocator * alloc = bitbuddy_allocator::generate_on_device(ext_memory, num_allocs, 1);

   cudaDeviceSynchronize();


   for (uint64_t i = 0; i <  num_loops; i++){


      cudaMemset(grabbed_allocs, 0ULL, sizeof(char *)*num_allocs);


      buddy_alloc_malloc_test_2<<<(num_allocs-1)/512+1,512>>>(alloc, num_allocs, grabbed_allocs);

      cudaDeviceSynchronize();

      fflush(stdout);
      //printf("Done with mallocs\n");

      cudaDeviceSynchronize();

      assert_unique<<<(num_allocs-1)/512+1,512>>>(grabbed_allocs, num_allocs);

      cudaDeviceSynchronize();


      //fflush(stdout);
      //printf("Done with check\n");

      cudaDeviceSynchronize();

      //buddy_alloc_free_test_1<<<1,1>>>(alloc, 31, grabbed_allocs);

      buddy_alloc_free_test_2<<<(num_allocs-1)/512+1,512>>>(alloc, num_allocs, grabbed_allocs);

      cudaDeviceSynchronize();









   }

   bitbuddy_allocator::free_on_device(alloc);

   cudaDeviceSynchronize();


   
   cudaFree(ext_memory);

   cudaFree(grabbed_allocs);


   printf("Done with repetition test: %llu %llu\n", num_allocs, num_loops);


   return true;



}




//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   // if (!test_one_thread()){
   //    printf("Test one thread: [FAIL]\n");
   // } else {
   //    printf("Test one thread: [PASS]\n");
   // }

   // if (!test_one_thread_two()){
   //    printf("Test one thread two items: [FAIL]\n");
   // } else {
   //    printf("Test one thread two items: [PASS]\n");
   // }


   // if (!test_1()){
   //    printf("Test one thread 31 items: [FAIL]\n");
   // } else {
   //    printf("Test one thread 31 items: [PASS]\n");
   // }

   // if (!test_2()){
   //    printf("Test 2: [FAIL]\n");
   // } else {
   //    printf("Test 2: [PASS]\n");
   // }   


   // if (!test_num_allocs(32)){
   //    printf("Test 3: [FAIL]\n");
   // } else {
   //    printf("Test 3: [PASS]\n");
   // }   


   // if (!test_num_allocs_one_thread(33)){
   //    printf("Test 33 One Thread: [FAIL]\n");
   // } else {
   //    printf("Test 33 One Thread: [PASS]\n");
   // }  


   // if (!test_num_allocs_one_thread(1025)){
   //    printf("Tier 3 One Thread: [FAIL]\n");
   // } else {
   //    printf("Tier 3 One Thread: [PASS]\n");
   // }  


   if (!test_num_allocs_one_thread(4097)){
      printf("Tier 3 - 4097 One Thread: [FAIL]\n");
   } else {
      printf("Tier 3 - 4097 One Thread: [PASS]\n");
   }  


   if (!test_num_allocs(31)){
      printf("Test 4: [FAIL]\n");
   } else {
      printf("Test 4: [PASS]\n");
   }   

   if (!test_multi_threaded_boundary_singleton()){
      printf("Test multi 2: [FAIL]\n");
   } else {
      printf("Test multi 2: [PASS]\n");
   }   


   if (!test_multi_threaded_boundary()){
      printf("Test multi 2: [FAIL]\n");
   } else {
      printf("Test multi 2: [PASS]\n");
   }   


   if (!test_num_allocs(129)){
      printf("Test 4: [FAIL]\n");
   } else {
      printf("Test 4: [PASS]\n");
   }   


   if (!test_num_allocs(1025)){
      printf("Test 1025: [FAIL]\n");
   } else {
      printf("Test 1025: [PASS]\n");
   }   

   if (!test_num_allocs(2049)){
      printf("Test 2049: [FAIL]\n");
   } else {
      printf("Test 2049: [PASS]\n");
   }   


   if (!test_num_allocs(4097)){
      printf("Test 4097: [FAIL]\n");
   } else {
      printf("Test 4097: [PASS]\n");
   }   

    
   if (!test_num_allocs(8193)){
      printf("Test 8193: [FAIL]\n");
   } else {
      printf("Test 8193: [PASS]\n");
   }   



   for (int i = 5; i < 16; i++){


      test_num_misses((1ULL << i));

   }


   //these do fine.
   printf("repetition tests\n");
   for (int i=5; i < 10; i++){
      test_num_allocs_repeat((1ULL << i), 1000);
   }



   printf("Cycle tests 1\n");
   for (int i=5; i < 16; i++){
      test_cycles((1ULL << i), 1);
   }


   printf("Cycle tests 2\n");
   for (int i=5; i < 16; i++){
      test_cycles((1ULL << i), 2);
   }



   printf("Cycle tests 50\n");
   for (int i=5; i < 16; i++){
      test_cycles((1ULL << i), 50);
   }



   return 0;

}
