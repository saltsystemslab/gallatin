/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/representations/key_val_pair.cuh>

#include <poggers/representations/dynamic_container.cuh>
#include <poggers/representations/key_only.cuh>

#include <stdio.h>
#include <iostream>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace poggers::representations;

struct large_uint { 

   uint64_t internal_rep [8];

};

__host__ __device__ bool operator==(large_uint A, large_uint B){

   for (int i =0; i < 8; i++){
      if (A.internal_rep[i] != B.internal_rep[i]) return false;
   }

   return true;

}

__global__ void test_with_malloced(key_val_pair<uint64_t,uint64_t> * test){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   printf("should be 1: %d\n", test[0].atomic_swap(34, 0));

   printf("should be 0: %d\n", test[0].atomic_swap(34,1));

   printf("done\n\n");

}

__global__ void test_big_with_malloced(key_val_pair<large_uint,uint64_t> * test){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;


   large_uint to_insert;

   printf("size of to_insert: %llu\n", sizeof(to_insert));
   printf("Size of big key %llu\n", sizeof(key_val_pair<large_uint, uint64_t>));

   to_insert.internal_rep[0] = 34;

   printf("should be 1: %d\n", test[0].atomic_swap(to_insert, 0));

   printf("should be 0: %d\n", test[0].atomic_swap(to_insert,1));

   printf("done\n\n");

}

__global__ void test_key_val_pair(){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   printf("Thread %llu starting!\n", tid);

   key_val_pair<uint64_t, uint64_t> test(64,42);


    printf("should be 0: %d\n", test.atomic_swap(34, 0));

   key_val_pair<uint64_t, uint64_t> test3;

    printf("should be 1: %d\n", test3.atomic_swap(34, 0));




   // large_uint large_uint_test;

   // large_uint_test.internal_rep[0] =1234;

   // key_val_pair<large_uint, uint64_t> test2;

   //printf("%d\n", test2.atomic_swap(large_uint_test, 0));

   // if (test2.atomic_swap(large_uint_test, 0UL)){

   //    printf("first %llu\n", test2.key.internal_rep[0]);
   // } else {
   //    printf("Failure on swap\n");
   // }

   printf("done\n\n");

}


int main(int argc, char** argv) {

   key_val_pair<uint64_t, uint64_t> * test;

   cudaMalloc((void ** )& test, sizeof(key_val_pair<uint64_t, uint64_t>));
   cudaMemset(test, 0, sizeof(key_val_pair<uint64_t, uint64_t>));

   cudaDeviceSynchronize();

   test_with_malloced<<<1,1>>>(test);

   cudaDeviceSynchronize();

   cudaFree(test);

   key_val_pair<large_uint, uint64_t> * big_test;

   cudaMalloc((void ** )& big_test, sizeof(key_val_pair<large_uint, uint64_t>));
   cudaMemset(big_test, 0, sizeof(key_val_pair<large_uint, uint64_t>));

   cudaDeviceSynchronize();

   test_big_with_malloced<<<1,1>>>(big_test);

   using smallkeytype = dynamic_container<key_val_pair, uint16_t>::representation<uint64_t, uint16_t>;


   smallkeytype test_smallkey;


   using smallkeyonly = dynamic_container<key_container, uint16_t>::representation<uint64_t, uint16_t>;

   smallkeyonly test_smallkeyonly;

   //test_key_val_pair<<<100,100>>>();

   gpuErrorCheck(cudaPeekAtLastError());
   gpuErrorCheck(cudaDeviceSynchronize());


	return 0;

}
