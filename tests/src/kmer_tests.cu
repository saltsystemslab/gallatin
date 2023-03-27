/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 *
 *        About:
 *          This file contains k-mer speed tests for several Hash Table Types
 *          built using POGGERS. For more verbose testing please see the 
 *          benchmarks folder.
 *
 * ============================================================================
 */




//#include "include/templated_quad_table.cuh"
#include <poggers/metadata.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/probing_schemes/linear_probing.cuh>
#include <poggers/probing_schemes/double_hashing.cuh>
#include <poggers/probing_schemes/power_of_two.cuh>
#include <poggers/insert_schemes/single_slot_insert.cuh>
#include <poggers/insert_schemes/bucket_insert.cuh>
#include <poggers/insert_schemes/power_of_n.cuh>
#include <poggers/representations/key_val_pair.cuh>
#include <poggers/representations/shortened_key_val_pair.cuh>
#include <poggers/sizing/default_sizing.cuh>
#include <poggers/tables/base_table.cuh>

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <openssl/rand.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>



template <size_t size_in_bytes> struct kmer {

 char data [size_in_bytes];


 operator int() const { return data[0]; }

 kmer& operator=(int i){
    data[0] = (char) i;
    return *this;
  }

};


template <std::size_t size_in_bytes>
__host__ __device__ bool operator==(kmer<size_in_bytes> A, kmer<size_in_bytes> B){

   for (int i=0; i < size_in_bytes; i++){

      if (A.data[i] != B.data[i]) return false;
   }

   return true;

}


using key_type = kmer<20>;
using val_type = uint64_t;



//using insert_type = poggers::insert_schemes::single_slot_insert<uint64_t, uint64_t, 8, 8, poggers::representations::key_val_pair, 5, poggers::hashers::murmurHasher, poggers::probing_schemes::doubleHasher>;

using table_type = poggers::tables::static_table<key_type, val_type, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 200, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
     // poggers::representations::key_val_pair, 8>

     //using forst_tier_table_type = poggers::tables::static_table<key_type, val_type, poggers::representations::key_val_pair, 8, poggers::insert_schemes::single_slot_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
    
using second_tier_table_type = poggers::tables::static_table<key_type, val_type, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::single_slot_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, table_type>;

using inner_table = poggers::tables::static_table<key_type, val_type, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

using small_double_type = poggers::tables::static_table<key_type, val_type, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, inner_table>;

//using p2_table = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 8, 32, poggers::insert_schemes::power_of_n_insert_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

   
using tier_one_iceberg = poggers::tables::static_table<key_type, val_type, poggers::representations::key_val_pair, 8, 64, poggers::insert_schemes::bucket_insert, 1, poggers::probing_schemes::linearProber, poggers::hashers::murmurHasher>;

using tier_two_icerberg = poggers::tables::static_table<key_type, val_type, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::power_of_n_insert_scheme, 2, poggers::probing_schemes::powerOfTwoHasher, poggers::hashers::murmurHasher>;

using tier_three_iceberg = poggers::tables::static_table<key_type, val_type, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 10, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


using tier_two_icerberg_joined = poggers::tables::static_table<key_type, val_type, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::power_of_n_insert_scheme, 2, poggers::probing_schemes::powerOfTwoHasher, poggers::hashers::murmurHasher>;

using iceberg_table = poggers::tables::static_table<key_type, val_type, poggers::representations::key_val_pair, 8, 64, poggers::insert_schemes::bucket_insert, 1, poggers::probing_schemes::linearProber, poggers::hashers::murmurHasher, true, tier_two_icerberg_joined>;



#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
__host__ T * generate_data(uint64_t nitems){


   //malloc space

   T * vals = (T *) malloc(nitems * sizeof(T));


   //          100,000,000
   uint64_t cap = 100000000ULL;

   for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

      uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


      RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



      to_fill += togen;

      //printf("Generated %llu/%llu\n", to_fill, nitems);

   }

   return vals;
}


template <typename Filter, typename Key, typename Val>
__global__ void print_tid_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals){


      auto tile = filter->get_my_tile();

      uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

      if (tid >= nvals) return;


      if (tile.thread_rank() == 0) printf("%llu: %d, %d\n", tid, keys[tid], vals[tid]);

}

template <typename Filter, typename Key, typename Val>
__global__ void speed_insert_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;




   if (!filter->insert(tile, keys[tid], vals[tid]) && tile.thread_rank() == 0){
      atomicAdd((unsigned long long int *) misses, 1ULL);
   } 
      //else{

   //    Val test_val = 0;
   //    assert(filter->query(tile, keys[tid], test_val));
   // }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}

template <typename Filter, typename Key, typename Val>
__global__ void speed_insert_kernel_one_thread(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid2 = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid2 > 0) return;


   for (uint64_t tid=0; tid < nvals; tid++){

      if (tid % 10 == 0 && tile.thread_rank() == 0) printf("%llu\n", tid);

      if (!filter->insert(tile, keys[tid], vals[tid]) && tile.thread_rank() == 0){
      atomicAdd((unsigned long long int *) misses, 1ULL);
   } else{

      Val test_val = 0;
      assert(filter->query(tile, keys[tid], test_val));
   }


   }





   //assert(filter->insert(tile, keys[tid], vals[tid]));


}



template <typename Filter, typename Key, typename Val>
__global__ void speed_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   Val val = 0;
   //val += 0;

   if (!filter->query(tile,keys[tid], val) && tile.thread_rank() == 0){
      atomicAdd((unsigned long long int *) query_misses, 1ULL);
   } else {

      if (val != vals[tid] && tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *) query_failures, 1ULL);
      }

   }
   //assert(filter->query(tile, keys[tid], val));


}

template <typename Filter, typename Key, typename Val, typename Sizing_Type>
__host__ void test_speed(Sizing_Type * Initializer){

   uint64_t nitems = Initializer->total()*.9;

   Key * host_keys = generate_data<Key>(nitems);
   Val * host_vals = generate_data<Val>(nitems);

   Key * dev_keys;

   Val * dev_vals;

   cudaMalloc((void **)& dev_keys, nitems*sizeof(Key));
   cudaMalloc((void **)& dev_vals, nitems*sizeof(Val));

   cudaMemcpy(dev_keys, host_keys, nitems*sizeof(Key), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_vals, host_vals, nitems*sizeof(Val), cudaMemcpyHostToDevice);


   uint64_t * misses;

   cudaMallocManaged((void **)& misses, sizeof(uint64_t)*3);
   cudaDeviceSynchronize();

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;

   //static seed for testing
   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

   cudaDeviceSynchronize();

   //print_tid_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems);

   cudaDeviceSynchronize();

   auto insert_start = std::chrono::high_resolution_clock::now();

   //add function for configure parameters - should be called by ht and return dim3
   speed_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems, misses);
   cudaDeviceSynchronize();
   auto insert_end = std::chrono::high_resolution_clock::now();


   cudaMemcpy(dev_keys, host_keys, nitems*sizeof(Key), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_vals, host_vals, nitems*sizeof(Val), cudaMemcpyHostToDevice);


   cudaDeviceSynchronize();

   auto query_start = std::chrono::high_resolution_clock::now();

   speed_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems, &misses[1], &misses[2]);
   cudaDeviceSynchronize();
   auto query_end = std::chrono::high_resolution_clock::now();




   std::chrono::duration<double> insert_diff = insert_end-insert_start;
   std::chrono::duration<double> query_diff = query_end-query_start;


   cudaDeviceSynchronize();
   std::cout << "Inserted " << nitems << " in " << insert_diff.count() << " seconds\n";
   std::cout << "Queried " << nitems << " in " << query_diff.count() << " seconds\n";

   printf("Inserts/Queries: %f / %f\n", 1.0*nitems/insert_diff.count(), 1.0*nitems/query_diff.count());
   printf("%llu / %llu / %llu\n", misses[0], misses[1], misses[2]);

   cudaDeviceSynchronize();

   cudaFree(misses);

   cudaDeviceSynchronize();

   cudaFree(dev_keys);
   cudaFree(dev_vals);

   Filter::free_on_device(test_filter);

   free(host_keys);
   free(host_vals);

}


int main(int argc, char** argv) {

   // poggers::sizing::size_in_num_slots<1> first_size_20(1ULL << 20);
   // printf("2^20\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_20);

   // poggers::sizing::size_in_num_slots<1> first_size_22(1ULL << 22);
   // printf("2^22\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_22);

   // poggers::sizing::size_in_num_slots<1> first_size_24(1ULL << 24);
   // printf("2^24\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_24);

   // poggers::sizing::size_in_num_slots<1> first_size_26(1ULL << 26);
   // printf("2^26\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_26);

   // poggers::sizing::size_in_num_slots<1> first_size_28(1ULL << 28);
   // printf("2^28\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_28);


   // printf("alt table\n");

   // poggers::sizing::size_in_num_slots<1>half_split_20(6000);
   // test_speed<p2_table, key_type, val_type>(&half_split_20);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_22);

   // poggers::sizing::size_in_num_slots<2>half_split_24(1ULL << 23, 1ULL << 23);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_24);

   // poggers::sizing::size_in_num_slots<2>half_split_26(1ULL << 25, 1ULL << 25);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_26);


//   printf("P2 tiny table\n");
   // poggers::sizing::size_in_num_slots<1>half_split_28(1ULL << 28);
   // test_speed<p2_table, key_type, val_type>(&half_split_28);


   printf("Double Hashing\n");

   poggers::sizing::size_in_num_slots<1> double_hashing_size(1ULL << 10);
   test_speed<table_type, key_type, val_type>(&double_hashing_size);

   printf("Icerberg - Tier one\n");


   poggers::sizing::size_in_num_slots<1> tier_one_iceberg_size(1ULL << 10);
   test_speed<tier_one_iceberg, key_type, val_type>(&tier_one_iceberg_size);


   printf("Icerberg - Tier two\n");

   //this section is allocated 1/8th of the space as tier one
   poggers::sizing::size_in_num_slots<1> tier_two_iceberg_size((1ULL << 10)/8);
   test_speed<tier_two_icerberg, key_type, val_type>(&tier_two_iceberg_size);

   printf("Icerberg - Tier three\n");

   poggers::sizing::size_in_num_slots<1> tier_three_iceberg_size((1500));
   test_speed<tier_three_iceberg, key_type, val_type>(&tier_three_iceberg_size);


   printf("Icerberg - Joined\n");

   poggers::sizing::size_in_num_slots<3> iceberg_size((1ULL << 10), (1ULL << 10)/8, 1500);
   test_speed<iceberg_table, key_type, val_type>(&iceberg_size);


	return 0;

}
