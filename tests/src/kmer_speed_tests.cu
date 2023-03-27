/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 *
 *        About:
 *          This file contains speed tests for several Hash Table Types
 *          built using POGGERS. For more verbose testing please see the 
 *          benchmarks folder.
 *
 * ============================================================================
 */

#define DEBUG_PRINTS 1



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
#include <poggers/representations/dynamic_container.cuh>
#include <poggers/sizing/default_sizing.cuh>
#include <poggers/tables/base_table.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut.cuh>

#include <poggers/sizing/variadic_sizing.cuh>

#include <poggers/representations/soa.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut_buckets.cuh>

#include <poggers/representations/packed_bucket.cuh>
#include <poggers/representations/12_bit_bucket.cuh>


#include <poggers/insert_schemes/linear_insert_buckets.cuh>

#include <poggers/tables/bucketed_table.cuh>

#include <poggers/representations/grouped_key_val_pair.cuh>

#include <poggers/representations/grouped_storage_sub_bits.cuh>

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



//internal container

// class data_container {

//    template<typename Key, typename Val>
//    using representation = poggers::representations::dynamic_container<poggers::representations::key_val_pair, uint16_t>::representation<Key, Val>;

// };


// struct bucket_type {
//    template<typename Key, typename Val, std::size_t Partition_Size, std::size_t Bucket_Size>
//    using representation = poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<poggers::representations::key_val_pair, uint16_t>::representation>::representation<Key, Val, Partition_Size, Bucket_Size>;
// };


using grouped_items = typename poggers::representations::internal_key_val_storage<8,8>::type;

//32 bit grouped
using backing_table = poggers::tables::bucketed_table<uint64_t, uint8_t, poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<poggers::representations::bit_grouped_container<10,6>::representation, uint16_t>::representation>::representation, 1, 8, poggers::insert_schemes::linear_insert_bucket_scheme, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
using tcqf = poggers::tables::bucketed_table<uint64_t,uint8_t, poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<poggers::representations::bit_grouped_container<10,6>::representation, uint16_t>::representation>::representation, 1, 8, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, backing_table>;

//regular backend
// using backing_table = poggers::tables::bucketed_table<uint64_t, uint16_t, poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<poggers::representations::key_val_pair, uint16_t>::representation>::representation, 1, 8, poggers::insert_schemes::linear_insert_bucket_scheme, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
// using tcqf = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<poggers::representations::key_val_pair, uint16_t>::representation>::representation, 1, 8, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, backing_table>;

//16 bit grouped
// using backing_table = poggers::tables::bucketed_table<uint64_t, uint8_t, poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<poggers::representations::grouped_key_val_pair, uint8_t>::representation>::representation, 1, 8, poggers::insert_schemes::linear_insert_bucket_scheme, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
// using tcqf = poggers::tables::bucketed_table<uint64_t,uint8_t, poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<poggers::representations::grouped_key_val_pair, uint8_t>::representation>::representation, 1, 8, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, backing_table>;


//using tcqf_twelve = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


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
__global__ void speed_insert_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses, uint64_t * overwrite){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;


   Val ext_val = 0;
   bool found = false;

   // if (!filter->insert_with_delete(tile, keys[tid], vals[tid]) && tile.thread_rank() == 0){
   //    atomicAdd((unsigned long long int *) misses, 1ULL);
   // } 




   if (!filter->insert(tile, keys[tid], vals[tid])){

     //filter->insert_if_not_exists(tile, keys[tid], vals[tid], ext_val, found);

      if (tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *) misses, 1ULL);
      }



   } else {

      filter->query(tile, keys[tid], ext_val);



   }


   //if found, you deleted
   if (tile.thread_rank() == 0 && found){

         atomicAdd((unsigned long long int *) overwrite, 1ULL);

   }

   // if (!filter->insert_if_not_exists(tile, keys[tid], vals[tid], ext_val, found) && tile.thread_rank() == 0){
   //    atomicAdd((unsigned long long int *)misses, 1ULL);
   // }

}


template <typename Filter, typename Key, typename Val>
__global__ void speed_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   Val val = 0;
   val += 0;

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


   uint64_t * overwrite;

   cudaMallocManaged((void **)&overwrite, sizeof(uint64_t));

   cudaDeviceSynchronize();

   overwrite[0] = 0;

   //static seed for testing
   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

   cudaDeviceSynchronize();

   //print_tid_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems);

   cudaDeviceSynchronize();

   auto insert_start = std::chrono::high_resolution_clock::now();

   //add function for configure parameters - should be called by ht and return dim3
   speed_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems, misses, overwrite);
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
   printf("Overwrite was %llu\n", overwrite[0]);

   printf("End fill was: %llu/%llu\n", test_filter->get_fill(), test_filter->get_num_slots());

   cudaDeviceSynchronize();

   cudaFree(misses);

   cudaFree(overwrite);

   cudaDeviceSynchronize();

   cudaFree(dev_keys);
   cudaFree(dev_vals);

   Filter::free_on_device(test_filter);

   free(host_keys);
   free(host_vals);

}


template <typename Filter, typename Key, typename Val, typename Sizing_Type>
__host__ void test_speed_batched(const std::string& filename, Sizing_Type * Initializer, int num_batches){


   uint64_t nitems = Initializer->total()*.9;

   std::cout << "Starting " << filename << " with " << nitems << " items." << std::endl;

   

   Key * host_keys = generate_data<Key>(nitems);
   Val * host_vals = generate_data<Val>(nitems);


   Key * fp_keys = generate_data<Key>(nitems);

   Key * dev_keys;

   Val * dev_vals;




   uint64_t * misses;

   cudaMallocManaged((void **)& misses, sizeof(uint64_t)*5);
   cudaDeviceSynchronize();

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;


   uint64_t * overwrite;

   cudaMallocManaged((void **)&overwrite, sizeof(uint64_t));

   cudaDeviceSynchronize();

   overwrite[0] = 0;

   //static seed for testing
   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

   cudaDeviceSynchronize();

   printf("Filter using %llu bytes\n", test_filter->host_bytes_in_use());

   cudaDeviceSynchronize();

   //init timing materials
   std::chrono::duration<double>  * insert_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
   std::chrono::duration<double>  * query_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
   std::chrono::duration<double>  * fp_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));

   uint64_t * batch_amount = (uint64_t *) malloc(num_batches*sizeof(uint64_t));

   //print_tid_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems);


   for (uint64_t i = 0; i < num_batches; i++){

      uint64_t start_of_batch = i*nitems/num_batches;
      uint64_t items_in_this_batch = (i+1)*nitems/num_batches;

      if (items_in_this_batch > nitems) items_in_this_batch = nitems;

      items_in_this_batch = items_in_this_batch - start_of_batch;


      batch_amount[i] = items_in_this_batch;


      cudaMalloc((void **)& dev_keys, items_in_this_batch*sizeof(Key));
      cudaMalloc((void **)& dev_vals, items_in_this_batch*sizeof(Val));


      cudaMemcpy(dev_keys, host_keys+start_of_batch, items_in_this_batch*sizeof(Key), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), cudaMemcpyHostToDevice);



      //ensure GPU is caught up for next task
      cudaDeviceSynchronize();

      auto insert_start = std::chrono::high_resolution_clock::now();

      //add function for configure parameters - should be called by ht and return dim3
      speed_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_in_this_batch),test_filter->get_block_size(items_in_this_batch)>>>(test_filter, dev_keys, dev_vals, items_in_this_batch, misses, overwrite);
      cudaDeviceSynchronize();
      auto insert_end = std::chrono::high_resolution_clock::now();

      insert_diff[i] = insert_end-insert_start;

      cudaMemcpy(dev_keys, host_keys+start_of_batch, items_in_this_batch*sizeof(Key), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), cudaMemcpyHostToDevice);


      cudaDeviceSynchronize();

      auto query_start = std::chrono::high_resolution_clock::now();

      speed_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_in_this_batch),test_filter->get_block_size(items_in_this_batch)>>>(test_filter, dev_keys, dev_vals, items_in_this_batch, &misses[1], &misses[2]);
      cudaDeviceSynchronize();
      auto query_end = std::chrono::high_resolution_clock::now();



      query_diff[i] = query_end - query_start;

      cudaMemcpy(dev_keys, fp_keys+start_of_batch, items_in_this_batch*sizeof(Key), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), cudaMemcpyHostToDevice);


      cudaDeviceSynchronize();

      auto fp_start = std::chrono::high_resolution_clock::now();

      speed_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_in_this_batch),test_filter->get_block_size(items_in_this_batch)>>>(test_filter, dev_keys, dev_vals, items_in_this_batch, &misses[3], &misses[4]);


      cudaDeviceSynchronize();
      auto fp_end = std::chrono::high_resolution_clock::now();

      fp_diff[i] = fp_end-fp_start;


      cudaFree(dev_keys);
      cudaFree(dev_vals);


   }

   cudaDeviceSynchronize();


   printf("End fill was: %llu/%llu\n", test_filter->get_fill(), test_filter->get_num_slots());

   Filter::free_on_device(test_filter);

   free(host_keys);
   free(host_vals);
   free(fp_keys);

   //free pieces

   //time to output




   printf("%llu %llu %llu %llu %llu %llu\n", nitems, misses[0], misses[1], misses[2], misses[3], misses[4]);
   printf("Overwrite was %llu\n", overwrite[0]);

   
   cudaFree(misses);
   cudaFree(overwrite);


   std::chrono::duration<double> summed_insert_diff = std::chrono::nanoseconds::zero();

   for (int i =0; i < num_batches;i++){
      summed_insert_diff += insert_diff[i];
   }

   std::chrono::duration<double> summed_query_diff = std::chrono::nanoseconds::zero();

   for (int i =0; i < num_batches;i++){
      summed_query_diff += query_diff[i];
   }

   std::chrono::duration<double> summed_fp_diff = std::chrono::nanoseconds::zero();

   for (int i =0; i < num_batches;i++){
      summed_fp_diff += fp_diff[i];
   }

   std::string insert_file = filename + "_insert.txt";
   std::string query_file = filename + "_lookup.txt";
   std::string fp_file = filename + "_fp.txt";
   std::string agg_file = filename + "_aggregate.txt";


   std::cout << insert_file << std::endl;


   FILE *fp_insert = fopen(insert_file.c_str(), "w");
   FILE *fp_lookup = fopen(query_file.c_str(), "w");
   FILE *fp_false_lookup = fopen(fp_file.c_str(), "w");
   FILE *fp_agg = fopen(agg_file.c_str(), "w");


   if (fp_insert == NULL) {
    printf("Can't open the data file %s\n", insert_file);
    exit(1);
   }

   if (fp_lookup == NULL ) {
      printf("Can't open the data file %s\n", query_file);
      exit(1);
   }

   if (fp_false_lookup == NULL) {
      printf("Can't open the data file %s\n", fp_file);
      exit(1);
   }

   if (fp_agg == NULL) {
      printf("Can't open the data file %s\n", agg_file);
      exit(1);
   }


   //inserts

   printf("Writing results to file: %s\n",  insert_file);

   fprintf(fp_insert, "x_0 y_0\n");
   for (int i = 0; i < num_batches; i++){
      fprintf(fp_insert, "%d", i*100/num_batches);

      fprintf(fp_insert, " %f\n", batch_amount[i]/insert_diff[i].count());
   }


   //queries
   printf("Writing results to file: %s\n",  query_file);

   fprintf(fp_lookup, "x_0 y_0\n");
   for (int i = 0; i < num_batches; i++){
      fprintf(fp_lookup, "%d", i*100/num_batches);

      fprintf(fp_lookup, " %f\n", batch_amount[i]/query_diff[i].count());
   }


   printf("Writing results to file: %s\n",  fp_file);

   fprintf(fp_false_lookup, "x_0 y_0\n");
   for (int i = 0; i < num_batches; i++){
      fprintf(fp_false_lookup, "%d", i*100/num_batches);

      fprintf(fp_false_lookup, " %f\n", batch_amount[i]/fp_diff[i].count());
   }

   fprintf(fp_agg, "Aggregate inserts: %f\n", nitems/summed_insert_diff.count());
   fprintf(fp_agg, "Aggregate Queries: %f\n", nitems/summed_query_diff.count());
   fprintf(fp_agg, "Aggregate fp: %f\n", nitems/summed_fp_diff.count());

   fclose(fp_insert);
   fclose(fp_lookup);
   fclose(fp_false_lookup);
   fclose(fp_agg);

   return;




}



__host__ poggers::sizing::size_in_num_slots<2> get_tcf_sizing(uint64_t num_bits){

  uint64_t max_num_kmers = (1ULL << num_bits);

  uint64_t max_slots = max_num_kmers*1.2;

  //90/11 split over size for forward and backing tables.

  poggers::sizing::size_in_num_slots<2> my_size(max_slots*90ULL/100ULL, max_slots*10ULL/100ULL);
  return my_size;

}


__host__ __device__ uint8_t pack_extensions_small(char left, char right){

   char kmer_ext[6] = {'F', 'A', 'C', 'T', 'G', '0'};

  uint8_t ret_val = 0;

  for (uint i = 0; i < 6; i++){

    if (left == kmer_ext[i]){
      ret_val += i << 4;
    }

    if (right == kmer_ext[i]){
      ret_val += i;
    }

  }

  return ret_val;


}

__host__ __device__ bool unpack_extensions_small(uint8_t storage, char & left, char & right){

   char kmer_ext[6] = {'F', 'A', 'C', 'T', 'G', '0'};

  uint8_t left_val = ((storage & 0xf0) >> 4);

  uint8_t right_val = (storage & 0x0f);

  if ((left_val < 6) && (right_val < 6)){

    left = kmer_ext[left_val];
    right = kmer_ext[right_val];

    return true;
  } else {

    return false;
  }

}

int main(int argc, char** argv) {

   printf("Grouped item size: %d\n", sizeof(grouped_items));

   grouped_items test;

   for (uint16_t i = 0; i < 256; i++){

      for (uint16_t j=0; j < 256; j++){

         test = poggers::representations::join_in_storage<uint8_t, uint8_t, grouped_items, 8, 8>(i,j);

         uint16_t stored_key = poggers::representations::retrieve_key_from_storage<uint8_t, uint8_t, grouped_items, 8, 8>(test);


         uint16_t stored_val = poggers::representations::retrieve_val_from_storage<uint8_t, uint8_t, grouped_items, 8, 8>(test);

         if (stored_key != i || stored_val != j){
            printf("Displaying %lx, %u - %u, %u - %u\n", test, i, stored_key, j, stored_val);
         }


      }
   }

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

   // char kmer_ext[6] = {'F', 'A', 'C', 'T', 'G', '0'};

   // for (int i=0; i < 6; i++){

   //    for (int j =0; j < 6; j++){


   //       char my_left = kmer_ext[i];
   //       char my_right = kmer_ext[j];

   //       uint8_t compressed = pack_extensions_small(my_left, my_right);


   //       char new_left;
   //       char new_right;

   //       unpack_extensions_small(compressed, new_left, new_right);

   //       if (new_left != my_left || new_right != my_right){
   //          printf("Bugggg\n");
   //       }

   //    }

   // }

   auto test_size_20 = get_tcf_sizing(20);


   auto test_24  = get_tcf_sizing(24);
   auto test_26  = get_tcf_sizing(26);


   // //printf("22 size: %llu\n", test_size_24.total());


   test_speed_batched<tcqf, uint64_t, uint8_t>("results/test_20", &test_size_20, 20);
   test_speed_batched<tcqf, uint64_t, uint8_t>("results/test_24", &test_24, 20);
   test_speed_batched<tcqf, uint64_t, uint8_t>("results/test_26", &test_26, 20);





   // test_speed_batched<tcqf, uint64_t, uint16_t>("results/test_28", generate_size(28), 20);
   // test_speed_batched<tcqf, uint64_t, uint16_t>("results/test_30", generate_size(30), 20);

   cudaDeviceSynchronize();



   // poggers::sizing::size_in_num_slots<1> bucket_size (1ULL<<nbits);

   // test_speed_batched<double_buckets, uint64_t,uint64_t>("results/double_buckets", &bucket_size, 20);

   // cudaDeviceSynchronize();

   // printf("alt table\n");

   // test_p2(6000);

   // test_p2(1ULL << 22);
   // test_p2(1ULL << 24);
   // test_p2(1ULL << 26);
   // test_p2(1ULL << 28);
   // test_p2(1ULL << 30);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_22);

   // poggers::sizing::size_in_num_slots<2>half_split_24(1ULL << 23, 1ULL << 23);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_24);

   // poggers::sizing::size_in_num_slots<2>half_split_26(1ULL << 25, 1ULL << 25);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_26);


   // printf("P2 tiny table\n");
   // poggers::sizing::size_in_num_slots<1>half_split_28(1ULL << 30);
   // test_speed<p2_table, uint64_t, uint16_t>(&half_split_28);


   //printf("Icerberg - Tier one\n");


   //poggers::sizing::size_in_num_slots<1> tier_one_iceberg_size(1ULL << 28);
   //test_speed<tier_one_iceberg, uint64_t, uint64_t>(&tier_one_iceberg_size);


   // printf("Icerberg - Tier two\n");

   // //this section is allocated 1/8th of the space as tier one
   // poggers::sizing::size_in_num_slots<1> tier_two_iceberg_size((1ULL << 28)/8);
   // test_speed<tier_two_icerberg, uint64_t, uint64_t>(&tier_two_iceberg_size);

   // printf("Icerberg - Tier three\n");

   // poggers::sizing::size_in_num_slots<1> tier_three_iceberg_size((1500));
   // test_speed<tier_three_iceberg, uint64_t, uint64_t>(&tier_three_iceberg_size);


   // printf("Icerberg - Joined\n");

   // poggers::sizing::size_in_num_slots<3> iceberg_size((1ULL << 28), (1ULL << 28)/8, 1500);
   // test_speed<iceberg_table, uint64_t, uint64_t>(&iceberg_size);


	return 0;

}
