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
#include <poggers/insert_schemes/power_of_n_shortcut.cuh>

#include <poggers/sizing/variadic_sizing.cuh>

#include <poggers/representations/soa.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut_buckets.cuh>

#include <poggers/tables/bucketed_table.cuh>


#include <poggers/representations/12_bit_bucket.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut_buckets.cuh>
#include <poggers/representations/dynamic_container.cuh>
#include <poggers/representations/key_only.cuh>


#include <poggers/insert_schemes/grouped_power_buckets.cuh>

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


//using tiny_static_table_4 = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
//using tiny_static_table_4 = poggers::tables::bucketed_table<uint64_t, uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

//simple test version
 using tcf_twelve = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


//test version with old buckets
//this works
//using tcf = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::struct_of_arrays, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


//12-16
//using tcf = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

//12-8
//using tcf_twelve_8 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 8, poggers::insert_schemes::blocked_bucket_insert<10>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


//12-12
//using tcf_twelve_12 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 8, poggers::insert_schemes::blocked_bucket_insert<6>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


//12-32
//using tcf_twelve_32 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 32, poggers::insert_schemes::blocked_bucket_insert<2>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


//using tcf_no_back = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


//using tcf_16_16 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

//using tcf_16_32 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 8, 32, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


//using warpcore_bloom = warpcore::BloomFilter<uint64_t>;

//using tcf_8_8 = poggers::tables::static_table<uint64_t,uint8_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint8_t>::representation, 4, 8, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


uint64_t num_slots_per_p2(uint64_t nitems){

   //uint64_t nitems = .9*(1ULL << nbits);

   //for p=1/100, this is the correct value

   uint64_t nslots = 959*nitems/100;
   printf("using %llu slots\n", nslots);
   return nslots; 

}


template <typename T>
__host__ T * load_main_data(uint64_t nitems){


   char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/main_data-32-data.txt";

   //char main_location[] = "/pscratch/sd/h/hunterm/vqf_data/main_data-32-data.txt";

   char * vals = (char * ) malloc(nitems * sizeof(T));

   //std::ifstream myfile(main_location);

   //std::string line;


   FILE * pFile;


   pFile = fopen(main_location, "rb");

   if (pFile == NULL) abort();

   size_t result;

   result = fread(vals, 1, nitems*sizeof(T), pFile);

   if (result != nitems*sizeof(T)) abort();



   // //current supported format is no spacing one endl for the file terminator.
   // if (myfile.is_open()){


   //    getline(myfile, line);

   //    strncpy(vals, line.c_str(), sizeof(uint64_t)*nitems);

   //    myfile.close();
      

   // } else {

   //    abort();
   // }


   return (T *) vals;


}

template <typename T>
__host__ T * load_alt_data(uint64_t nitems){


   char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/fp_data-32-data.txt";

   //char main_location[] = "/pscratch/sd/h/hunterm/vqf_data/fp_data-32-data.txt";


   char * vals = (char * ) malloc(nitems * sizeof(T));


   //std::ifstream myfile(main_location);

   //std::string line;


   FILE * pFile;


   pFile = fopen(main_location, "rb");

   if (pFile == NULL) abort();

   size_t result;

   result = fread(vals, 1, nitems*sizeof(T), pFile);

   if (result != nitems*sizeof(T)) abort();



   return (T *) vals;


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
__global__ void find_first_fill(Filter * filter, Key * keys, Val * vals, uint64_t nitems, uint64_t * returned_nitems){


   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid != 0) return;

   // if (tile.thread_rank() == 0){

   //    for (int i = 0; i < 10; i++){
   //       printf("%d: %llu, %llu\n", i, keys[i], vals[i]);
   //    }
   // }


   //printf("Starting!\n");

   for (uint64_t i = 0; i < nitems; i++){


      if (!filter->insert(tile, keys[i])){

         if (tile.thread_rank() == 0){

            printf("Inserted %llu / %llu, %f full\n", i, nitems, 1.0*i/nitems);

         }

         returned_nitems[0] = i;

         return;

      } else {

         Val alt_val = 0;
         assert(filter->query(tile, keys[i], alt_val));
         assert(alt_val == vals[i]);


      }

      
   }

   if (tile.thread_rank() == 0) printf("All %llu items inserted\n", nitems);

}



template <typename Filter, typename Key, typename Val>
__global__ void speed_insert_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;


   if (!filter->insert(tile, keys[tid], vals[tid])){

   //    if (tile.thread_rank() == 0)
   //    atomicAdd((unsigned long long int *) misses, 1ULL);


   // } else{

   //    Val test_val = 0;
   //    assert(filter->query(tile, keys[tid], test_val));

   //    //assert(test_val == vals[tid]);
   }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}

template <typename Filter, typename Key, typename Val>
__global__ void debug_insert_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses, bool * missed){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;


   if (!filter->insert(tile, keys[tid], vals[tid])){

      //filter->insert(tile, keys[tid], vals[tid]);

      if (tile.thread_rank() == 0)
      atomicAdd((unsigned long long int *) misses, 1ULL);

      missed[tid] = true;


   } else{

      Val test_val = 0;
      assert(filter->query(tile, keys[tid], test_val));

      missed[tid] = false;

      //assert(test_val == vals[tid]);
   }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}

template <typename Filter, typename Key, typename Val>
__global__ void debug_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures, bool * missed){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   if (missed[tid]) return;

   Val test_val = 0;

   if (!filter->query(tile,keys[tid], test_val)){


      filter->query(tile,keys[tid], test_val);


      if(tile.thread_rank() == 0)
      atomicAdd((unsigned long long int *) query_misses, 1ULL);

   } else {


      // if (test_val != vals[tid] && tile.thread_rank() == 0){
      //    atomicAdd((unsigned long long int *) query_failures, 1ULL);
      // }

   }
   //assert(filter->query(tile, keys[tid], val));


}


template <typename Filter, typename Key, typename Val>
__global__ void speed_remove_kernel(Filter * filter, Key * keys, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;




   if (!filter->remove(tile, keys[tid]) && tile.thread_rank() == 0){
      atomicAdd((unsigned long long int *) misses, 1ULL);
   } 
      //else{

   //    Val test_val = 0;
   //    assert(filter->query(tile, keys[tid], test_val));
   // }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}

__global__ void count_bf_misses(bool * vals, uint64_t nitems, uint64_t * misses){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= nitems) return;


   if (!vals[tid]){
      atomicAdd((unsigned long long int *) misses, 1ULL);
   }
}

template <typename Filter, typename Key, typename Val>
__global__ void speed_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   Val test_val = 0;

   if (!filter->query(tile,keys[tid], test_val)){


   //    filter->query(tile,keys[tid], test_val);


   //    if(tile.thread_rank() == 0)
   //    atomicAdd((unsigned long long int *) query_misses, 1ULL);

   // } else {


      // if (test_val != vals[tid] && tile.thread_rank() == 0){
      //    atomicAdd((unsigned long long int *) query_failures, 1ULL);
      // }

   }
   //assert(filter->query(tile, keys[tid], val));


}


template <typename Filter, typename Key, typename Val>
__global__ void fp_speed_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   Val test_val = 0;

   if (!filter->query(tile,keys[tid], test_val)){


   //    filter->query(tile,keys[tid], test_val);


      if(tile.thread_rank() == 0)
      atomicAdd((unsigned long long int *) query_misses, 1ULL);

   // } else {


      // if (test_val != vals[tid] && tile.thread_rank() == 0){
      //    atomicAdd((unsigned long long int *) query_failures, 1ULL);
      // }

   }
   //assert(filter->query(tile, keys[tid], val));


}


template <typename Filter, typename Val>
__host__ void test_tcf_speed(const std::string& filename, int num_bits, int num_batches){


   using Key = uint64_t;
   //using Val = uint8_t;

   //using Filter = tcf;

   //std::cout << "Starting " << filename << " " << num_bits << std::endl;

   // poggers::sizing::size_in_num_slots<2> pre_init ((1ULL << num_bits), (1ULL << num_bits)/100);

   // poggers::sizing::size_in_num_slots<2> * Initializer = &pre_init;


   poggers::sizing::size_in_num_slots<1> pre_init ((1ULL << num_bits));

   poggers::sizing::size_in_num_slots<1> * Initializer = &pre_init;



   uint64_t nitems = Initializer->total()*.9;

   Key * host_keys = generate_data<Key>(nitems);
   Val * host_vals = generate_data<Val>(nitems);


   Key * fp_keys = generate_data<Key>(nitems);

   Key * dev_keys;

   Val * dev_vals;




   uint64_t * misses;

   cudaMallocManaged((void **)& misses, sizeof(uint64_t)*5);
   cudaDeviceSynchronize();

   //printf("Data generated\n");

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;

   //static seed for testing
   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

   cudaDeviceSynchronize();

   //init timing materials
   std::chrono::duration<double>  * insert_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
   std::chrono::duration<double>  * query_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
   std::chrono::duration<double>  * fp_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));

   std::chrono::duration<double>  * delete_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));



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


      bool * missed;

      cudaMalloc((void **)&missed, items_in_this_batch*sizeof(bool));



      //ensure GPU is caught up for next task
      cudaDeviceSynchronize();

      auto insert_start = std::chrono::high_resolution_clock::now();

      //add function for configure parameters - should be called by ht and return dim3
      speed_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_in_this_batch),test_filter->get_block_size(items_in_this_batch)>>>(test_filter, dev_keys, dev_vals, items_in_this_batch, misses);
      //debug_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_in_this_batch),test_filter->get_block_size(items_in_this_batch)>>>(test_filter, dev_keys, dev_vals, items_in_this_batch, misses, missed);
      
      cudaDeviceSynchronize();
      auto insert_end = std::chrono::high_resolution_clock::now();

      insert_diff[i] = insert_end-insert_start;

      cudaMemcpy(dev_keys, host_keys+start_of_batch, items_in_this_batch*sizeof(Key), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), cudaMemcpyHostToDevice);


      cudaDeviceSynchronize();

      auto query_start = std::chrono::high_resolution_clock::now();

      speed_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_in_this_batch),test_filter->get_block_size(items_in_this_batch)>>>(test_filter, dev_keys, dev_vals, items_in_this_batch, &misses[1], &misses[2]);
      //debug_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_in_this_batch),test_filter->get_block_size(items_in_this_batch)>>>(test_filter, dev_keys, dev_vals, items_in_this_batch, &misses[1], &misses[2], missed);
      
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

      cudaFree(missed);


   }

   //deletes
   // for (uint64_t i = 0; i < num_batches; i++){

   //    uint64_t start_of_batch = i*nitems/num_batches;
   //    uint64_t items_in_this_batch = (i+1)*nitems/num_batches;

   //    if (items_in_this_batch > nitems) items_in_this_batch = nitems;

   //    items_in_this_batch = items_in_this_batch - start_of_batch;


   //   // batch_amount[i] = items_in_this_batch;


   //    cudaMalloc((void **)& dev_keys, items_in_this_batch*sizeof(Key));
   //    //cudaMalloc((void **)& dev_vals, items_in_this_batch*sizeof(Val));


   //    cudaMemcpy(dev_keys, host_keys+start_of_batch, items_in_this_batch*sizeof(Key), cudaMemcpyHostToDevice);
   //    //cudaMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), cudaMemcpyHostToDevice);

   //    cudaDeviceSynchronize();

   //    auto delete_start = std::chrono::high_resolution_clock::now();

   //    speed_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_in_this_batch),test_filter->get_block_size(items_in_this_batch)>>>(test_filter, dev_keys, dev_vals, items_in_this_batch, &misses[1], &misses[2]);
   //    cudaDeviceSynchronize();
   //    auto delete_end = std::chrono::high_resolution_clock::now();


     
   //    delete_diff[i] = delete_end - delete_start;

   // }

   cudaDeviceSynchronize();


   Filter::free_on_device(test_filter);

   free(host_keys);
   free(host_vals);
   free(fp_keys);

   //free pieces

   //time to output


   //printf("nitems: %llu, insert misses: %llu, query missed: %llu, query wrong %llu, fp missed %llu, fp wrong %llu\n", nitems, misses[0], misses[1], misses[2], misses[3], misses[4]);

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

   std::chrono::duration<double> summed_delete_diff = std::chrono::nanoseconds::zero();

   for (int i =0; i < num_batches;i++){
      summed_delete_diff += delete_diff[i];
   }

  


   const uint64_t scaling_factor = 1000000ULL;



   double insert_throughput = nitems/(scaling_factor*summed_insert_diff.count());
      
   double lookup_throughput = nitems/(scaling_factor*summed_query_diff.count());

   double fp_throughput = nitems/(scaling_factor*summed_fp_diff.count());

   std::cout << "Name,  insert perf (M/s), lookup perf (M/s), FP perf (M/s)" << std::endl;
   std::cout << filename << ": " << insert_throughput << ", " << lookup_throughput << ", " << fp_throughput <<"." << std::endl;

   // std::cout << insert_file << std::endl;
   return;




}


template <typename Filter, typename Key, typename Val>
__host__ void tcf_find_first_fill(uint64_t num_bits){


   //std::cout << "Starting " << filename << " " << num_bits << std::endl;

   poggers::sizing::size_in_num_slots<1> pre_init ((1ULL << num_bits));

   poggers::sizing::size_in_num_slots<1> * Initializer = &pre_init;


   // poggers::sizing::size_in_num_slots<2> pre_init ((1ULL << num_bits), (1ULL << num_bits)/100);


   //  poggers::sizing::size_in_num_slots<2> * Initializer = &pre_init;

   // poggers::sizing::size_in_num_slots<1> pre_init ((1ULL << num_bits));

   // poggers::sizing::size_in_num_slots<1> * Initializer = &pre_init;



   uint64_t nitems = Initializer->total();

   Key * host_keys = generate_data<Key>(nitems);
   Val * host_vals = generate_data<Val>(nitems);


   Key * dev_keys;
   Val * dev_vals;


   // printf("Host keys\n");
   // for (int i = 0; i < 10; i++){
   //       printf("%d: %llu, %llu\n", i, host_keys[i], host_vals[i]);
   //    }

   uint64_t * misses;

   cudaMallocManaged((void ** )&misses, sizeof(uint64_t)*2);

   misses[0] = 0;
   misses[1] = 0;

   uint64_t * returned_nitems;
   cudaMallocManaged((void **)&returned_nitems, sizeof(uint64_t));  

   returned_nitems[0] = 0;

   cudaMalloc((void **)&dev_keys, sizeof(Key)*nitems);
   cudaMalloc((void **)&dev_vals, sizeof(Val)*nitems);

   cudaMemcpy(dev_keys, host_keys, sizeof(Key)*nitems, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_vals, host_vals, sizeof(Val)*nitems, cudaMemcpyHostToDevice);

   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

   printf("Test size: %llu\n", num_bits);

   cudaDeviceSynchronize();

   find_first_fill<Filter, Key, Val><<<1, 32>>>(test_filter, dev_keys, dev_vals, nitems, returned_nitems);

   cudaDeviceSynchronize();

   printf("Returned %llu\n", returned_nitems[0]);

   cudaMemcpy(dev_keys, host_keys, sizeof(Key)*nitems, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_vals, host_vals, sizeof(Val)*nitems, cudaMemcpyHostToDevice);

   cudaDeviceSynchronize();

   uint64_t new_nitems = returned_nitems[0];

   speed_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(new_nitems), test_filter->get_block_size(new_nitems)>>>(test_filter, dev_keys, dev_vals, new_nitems, &misses[0], &misses[1]);

   cudaDeviceSynchronize();

   printf("Final misses: initial misses %llu %f wrong values %llu %f\n", misses[0], 1.0*misses[0]/new_nitems, misses[1], 1.0*misses[1]/new_nitems);

   cudaDeviceSynchronize();

   cudaFree(misses);

   cudaFree(returned_nitems);

   Filter::free_on_device(test_filter);

   cudaFree(dev_keys);
   cudaFree(dev_vals);

}



// __host__ void test_first_fail(uint64_t num_bits){

//    //tcf_find_first_fill<tcf, uint16_t, uint16_t>(num_bits);

//    tcf_find_first_fill<tcf_no_back, uint64_t, uint16_t>(num_bits);

// }


__host__ void test_twelve_16(){

   printf("test_twelve_16\n");
   using tcf_twelve_16_1 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 1, 16, poggers::insert_schemes::blocked_bucket_insert<5>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_16_2 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 2, 16, poggers::insert_schemes::blocked_bucket_insert<5>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_16_4 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 16, poggers::insert_schemes::blocked_bucket_insert<5>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_16_8 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 8, 16, poggers::insert_schemes::blocked_bucket_insert<5>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_16_16 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 16, 16, poggers::insert_schemes::blocked_bucket_insert<5>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


   test_tcf_speed<tcf_twelve_16_1, uint16_t>("tcf_twelve_16_1 ", 28, 20);
   test_tcf_speed<tcf_twelve_16_2, uint16_t>("tcf_twelve_16_2 ", 28, 20);
   test_tcf_speed<tcf_twelve_16_4, uint16_t>("tcf_twelve_16_4 ", 28, 20);
   test_tcf_speed<tcf_twelve_16_8, uint16_t>("tcf_twelve_16_8 ", 28, 20);
   test_tcf_speed<tcf_twelve_16_16, uint16_t>("tcf_twelve_16_16", 28, 20);

   printf("\n\n\n");

}

__host__ void test_twelve_8(){

   printf("test_twelve_8\n");
   using tcf_twelve_8_1 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 1, 8, poggers::insert_schemes::blocked_bucket_insert<10>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_8_2 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 2, 8, poggers::insert_schemes::blocked_bucket_insert<10>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_8_4 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 8, poggers::insert_schemes::blocked_bucket_insert<10>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_8_8 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 8, 8, poggers::insert_schemes::blocked_bucket_insert<10>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

   test_tcf_speed<tcf_twelve_8_1, uint16_t>("tcf_twelve_8_1 ", 28, 20);
   test_tcf_speed<tcf_twelve_8_2, uint16_t>("tcf_twelve_8_2 ", 28, 20);
   test_tcf_speed<tcf_twelve_8_4, uint16_t>("tcf_twelve_8_4 ", 28, 20);
   test_tcf_speed<tcf_twelve_8_8, uint16_t>("tcf_twelve_8_8 ", 28, 20);
   
   printf("\n\n\n");

}

__host__ void test_twelve_12(){

   printf("test_twelve_12\n");
   using tcf_twelve_12_1 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 1, 12, poggers::insert_schemes::blocked_bucket_insert<6>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_12_2 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 2, 12, poggers::insert_schemes::blocked_bucket_insert<6>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_12_4 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 12, poggers::insert_schemes::blocked_bucket_insert<6>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_12_8 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 8, 8, poggers::insert_schemes::blocked_bucket_insert<5>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_16_16 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 16, 16, poggers::insert_schemes::blocked_bucket_insert<5>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


   test_tcf_speed<tcf_twelve_12_1, uint16_t>("tcf_twelve_12_1 ", 28, 20);
   test_tcf_speed<tcf_twelve_12_2, uint16_t>("tcf_twelve_12_2 ", 28, 20);
   test_tcf_speed<tcf_twelve_12_4, uint16_t>("tcf_twelve_12_4 ", 28, 20);
   test_tcf_speed<tcf_twelve_12_8, uint16_t>("tcf_twelve_12_8 ", 28, 20);
   test_tcf_speed<tcf_twelve_16_16, uint16_t>("tcf_twelve_16_16", 28, 20);

   printf("\n\n\n");

}

//using tcf_twelve_32 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 32, poggers::insert_schemes::blocked_bucket_insert<2>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

__host__ void test_twelve_32(){

   printf("test_twelve_32\n");
   using tcf_twelve_32_1 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 1, 32, poggers::insert_schemes::blocked_bucket_insert<2>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_32_2 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 2, 32, poggers::insert_schemes::blocked_bucket_insert<2>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_32_4 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 4, 32, poggers::insert_schemes::blocked_bucket_insert<2>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_32_8 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 8, 32, poggers::insert_schemes::blocked_bucket_insert<2>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_32_16 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 16, 32, poggers::insert_schemes::blocked_bucket_insert<2>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_twelve_32_32 = poggers::tables::bucketed_table<uint64_t,uint16_t, poggers::representations::wrapper_half_bucket<uint16_t>::representation, 32, 32, poggers::insert_schemes::blocked_bucket_insert<2>::representation, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


   test_tcf_speed<tcf_twelve_32_1, uint16_t>("tcf_twelve_32_1 ", 28, 20);
   test_tcf_speed<tcf_twelve_32_2, uint16_t>("tcf_twelve_32_2 ", 28, 20);
   test_tcf_speed<tcf_twelve_32_4, uint16_t>("tcf_twelve_32_4 ", 28, 20);
   test_tcf_speed<tcf_twelve_32_8, uint16_t>("tcf_twelve_32_8 ", 28, 20);
   test_tcf_speed<tcf_twelve_32_16, uint16_t>("tcf_twelve_32_16", 28, 20);
   test_tcf_speed<tcf_twelve_32_32, uint16_t>("tcf_twelve_32_32", 28, 20);

   printf("\n\n\n");

}

__host__ void test_sixteen_16(){

   printf("test_sixteen_16\n");
   using tcf_sixteen_16_1 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 1, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_sixteen_16_2 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 2, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

   using tcf_sixteen_16_4 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

   using tcf_sixteen_16_8 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 8, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

   using tcf_sixteen_16_16 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 16, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

   test_tcf_speed<tcf_sixteen_16_1, uint16_t>("tcf_sixteen_16_1 ", 28, 20);
   test_tcf_speed<tcf_sixteen_16_2, uint16_t>("tcf_sixteen_16_2 ", 28, 20);
   test_tcf_speed<tcf_sixteen_16_4, uint16_t>("tcf_sixteen_16_4 ", 28, 20);
   test_tcf_speed<tcf_sixteen_16_8, uint16_t>("tcf_sixteen_16_8 ", 28, 20);
   test_tcf_speed<tcf_sixteen_16_16, uint16_t>("tcf_sixteen_16_16", 28, 20);
   
   printf("\n\n\n");

}

__host__ void test_sixteen_32(){

   printf("test_sixteen_32\n");
   using tcf_sixteen_32_1 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 1, 32, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_sixteen_32_2 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 2, 32, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_sixteen_32_4 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 4, 32, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_sixteen_32_8 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 8, 32, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_sixteen_32_16 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 16, 32, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_sixteen_32_32 = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint16_t>::representation, 32, 32, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


   test_tcf_speed<tcf_sixteen_32_1, uint16_t>("tcf_sixteen_32_1 ", 28, 20);
   test_tcf_speed<tcf_sixteen_32_2, uint16_t>("tcf_sixteen_32_2 ", 28, 20);
   test_tcf_speed<tcf_sixteen_32_4, uint16_t>("tcf_sixteen_32_4 ", 28, 20);
   test_tcf_speed<tcf_sixteen_32_8, uint16_t>("tcf_sixteen_32_8 ", 28, 20);
   test_tcf_speed<tcf_sixteen_32_16, uint16_t>("tcf_sixteen_32_16", 28, 20);
   test_tcf_speed<tcf_sixteen_32_32, uint16_t>("tcf_sixteen_32_32", 28, 20);

   printf("\n\n\n");

}


__host__ void test_eight_8(){

   printf("test_8_8\n");
   using tcf_8_8_1 = poggers::tables::static_table<uint64_t,uint8_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint8_t>::representation, 1, 8, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_8_8_2 = poggers::tables::static_table<uint64_t,uint8_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint8_t>::representation, 2, 8, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_8_8_4 = poggers::tables::static_table<uint64_t,uint8_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint8_t>::representation, 4, 8, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
   using tcf_8_8_8 = poggers::tables::static_table<uint64_t,uint8_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint8_t>::representation, 8, 8, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;




   test_tcf_speed<tcf_8_8_1, uint8_t>("tcf_8_8_1 ", 28, 20);
   test_tcf_speed<tcf_8_8_2, uint8_t>("tcf_8_8_2 ", 28, 20);
   test_tcf_speed<tcf_8_8_4, uint8_t>("tcf_8_8_4 ", 28, 20);
   test_tcf_speed<tcf_8_8_8, uint8_t>("tcf_8_8_8 ", 28, 20);

   printf("\n\n\n");

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

   // poggers::sizing::variadic_size size(100000,100);
   // tcf * test_tcf = tcf::generate_on_device(&size, 42);


   // cudaDeviceSynchronize();

   // tcf::free_on_device(test_tcf);


   // warpcore_bloom my_filter((1ULL << 20), 7);


   // test_bloom_speed("bloom_results/test", 20, 20, true);
   // test_bloom_speed("bloom_results/test", 22, 20, false);
   // test_bloom_speed("bloom_results/test", 24, 20, false);
   // test_bloom_speed("bloom_results/test", 26, 20, false);
   // test_bloom_speed("bloom_results/test", 28, 20, false);
   //test_bloom_speed("bloom_results/test", 30, 20, false);

   //test_tcf_speed("results/test", 10, 20, true);
   

   //test_tcf_speed("results/test", 6, 1, true);
   //test_tcf_speed("results/test", 20, 1, true);

   printf("Starting CG Tests\n");

   test_eight_8();

   test_twelve_8();

   test_twelve_12();

   test_twelve_16();

   test_twelve_32();

   test_sixteen_16();

   test_sixteen_32();

   

   cudaDeviceSynchronize();

   printf("Tests over\n");



   // test_first_fail(22);
   // test_first_fail(24);
   // test_first_fail(26);
   // test_first_fail(28);
   // test_first_fail(30);


   return 0;

}
