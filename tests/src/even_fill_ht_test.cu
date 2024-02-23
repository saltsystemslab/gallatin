/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */


//This test is a variation of extendible_ht_test that forcibly inserts in an even pattern.
//trying to probe if the clipping functionality is correct.




#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <gallatin/data_structs/extendible_ht.cuh>


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


// template <typename ht_type>
// __global__ void init_ht_kernel(ht_type * table, uint64_t num_slots, uint64_t seed, double resize_ratio){

//    uint64_t tid = gallatin::utils::get_tid();

//    if (tid != 0) return;

//    table->init(num_slots, seed, resize_ratio);

// }

// template <typename ht_type>
// __global__ void insert_kernel(ht_type * table, uint64_t ninserts){

//    uint64_t tid = gallatin::utils::get_tid();

//    if (tid >= ninserts) return;

//    table->insert(tid+1, tid);

// }


// template <typename ht_type>
// __global__ void insert_kernel_single(ht_type * table, uint64_t ninserts){

//    uint64_t tid = gallatin::utils::get_tid();

//    if (tid != 0) return;

//    for (uint i = 0; i < ninserts; i++){
//       table->insert(i+1, i);
//    }
   

// }

// template <typename ht_type, typename Val>
// __global__ void query_kernel(ht_type * table, uint64_t ninserts){

//    uint64_t tid = gallatin::utils::get_tid();

//    if (tid >= ninserts) return;

//    Val temp_val;

//    if (!table->query(tid+1, temp_val)){

//       printf("%lu Failed to query %lu\n", tid, tid+1);


//       table->query(tid+1, temp_val);

//    }

// }

// template <typename Key, typename Val, int stride = 1>
// __host__ void gallatin_ht_noresize(uint64_t num_bytes, uint64_t num_inserts, double init_fill_ratio, double resize_ratio){


//    using ht_type = gallatin::data_structs::quad_table<Key, Val, stride>;

//    gallatin::utils::timer boot_timing;

//    init_global_allocator(num_bytes, 42, false);


//    ht_type * table;
//    cudaMalloc((void **)&table, sizeof(ht_type));

//    if (table == nullptr){
//       printf("Failed to malloc table\n");
//       free_global_allocator();
//       return;

//    }


//    init_ht_kernel<ht_type><<<1,1>>>(table, num_inserts*init_fill_ratio, 42, resize_ratio);

//    cudaDeviceSynchronize();

//    //generate bitarry
//    //space reserved is one 

//    // uint64_t * misses;
//    // cudaMallocManaged((void **)&misses, sizeof(uint64_t));

//    // cudaDeviceSynchronize();

//    // misses[0] = 0;

//    cudaDeviceSynchronize();

//    boot_timing.sync_end();

//    boot_timing.print_throughput("Booted", 1);

//    //and start kernel

//    gallatin::utils::timer insert_timing;

//    insert_kernel<ht_type><<<(num_inserts-1)/ TEST_BLOCK_SIZE +1, TEST_BLOCK_SIZE>>>(table, num_inserts);

//    //insert_kernel_single<ht_type><<<(num_inserts-1)/ TEST_BLOCK_SIZE +1, TEST_BLOCK_SIZE>>>(table, num_inserts);


//    insert_timing.sync_end();

//    gallatin::utils::timer query_timing;

//    query_kernel<ht_type, Val><<<(num_inserts-1)/ TEST_BLOCK_SIZE +1, TEST_BLOCK_SIZE>>>(table, num_inserts);

//    query_timing.sync_end();



//    insert_timing.print_throughput("Inserted", num_inserts);
//    query_timing.print_throughput("Queried", num_inserts);

//    free_global_allocator();


// }


template <typename ht>
__global__ void insert_ht_kernel(ht * table, uint64_t nitems, uint64_t * misses, int n_rounds){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems) return;

   if (tid == 0) return;


   for (int i = 0; i < n_rounds; i++){

      if (!table->insert(tid, tid+1)){

         table->insert(tid, tid+1);

         atomicAdd((unsigned long long int *)&misses[i], 1ULL);
      } else {
         return;
      }


   }



   //printf("Done with %llu\n", tid+1);

}


template <typename ht>
__global__ void query_ht_kernel(ht * table, uint64_t nitems, uint64_t * missed_key, uint64_t * missed_val){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems) return;

   uint64_t val_read;

   if (tid == 0) return;

   if (!table->query(tid, val_read)){

      atomicAdd((unsigned long long int *)missed_key, 1ULL);
      //printf("Failed to query %llu\n", tid+1);

      return;
   }


   if (val_read != tid+1){

      atomicAdd((unsigned long long int *)missed_val, 1ULL);
      //printf("Failed to read correct query val %lu is not expected %lu\n", val_read, tid);
   }

   //printf("Done with %llu\n", tid+1);

}


template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, int num_slots, uint64_t min_bits, uint64_t max_bits>
__host__ void extendible_ht_test(uint64_t num_bytes, uint64_t nitems, int n_rounds){


   using ht_type = gallatin::data_structs::extendible_hash_table<Key, defaultKey, tombstoneKey, Val, num_slots, min_bits, max_bits>;


   init_global_allocator(num_bytes, 42, false);

   auto my_table = ht_type::generate_on_device();

   // gallatin::utils::timer boot_timing;

   
   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*n_rounds+2);

   cudaDeviceSynchronize();

   for (int i =0; i < n_rounds+2; i++){
      misses[i] = 0ULL;
   }

   

   cudaDeviceSynchronize();


   // ht_type * table;
   // cudaMalloc((void **)&table, sizeof(ht_type));

   // if (table == nullptr){
   //    printf("Failed to malloc table\n");
   //    free_global_allocator();
   //    return;

   // }

   //uint64_t nitems = (1ULL << max_bits)*num_slots*.7;

   cudaDeviceSynchronize();


   gallatin::utils::timer insert_timing;


   insert_ht_kernel<<<(nitems-1)/TEST_BLOCK_SIZE+1, TEST_BLOCK_SIZE>>>(my_table, nitems, misses, n_rounds);

   insert_timing.sync_end();

   insert_timing.print_throughput("Inserted", nitems);

   //double fill = 0;
   double fill = my_table->calculate_fill(true);

   double exist_fill = my_table->calculate_fill(false);

   printf("Fill is %f - total fill %f\n", fill, exist_fill);

   for (int i = 0; i < n_rounds; i++){

      printf("Round %d missed %llu/%llu, %f\n",  i, misses[i], nitems, 1.0*misses[i]/nitems);

   }

   gallatin::utils::timer query_timing;


   query_ht_kernel<<<(nitems-1)/TEST_BLOCK_SIZE+1, TEST_BLOCK_SIZE>>>(my_table, nitems, misses+n_rounds, misses+n_rounds+1);

   query_timing.sync_end();

   query_timing.print_throughput("Queried", nitems);

   printf("Query missed %llu keys, %llu vals off\n", misses[n_rounds], misses[n_rounds+1]);

   cudaFree(misses);

   // init_ht_kernel<ht_type><<<1,1>>>(table, num_inserts*init_fill_ratio, 42, resize_ratio);

   // cudaDeviceSynchronize();

   // //generate bitarry
   // //space reserved is one 

   // // uint64_t * misses;
   // // cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   // // cudaDeviceSynchronize();

   // // misses[0] = 0;

   // cudaDeviceSynchronize();

   // boot_timing.sync_end();

   // boot_timing.print_throughput("Booted", 1);

   // //and start kernel

   // gallatin::utils::timer insert_timing;

   // insert_kernel<ht_type><<<(num_inserts-1)/ TEST_BLOCK_SIZE +1, TEST_BLOCK_SIZE>>>(table, num_inserts);

   // //insert_kernel_single<ht_type><<<(num_inserts-1)/ TEST_BLOCK_SIZE +1, TEST_BLOCK_SIZE>>>(table, num_inserts);


   // insert_timing.sync_end();

   // gallatin::utils::timer query_timing;

   // query_kernel<ht_type, Val><<<(num_inserts-1)/ TEST_BLOCK_SIZE +1, TEST_BLOCK_SIZE>>>(table, num_inserts);

   // query_timing.sync_end();



   // insert_timing.print_throughput("Inserted", num_inserts);
   // query_timing.print_throughput("Queried", num_inserts);

   cudaDeviceSynchronize();

   free_global_allocator();


}



int main(int argc, char** argv) {

   uint64_t num_segments;

   uint64_t num_inserts;


   if (argc < 2){
      num_segments = 1000;
   } else {
      num_segments = std::stoull(argv[1]);
   }

   if (argc < 3){
      num_inserts = 1000000;
   } else {
      num_inserts = std::stoull(argv[2]);
   }



   //printf("Stride 0\n");
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 1, 20, 20>(num_segments*16*1024*1024, 1048576, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 1, 20, 21>(num_segments*16*1024*1024, 2097152, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 1, 20, 22>(num_segments*16*1024*1024, 4194304, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 1, 20, 23>(num_segments*16*1024*1024, 8388608, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 1, 20, 24>(num_segments*16*1024*1024, 16777216, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 1, 20, 25>(num_segments*16*1024*1024, 33554432, 1);

   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 8, 20, 28>(num_segments*16*1024*1024, 2000000, 5);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 8, 20, 28>(num_segments*16*1024*1024, 3000000, 5);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 8, 20, 28>(num_segments*16*1024*1024, 4000000, 5);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 8, 20, 28>(num_segments*16*1024*1024, 5000000, 5);


   //extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 2, 20, 20>(num_segments*16*1024*1024, 2097152, 1);
   //extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 2, 20, 21>(num_segments*16*1024*1024, 4194304, 1);
   extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 2, 20, 22>(num_segments*16*1024*1024, 8388608, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 2, 20, 23>(num_segments*16*1024*1024, 16777216, 1);
   extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 2, 20, 24>(num_segments*16*1024*1024, 33554432, 1);
   extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 2, 20, 25>(num_segments*16*1024*1024, 67108864, 1);




   cudaDeviceSynchronize();

   cudaDeviceReset();
   return 0;

}
