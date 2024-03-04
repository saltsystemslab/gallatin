/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */





#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <gallatin/data_structs/ext_full_growing.cuh>

#include <openssl/rand.h>

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


template <typename T>
__host__ T * generate_data(uint64_t nitems){


   //malloc space

   T * vals;

   cudaMallocHost((void **)&vals, sizeof(T)*nitems);


   //          100,000,000
   uint64_t cap = 100000000ULL;

   for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

      uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


      RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



      to_fill += togen;

      //printf("Generated %llu/%llu\n", to_fill, nitems);

   }

   printf("Generation done\n");
   return gallatin::utils::move_to_device<T>(vals, nitems);
}


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


template <typename ht, typename key_type>
__global__ void insert_ht_kernel(ht * table, key_type * data, uint64_t nitems, uint64_t * misses, int n_rounds){


   auto my_tile = table->get_my_tile();

   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);


   //uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems) return;


   for (int i = 0; i < n_rounds; i++){

      if (!table->insert(data[tid], data[tid], my_tile)){

         if (my_tile.thread_rank() == 0) atomicAdd((unsigned long long int *)&misses[i], 1ULL);

         //if (my_tile.thread_rank() == 0) printf("Failed - Done with %llu\n", tid);

      }


   }

   //printf("Done with %llu\n", tid);

}



template <typename ht, typename key_type>
__global__ void query_ht_kernel(ht * table, key_type * data, uint64_t nitems, uint64_t * missed_key, uint64_t * missed_val){

   auto my_tile = table->get_my_tile();

   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);


   if (tid >= nitems) return;

   uint64_t val_read;

   if (!table->query(data[tid], val_read, my_tile)){

      if (my_tile.thread_rank() == 0) atomicAdd((unsigned long long int *)missed_key, 1ULL);
      //printf("Failed to query %llu\n", tid+1);

      return;
   }


   if (val_read != data[tid] && my_tile.thread_rank() == 0){

      atomicAdd((unsigned long long int *)missed_val, 1ULL);
      //printf("Failed to read correct query val %lu is not expected %lu\n", val_read, tid);
   }

   //if (my_tile.thread_rank() == 0) printf("Done with %llu\n", tid);

}


template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, int num_slots, uint64_t min_bits, uint64_t max_bits, int group_size>
__host__ void extendible_ht_test(uint64_t num_bytes, uint64_t nitems, int n_rounds){


   using ht_type = gallatin::data_structs::extendible_hash_table<Key, defaultKey, tombstoneKey, Val, num_slots, min_bits, max_bits, group_size>;


   init_global_allocator(num_bytes, 42, false);

   auto my_table = ht_type::generate_on_device();



   auto data = generate_data<Key>(nitems);

   // gallatin::utils::timer boot_timing;

   
   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*(n_rounds+2));

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


   insert_ht_kernel<<<((nitems-1)/TEST_BLOCK_SIZE+1)*group_size, TEST_BLOCK_SIZE>>>(my_table, data, nitems, misses, n_rounds);

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


   query_ht_kernel<<<((nitems-1)/TEST_BLOCK_SIZE+1)*group_size, TEST_BLOCK_SIZE>>>(my_table, data, nitems, misses+n_rounds, misses+n_rounds+1);

   query_timing.sync_end();

   query_timing.print_throughput("Queried", nitems);

   printf("Query missed %llu keys, %llu vals off\n", misses[n_rounds], misses[n_rounds+1]);

   cudaFree(misses);


   //print_global_stats();

   cudaDeviceSynchronize();


   ht_type::free_on_device(my_table);

   cudaFree(data);

   //printf("After ht free\n");

   //print_global_stats();

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
      num_segments = 1500;
   } else {
      num_segments = std::stoull(argv[1]);
   }

   if (argc < 3){
      num_inserts = 1000000;
   } else {
      num_inserts = std::stoull(argv[2]);
   }


   // printf(".2 .77\n");
   // gallatin_ht_noresize<uint64_t, uint64_t>(num_segments*16*1024*1024, num_inserts, .2, .77);


   // printf(".4 .77\n");
   // //gallatin_ht_noresize<uint64_t, uint64_t>(num_segments*16*1024*1024, num_inserts, .4, .77);

   // printf(".8 .77\n");
   // gallatin_ht_noresize<uint64_t, uint64_t>(num_segments*16*1024*1024, num_inserts, .8, .77);


   // printf(".2 .5\n");
   // gallatin_ht_noresize<uint64_t, uint64_t>(num_segments*16*1024*1024, num_inserts, .2, .5);

   // printf(".4 .5\n");
   // gallatin_ht_noresize<uint64_t, uint64_t>(num_segments*16*1024*1024, num_inserts, .4, .5);

   // printf(".8 .5\n");
   // gallatin_ht_noresize<uint64_t, uint64_t>(num_segments*16*1024*1024, num_inserts, .8, .5);



   //printf("Stride 0\n");
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 8, 20, 28>(num_segments*16*1024*1024, 200000000, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 8, 20, 28>(num_segments*16*1024*1024, 400000000, 1);

   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 3, 20, 29>(num_segments*16*1024*1024, 1000000000, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 7, 20, 28>(num_segments*16*1024*1024, 1000000000, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 15, 20, 27>(num_segments*16*1024*1024, 1000000000, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 31, 20, 26>(num_segments*16*1024*1024, 1000000000, 1);

   //extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 3, 20, 29, 1>(num_segments*16*1024*1024, 100000000, 1);

   extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 3, 20, 28, 1>(num_segments*16*1024*1024, 30000000, 1);
   extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 3, 20, 28, 2>(num_segments*16*1024*1024, 30000000, 1);
   extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 3, 20, 28, 4>(num_segments*16*1024*1024, 30000000, 1);
   

   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 7, 20, 28, 1>(num_segments*16*1024*1024, 30000000, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 7, 20, 28, 2>(num_segments*16*1024*1024, 30000000, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 7, 20, 28, 4>(num_segments*16*1024*1024, 30000000, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 7, 20, 28, 8>(num_segments*16*1024*1024, 30000000, 1);
  


   //extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 3, 20, 20, 4>(num_segments*16*1024*1024, 300000, 1);
   //extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 7, 20, 22, 1>(num_segments*16*1024*1024, 2000000, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 7, 20, 30>(num_segments*16*1024*1024, 200000000, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 15, 20, 30>(num_segments*16*1024*1024, 200000000, 1);
   //extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 31, 20, 30>(num_segments*16*1024*1024, 200000000, 1);
   //extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 3, 20, 28>(num_segments*16*1024*1024, 400000000, 1);

   //extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 3, 20, 28>(num_segments*16*1024*1024, 500000000, 1);

   //extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 3, 20, 28>(num_segments*16*1024*1024, 600000000, 1);



   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 8, 20, 28>(num_segments*16*1024*1024, 800000000, 1);
   // extendible_ht_test<uint64_t, 0ULL, ~0ULL, uint64_t, 8, 20, 28>(num_segments*16*1024*1024, 1000000000, 1);

   // printf("Stride 1\n");
   // gallatin_ht_noresize<uint64_t, 0ULL, uint64_t, 1>(num_segments*16*1024*1024, .4, .5);

   // printf("Stride 2\n");
   // gallatin_ht_noresize<uint64_t, 0ULL, uint64_t, 2>(num_segments*16*1024*1024, .4, .5);

   // printf("Stride 8\n");
   // gallatin_ht_noresize<uint64_t, uint64_t, 8>(num_segments*16*1024*1024, num_inserts, .4, .5);

   //gallatin_ht_noresize<uint64_t, uint64_t, 3>(num_segments*16*1024*1024, num_inserts, 2, .77);

   //gallatin_ht_noresize<uint32_t, uint32_t, 4>(num_segments*16*1024*1024, num_inserts, 2, .77);

   cudaDeviceSynchronize();

   cudaDeviceReset();
   return 0;

}
