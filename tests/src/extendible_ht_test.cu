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
__global__ void insert_ht_kernel(ht * table, uint64_t nitems){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems) return;

   table->insert(tid, tid);

}


template <typename Key, Key defaultKey, typename Val, int num_slots, uint64_t min_size, uint64_t max_size>
__host__ void extendible_ht_test(uint64_t num_bytes, double init_fill_ratio, double resize_ratio){


   using ht_type = gallatin::data_structs::extendible_hash_table<Key, defaultKey, Val, num_slots, min_size, max_size>;


   init_global_allocator(num_bytes, 42, false);

   auto my_table = ht_type::generate_on_device();

   // gallatin::utils::timer boot_timing;

   


   // ht_type * table;
   // cudaMalloc((void **)&table, sizeof(ht_type));

   // if (table == nullptr){
   //    printf("Failed to malloc table\n");
   //    free_global_allocator();
   //    return;

   // }

   uint64_t nitems = max_size*.9;

   cudaDeviceSynchronize();


   gallatin::utils::timer insert_timing;


   insert_ht_kernel<<<(nitems-1)/256+1, 256>>>(my_table, nitems);

   insert_timing.sync_end();

   insert_timing.print_throughput("Inserted", max_size);

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
   extendible_ht_test<uint64_t, 0ULL, uint64_t, 15, 256, 256>(num_segments*16*1024*1024, .4, .5);

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
