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

#include <gallatin/data_structs/coop_chaining_table.cuh>


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



template <typename ht_type, uint team_size>
__global__ void insert_kernel(ht_type * table, uint64_t ninserts){

   

   cg::thread_block block = cg::this_thread_block();
   cg::thread_block_tile<team_size> team = cg::tiled_partition<team_size>(block);

   uint64_t tid = gallatin::utils::get_team_tid(team);

   if (tid >= ninserts) return;

   table->insert(team, tid+1, tid);

}


template <typename ht_type, typename Val, uint team_size>
__global__ void query_kernel(ht_type * table, uint64_t ninserts){

   cg::thread_block block = cg::this_thread_block();
   cg::thread_block_tile<team_size> team = cg::tiled_partition<team_size>(block);

   uint64_t tid = gallatin::utils::get_team_tid(team);

   if (tid >= ninserts) return;

   Val temp_val;

   if (!table->query(team, tid+1, temp_val)){

      if (team.thread_rank() == 0) printf("%lu Failed to query %lu\n", tid, tid+1);

   }

}

template <typename Key, typename Val, uint size, uint team_size>
__host__ void gallatin_ht_test(uint64_t num_bytes, uint64_t num_inserts){


   std::cout << "Table with " << typeid(Key).name() << " keys, " << typeid(Val).name() << " vals, and " << size << " slots per bucket, cg " << team_size << std::endl;

   using ht_type = gallatin::data_structs::coop_chaining_table<Key, Val, size, team_size>;

   gallatin::utils::timer boot_timing;

   init_global_allocator(num_bytes, 42);
   
   ht_type * table = ht_type::generate_on_device(num_inserts, 0, 123456ULL);

   //ht_type * table;
   //cudaMalloc((void **)&table, sizeof(ht_type));

   if (table == nullptr){
      printf("Failed to malloc table\n");
      free_global_allocator();
      return;

   }


   //init_ht_kernel<ht_type><<<1,1>>>(table, num_inserts*.5, 42);

   cudaDeviceSynchronize();

   //generate bitarry
   //space reserved is one 

   // uint64_t * misses;
   // cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   // cudaDeviceSynchronize();

   // misses[0] = 0;

   cudaDeviceSynchronize();

   boot_timing.sync_end();

   boot_timing.print_throughput("Booted", 1);

   //and start kernel

   gallatin::utils::timer insert_timing;

   insert_kernel<ht_type, team_size><<<(team_size*num_inserts-1)/ TEST_BLOCK_SIZE +1, TEST_BLOCK_SIZE>>>(table, num_inserts);

   //insert_kernel_single<ht_type><<<(num_inserts-1)/ TEST_BLOCK_SIZE +1, TEST_BLOCK_SIZE>>>(table, num_inserts);


   insert_timing.sync_end();

   gallatin::utils::timer query_timing;

   query_kernel<ht_type, Val, team_size><<<(team_size*num_inserts-1)/ TEST_BLOCK_SIZE +1, TEST_BLOCK_SIZE>>>(table, num_inserts);

   query_timing.sync_end();



   insert_timing.print_throughput("Inserted", num_inserts);
   query_timing.print_throughput("Queried", num_inserts);


   table->print_chain_stats();

   ht_type::free_on_device(table);

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

   // gallatin_ht_test<uint32_t, uint32_t, 1, 1>(num_segments*16*1024*1024, num_inserts);
   // gallatin_ht_test<uint32_t, uint32_t, 2, 1>(num_segments*16*1024*1024, num_inserts);
   // gallatin_ht_test<uint32_t, uint32_t, 4, 1>(num_segments*16*1024*1024, num_inserts);
   // gallatin_ht_test<uint32_t, uint32_t, 8, 1>(num_segments*16*1024*1024, num_inserts);
   // gallatin_ht_test<uint32_t, uint32_t, 16, 1>(num_segments*16*1024*1024, num_inserts);
   // gallatin_ht_test<uint32_t, uint32_t, 32, 1>(num_segments*16*1024*1024, num_inserts);


   // gallatin_ht_test<uint64_t, uint64_t, 1, 1>(num_segments*16*1024*1024, num_inserts);
   // gallatin_ht_test<uint64_t, uint64_t, 2, 1>(num_segments*16*1024*1024, num_inserts);
   // gallatin_ht_test<uint64_t, uint64_t, 4, 1>(num_segments*16*1024*1024, num_inserts);
   // gallatin_ht_test<uint64_t, uint64_t, 8, 1>(num_segments*16*1024*1024, num_inserts);
   // gallatin_ht_test<uint64_t, uint64_t, 16, 1>(num_segments*16*1024*1024, num_inserts);
   // gallatin_ht_test<uint64_t, uint64_t, 32, 1>(num_segments*16*1024*1024, num_inserts);



   gallatin_ht_test<uint32_t, uint32_t, 1, 1>(num_segments*16*1024*1024, num_inserts);
  
   gallatin_ht_test<uint32_t, uint32_t, 2, 1>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 2, 2>(num_segments*16*1024*1024, num_inserts);

   gallatin_ht_test<uint32_t, uint32_t, 4, 1>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 4, 2>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 4, 4>(num_segments*16*1024*1024, num_inserts);

   gallatin_ht_test<uint32_t, uint32_t, 8, 1>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 8, 2>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 8, 4>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 8, 8>(num_segments*16*1024*1024, num_inserts);
   
   gallatin_ht_test<uint32_t, uint32_t, 16, 1>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 16, 2>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 16, 4>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 16, 8>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 16, 16>(num_segments*16*1024*1024, num_inserts);
   
   gallatin_ht_test<uint32_t, uint32_t, 32, 1>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 32, 2>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 32, 4>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 32, 8>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 32, 16>(num_segments*16*1024*1024, num_inserts);
   gallatin_ht_test<uint32_t, uint32_t, 32, 32>(num_segments*16*1024*1024, num_inserts);




   cudaDeviceSynchronize();

   cudaDeviceReset();
   return 0;

}
