/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



//This tests the fixed vector type, which allows for vector operations within
// a set range of sizes.
//Fixing he size of the vector alows for faster operations
// due to the stability of the vector components.

#include <gallatin/data_structs/log.cuh>
#include <gallatin/allocators/timer.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace gallatin::data_structs;
using namespace gallatin::allocators;



template <typename log_type>
__global__ void simple_log_test(log_type * log, uint64_t n_items){


   //using vector_type = gallatin::data_structs::fixed_vector<uint64_t, min, max>;

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_items) return;

   double ratio = tid;
   ratio = ratio/n_items;

   log->add_log("Starting tid ", tid, "/", n_items, ": ", ratio);

   //log->add_log("Log added by tid ", tid);

   
}

template <typename log_type>
__global__ void view_log(log_type * log){

   uint64_t tid = gallatin::utils::get_tid();


}

template <bool on_host>
__host__ void init_and_test_log_simple(uint64_t max_items){


   using log_type = gallatin_log<on_host>;


   log_type * dev_log = log_type::generate_on_device();

   cudaDeviceSynchronize();

   gallatin::utils::timer log_timing;

   simple_log_test<log_type><<<(max_items-1)/256+1,256>>>(dev_log, max_items);

   log_timing.sync_end();


   view_log<log_type><<<1,1>>>(dev_log);

   cudaDeviceSynchronize();

   log_timing.print_throughput("log enqueued", max_items);


   auto log_strings = log_type::export_log(dev_log);

   log_type::free_on_device(dev_log);
   cudaDeviceSynchronize();

   printf("Printing vector\n");

   for (auto str : log_strings){
      std::cout << str << std::endl;
   }

}


template <bool on_host>
__host__ void init_and_test_log_file(uint64_t max_items, std::string filename){


   using log_type = gallatin_log<on_host>;


   log_type * dev_log = log_type::generate_on_device();

   cudaDeviceSynchronize();

   gallatin::utils::timer log_timing;

   simple_log_test<log_type><<<(max_items-1)/256+1,256>>>(dev_log, max_items);

   log_timing.sync_end();


   // view_log<log_type><<<1,1>>>(dev_log);

   cudaDeviceSynchronize();


   
   log_timing.print_throughput("log enqueued", max_items);


   //auto log_strings = log_type::export_log(dev_log);
   gallatin::utils::timer write_timer;


   dev_log->dump_to_file(filename);

   write_timer.sync_end();

   write_timer.print_throughput("Dumped", max_items);

   log_type::free_on_device(dev_log);
   cudaDeviceSynchronize();


}


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   //one_boot_betta_test_all_sizes<16ULL*1024*1024, 16ULL, 16ULL>(num_segments*16*1024*1024);  


   //beta_test_allocs_correctness<16ULL*1024*1024, 16ULL, 4096ULL>(num_segments*16*1024*1024, num_rounds, size);

   //init_global_allocator(30ULL*1024*1024*1024, 42);
   init_global_allocator_combined(8ULL*1024*1024*1024, 40ULL*1024*1024*1024, 42);

   //init_and_test_vector<16ULL, 16384ULL>();

   init_and_test_log_simple<false>(10);

   init_and_test_log_simple<true>(10);


   //init_and_test_log_file<false>(10000000, "dev_sample_log_output.txt");

   init_and_test_log_file<true>(10000000, "host_sample_log_output.txt");

   free_global_allocator_combined();

   //beta_full_churn<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments, num_rounds);


   //beta_pointer_churn<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments, num_rounds);


   //beta_churn_no_free<16ULL*1024*1024, 16ULL, 4096ULL>(1600ULL*16*1024*1024,  num_segments);



   cudaDeviceReset();
   return 0;

}
