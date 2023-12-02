/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gallatin/data_structs/ring_queue.cuh>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/allocators/timer.cuh>
#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/data_structs/single_vector.cuh>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

#include <vector>

using namespace gallatin::data_structs;
using namespace gallatin::allocators;


//enqueue test kernel loads nitems into the queue, with every item unique based on TID
//then dequeue tests correctness by mapping to bitarry.
__global__ void vector_kernel(uint64_t n_threads, uint64_t num_enqueues){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_threads) return;

   svector<uint64_t> my_vector(10);

   for (uint64_t i = 0; i < num_enqueues; i++){
      my_vector.push_back(i+tid);
   }

   for (uint64_t i = 0; i < 10; i++){
      my_vector.remove(0);
   }

   for (uint64_t i = 10; i < num_enqueues; i++){

      if (my_vector[i-10] != tid+i){
         printf("Tid %llu Failed to write to index %llu\n", tid, i);
      }

   }

   my_vector.free_vector();
   
}

__global__ void realloc_vector_kernel(uint64_t n_threads, uint64_t num_enqueues){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_threads) return;

   svector<uint64_t> my_vector(10);

   my_vector.realloc(num_enqueues);

   for (uint64_t i = 0; i < num_enqueues; i++){
      my_vector.push_back(i+tid);
   }

   my_vector.realloc(num_enqueues-10);

   for (uint64_t i = 0; i < num_enqueues-10; i++){

      if (my_vector[i] != tid+i){
         printf("Tid %llu Failed to write to index %llu\n", tid, i);
      }

   }

   my_vector.free_vector();
   
}


__global__ void read_copied_vector(uint64_t num_enqueues, svector<uint64_t> * alt_vector){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid != 0) return;

   for (uint64_t i = 0; i < num_enqueues; i++){

      if (alt_vector[0][i] != i){

         printf("Copied vector index %llu failed to copy\n", i);
      }


   }


   alt_vector->free_vector();


}


__host__ void vector_test(uint64_t n_threads, uint64_t num_enqueues){

   init_global_allocator(20ULL*1024*1024*1024, 11ULL);

   //boot with 20 Giga
   printf("Starting vector_test\n");

   gallatin::utils::timer enqueue_timing;

   vector_kernel<<<(n_threads-1)/256 +1, 256>>>(n_threads, num_enqueues);

   enqueue_timing.sync_end();

   enqueue_timing.print_throughput("Enqueued", n_threads*num_enqueues);

   gallatin::utils::timer realloc_timing;

   realloc_vector_kernel<<<(n_threads-1)/256 +1, 256>>>(n_threads, num_enqueues);

   realloc_timing.sync_end();

   realloc_timing.print_throughput("Realloc Enqueued", n_threads*num_enqueues);



   std::vector<uint64_t> test_vector;

   for (uint64_t i = 0; i < num_enqueues; i++){
      test_vector.push_back(i);
   }

   auto dev_vector = svector<uint64_t>::copy_to_device(test_vector);

   read_copied_vector<<<1,1>>>(num_enqueues, dev_vector);

   cudaDeviceSynchronize();

   auto dev_vector2 = svector<uint64_t>::copy_to_device(test_vector.data(), test_vector.size());

   read_copied_vector<<<1,1>>>(num_enqueues, dev_vector2);

   cudaDeviceSynchronize();

   print_global_stats();

   free_global_allocator();


}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   uint64_t num_threads;

   uint64_t num_enqueues;

   if (argc < 2){
      num_threads = 100;
   } else {
      num_threads = std::stoull(argv[1]);
   }

   if (argc < 3){
      num_enqueues = 1000;
   } else {
      num_enqueues = std::stoull(argv[2]);
   }


   vector_test(num_threads, num_enqueues);

   cudaDeviceReset();
   return 0;

}
