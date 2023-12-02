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


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace gallatin::data_structs;
using namespace gallatin::allocators;


//enqueue test kernel loads nitems into the queue, with every item unique based on TID
//then dequeue tests correctness by mapping to bitarry.
template <typename queue> 
__global__ void enqueue_test_kernel(queue * dev_queue, uint64_t nitems){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems) return;

   dev_queue->enqueue(tid+1);
   
}


template <typename queue>
__global__ void dequeue_test_kernel(queue * dev_queue, uint64_t * bitarray, uint64_t nitems){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems) return;

   uint64_t ext_tid = 0;

   if (!dev_queue->dequeue(ext_tid)){
      printf("Thread %llu\n failed to dequeue item...\n", tid);
      return;
   }


   if (ext_tid > nitems){
      printf("Pulled invalid item\n");
   }

   if (ext_tid == 0){

      printf("Failed to set dequeue_val\n");
   }

   ext_tid = ext_tid-1;

   //all items dequeued, let's check correctness

   uint64_t high = ext_tid / 64;

   uint64_t low = ext_tid % 64;

   auto bitmask = SET_BIT_MASK(low);

   uint64_t bits = atomicOr((unsigned long long int *) &bitarray[high], (unsigned long long int) bitmask);

   if (bits & bitmask){
      printf("Double dequeue bug in %llu: block %llu alloc %llu\n", ext_tid, ext_tid/4096, ext_tid % 4096);
   }

}


__host__ void queue_test(uint64_t n_threads){

   using queue_type = ring_queue<uint64_t , 0ULL>;


   init_global_allocator(20ULL*1024*1024*1024, 11ULL);

   //boot with 20 Giga
   queue_type * dev_queue = queue_type::generate_on_device(n_threads);


   uint64_t num_bytes_bitarray = sizeof(uint64_t)*((n_threads -1)/64+1);

   uint64_t * bits;

   cudaMalloc((void **)&bits, num_bytes_bitarray);

   cudaMemset(bits, 0, num_bytes_bitarray);

   cudaDeviceSynchronize();

   printf("Starting queue test\n");

   gallatin::utils::timer enqueue_timing;

   enqueue_test_kernel<queue_type><<<(n_threads-1)/256 +1, 256>>>(dev_queue, n_threads);

   enqueue_timing.sync_end();

   enqueue_timing.print_throughput("Enqueued", n_threads);

   gallatin::utils::timer dequeue_timing;

   dequeue_test_kernel<queue_type><<<(n_threads-1)/256 +1, 256>>>(dev_queue, bits, n_threads);

   dequeue_timing.sync_end();

   

   dequeue_timing.print_throughput("Dequeued", n_threads);

   cudaDeviceSynchronize();

   queue_type::free_on_device(dev_queue);

   free_global_allocator();


}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   uint64_t num_threads;

   if (argc < 2){
      num_threads = 100;
   } else {
      num_threads = std::stoull(argv[1]);
   }


   queue_test(num_threads);

   cudaDeviceReset();
   return 0;

}
