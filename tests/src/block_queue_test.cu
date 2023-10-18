/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/data_structs/block_queue.cuh>
#include <gallatin/allocators/timer.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace gallatin::allocators;

using namespace gallatin::data_structs;


//enqueue test kernel loads nitems into the queue, with every item unique based on TID
//then dequeue tests correctness by mapping to bitarry.
template <typename queue> 
__global__ void enqueue_test_kernel(queue * dev_queue, uint64_t nitems){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= nitems) return;

   dev_queue->enqueue(tid);

   //printf("Done with %llu\n", tid);
   
}


// template <typename queue>
// __global__ void dequeue_test_kernel(queue * dev_queue, uint64_t * bitarray, uint64_t nitems){

//    uint64_t tid = gallatin::utils::get_tid();

//    if (tid >= nitems) return;

//    uint64_t ext_tid;

//    if (!dev_queue->dequeue(ext_tid)){
//       printf("Thread %llu\n failed to dequeue item...\n", tid);
//       return;
//    }

//    //all items dequeued, let's check correctness

//    uint64_t high = ext_tid / 64;

//    uint64_t low = ext_tid % 64;

//    auto bitmask = SET_BIT_MASK(low);

//    uint64_t bits = atomicOr((unsigned long long int *) &bitarray[high], (unsigned long long int) bitmask);

//    if (bits & bitmask){
//       printf("Double dequeue bug in %llu: block %llu alloc %llu\n", ext_tid, ext_tid/4096, ext_tid % 4096);
//    }

// }


template <int nslots_per_block>
__host__ void queue_test(uint64_t n_threads){

   
   using queue_type = block_queue<uint64_t, nslots_per_block>;

   //boot with 20 Gigs
   //gallatin_allocator * alloc = gallatin_allocator::generate_on_device(20ULL*1024*1024*1024, 111);

   init_global_allocator(10ULL*1024*1024*1024, 1245632ULL);

   queue_type * dev_queue = queue_type::generate_on_device();


   uint64_t num_bytes_bitarray = sizeof(uint64_t)*((n_threads -1)/64+1);

   uint64_t * bits;

   cudaMalloc((void **)&bits, num_bytes_bitarray);

   cudaMemset(bits, 0, num_bytes_bitarray);

   cudaDeviceSynchronize();

   printf("Starting queue test with size %d\n", nslots_per_block);

   gallatin::utils::timer enqueue_timing;

   enqueue_test_kernel<queue_type><<<(n_threads-1)/256 +1, 256>>>(dev_queue, n_threads);

   enqueue_timing.sync_end();

   enqueue_timing.print_throughput("Enqueued", n_threads);

   // beta::utils::timer dequeue_timing;

   // dequeue_test_kernel<queue_type><<<(n_threads-1)/256 +1, 256>>>(dev_queue, bits, n_threads);

   // dequeue_timing.sync_end();

   free_global_allocator();
   

   //dequeue_timing.print_throughput("Dequeued", n_threads);


}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   uint64_t num_threads;

   if (argc < 2){
      num_threads = 100;
   } else {
      num_threads = std::stoull(argv[1]);
   }


   // queue_test<1>(num_threads);
   // queue_test<2>(num_threads);
   // queue_test<4>(num_threads);
   // queue_test<8>(num_threads);
   // queue_test<16>(num_threads);
   // queue_test<32>(num_threads);
   // queue_test<64>(num_threads);
   // queue_test<128>(num_threads);
   queue_test<256>(num_threads);


   cudaDeviceReset();
   return 0;

}
