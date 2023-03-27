/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */


#define DEBUG_ASSERTS 0

#define DEBUG_PRINTS 0

#define SHOW_PROGRESS 0

#define COUNTING_CYCLES 1

#include <poggers/allocators/cms.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

//#define stack_bytes 262144




//#define stack_bytes 4194304

#define MEGABYTE 1024*1024

#define GIGABYTE 1024*MEGABYTE

#define stack_bytes 4*MEGABYTE



using shibboleth = poggers::allocators::shibboleth<stack_bytes, 150, 8>;


struct queue_node {

   uint64_t item;
   queue_node * ptr_to_next;

};

struct queue {

   queue_node * head;
   queue_node * tail;

   __device__ queue_node * load_tail_atomic(){

     return (queue_node *) atomicCAS((unsigned long long int *)&tail, 0ULL, 0ULL);

   }

   __device__ queue_node * load_head_atomic(){

      return (queue_node *) atomicCAS((unsigned long long int *)&head, 0ULL, 0ULL);

   }


   __device__ void enqueue(queue_node * next){

      while (true){

         queue_node * my_tail = load_tail_atomic();

         if (my_tail == nullptr){

            if (((queue_node * ) atomicCAS((unsigned long long int *)&tail, (unsigned long long int) my_tail, (unsigned long long int) next) )== nullptr){
               return;
            }


         } else if (((queue_node * ) atomicCAS((unsigned long long int *)my_tail->ptr_to_next, (unsigned long long int) nullptr, (unsigned long long int) next)) == nullptr){

            atomicCAS((unsigned long long int *)&tail, (unsigned long long int) my_tail, (unsigned long long int) next);

            return;
         }

      }



   }

   __device__ queue_node * dequeue(){

      while (true){

         queue_node * my_head = load_head_atomic();

         if (my_head == nullptr) return nullptr;

         if (((queue_node * ) atomicCAS((unsigned long long int *)&head, (unsigned long long int) my_head, (unsigned long long int) my_head->ptr_to_next)) == my_head){
            return my_head;
         }

      }


   }


};


__global__ void cms_enqueue_benchmark(queue * group_queue, shibboleth * cms){

   uint64_t items_per_thread = 1;


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   #if SHOW_PROGRESS
   if (tid % 100000 == 0){
      printf("%llu\n", tid);
   }
   #endif


   for (uint64_t i = 0; i < items_per_thread; i++){

      queue_node * my_node = (queue_node *) cms->cms_malloc(sizeof(queue_node));

      if (my_node == nullptr){
         printf("Missed %llu\n", tid);
      } else {
         my_node->item = tid;
         my_node->ptr_to_next = nullptr;

      }



      //group_queue->enqueue(my_node);


   }




}

__global__ void cuda_enqueue_benchmark(){


   uint64_t items_per_thread = 1;

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   #if SHOW_PROGRESS
   if (tid % 100000 == 0){
      printf("%llu\n", tid);
   }
   #endif

   for (uint64_t i = 0; i < items_per_thread; i++){

      queue_node * my_node = (queue_node *) malloc(sizeof(queue_node));

      my_node->item = tid;
      my_node->ptr_to_next = nullptr;

   }
}

__global__ void drain_queue(queue * group_queue, shibboleth * cms){

   queue_node * current = group_queue->dequeue();

   while (current != nullptr){

      cms->cms_free(current);
      current = group_queue->dequeue();

   }

}

int main(int argc, char** argv) {


   //allocate 
   //const uint64_t meg = 1024*1024;
   const uint64_t bytes_in_use = 4ULL*GIGABYTE;

   const uint64_t block_size = 512;

   //should be plenty of space for 100 million
   //roughly 1.5 gigs of data to account for stack fragmentation
   //stack grouping will fix this but this benchmark is aboutr performance.
   //uint64_t cuda_items_to_enqueue = 20000000;

   uint64_t items_to_enqueue = 100000000;



   //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 3ULL*1024*1024*1024);

   printf("Starting cuda test\n");

   cudaDeviceSynchronize();

   auto cuda_start = std::chrono::high_resolution_clock::now();

   //cuda_enqueue_benchmark<<<(items_to_enqueue - 1)/block_size +1, block_size>>>();

   cudaDeviceSynchronize();

   auto cuda_end = std::chrono::high_resolution_clock::now();






   shibboleth * allocator = shibboleth::init(bytes_in_use);

   cudaDeviceSynchronize();

   printf("Report before\n");
   allocator->host_report();

   cudaDeviceSynchronize();


   queue * my_queue = (queue *) allocator->cms_host_malloc(sizeof(queue));

   cudaMemset(my_queue, 0, sizeof(queue));

   printf("Starting CMS test\n");

   cudaDeviceSynchronize();

   auto cms_start = std::chrono::high_resolution_clock::now();

   cms_enqueue_benchmark<<<(items_to_enqueue - 1)/block_size +1, block_size>>>(my_queue, allocator);

   cudaDeviceSynchronize();

   auto cms_end = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> cms_diff = cms_end-cms_start;

   std::chrono::duration<double> cuda_diff = cuda_end-cuda_start;

   

   //cudaDeviceSynchronize();
   std::cout << "cms Malloced " << items_to_enqueue << " in " << cms_diff.count() << " seconds\n";
   std::cout << "cuda Malloced " << items_to_enqueue << " in " << cuda_diff.count() << " seconds\n";

   //cms_single_threaded<<<1, 100>>>(allocator);

   cudaDeviceSynchronize();

   allocator->host_report();

   cudaDeviceSynchronize();


   shibboleth::free_cms_allocator(allocator);

   cudaDeviceSynchronize();




   return 0;




}
