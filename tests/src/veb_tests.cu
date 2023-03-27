/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/veb.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace poggers::allocators;


// __global__ void test_kernel(veb_tree * tree, uint64_t num_removes, int num_iterations){


//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    if (tid >= num_removes)return;


//       //printf("Tid %lu\n", tid);


//    for (int i=0; i< num_iterations; i++){


//       if (!tree->remove(tid)){
//          printf("BUG\n");
//       }

//       tree->insert(tid);

//    }


__global__ void view_bits(layer * dev_layer){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   printf("First set of bits: %lu\n", dev_layer->bits[0]);



}


__global__ void view_tree_bits(veb_tree * dev_tree){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   for (int i = 0; i< dev_tree->num_layers; i++){

      printf("First bits of layer %d: %lu\n", i, dev_tree->layers[i]->bits[0]);

   }

   printf("End of tree\n");



}


__global__ void remove_insert_kernel(veb_tree * dev_tree, uint64_t num_removes, int num_rounds, uint64_t * misses){


   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid >= num_removes) return;


   for (int i = 0; i < num_rounds; i++){

      uint64_t remove = dev_tree->malloc();

      if (remove != veb_tree::fail()){

          bool dev_removes = dev_tree->insert(remove);

         if (!dev_removes){ printf("Fail!\n"); }

      } else {

         atomicAdd((unsigned long long int *) misses, 1ULL);

      }





   }


}

__global__ void multi_run_remove_insert_kernel(veb_tree * dev_tree, uint64_t num_removes, int num_rounds, uint64_t * misses){


   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid >= num_removes) return;

   uint64_t addresses[10];

   for (int i = 0; i < num_rounds; i++){

      uint64_t remove = dev_tree->malloc();

      addresses[i] = remove;

      


   }

   for (int i = 0; i < num_rounds; i++){

      if (addresses[i] == veb_tree::fail()){
         atomicAdd((unsigned long long int *)misses, 1ULL);
      } else {

         bool remove = dev_tree->insert(addresses[i]);

         if (!remove) printf("Failed to re-insert %lu\n", addresses[i]);

      }
   }


}

__global__ void check_insert_kernel(veb_tree * dev_tree, uint64_t * items, uint64_t * misses, uint64_t num_items){


   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid >= num_items) return;


   uint64_t offset_at_lowest = dev_tree->malloc();

   if (offset_at_lowest == veb_tree::fail()){

      atomicAdd((unsigned long long int *)misses, 1ULL);

   } else {

      items[tid] = offset_at_lowest;

   }


}


__global__ void free_insert_kernel(veb_tree * dev_tree, uint64_t * items, uint64_t num_items){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_items) return;

   dev_tree->insert(items[tid]);

}


__global__ void assert_unique(uint64_t * items, uint64_t num_items){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_items) return;

   uint64_t item = items[tid];

   //0's are misses
   if (item == 0) return; 

   for (uint64_t i = 0; i < tid; i++){

      if (i == tid) continue;

      if (item == items[i]){
         printf("Conflict betwen %lu and %lu: %lu\n", tid, i, item);
      }

   }

}


__global__ void remove_insert_kernel_single_thread(veb_tree * dev_tree, uint64_t * items, uint64_t num_removes, int num_rounds){


   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid != 0) return;


   for (int i = 0; i < num_rounds; i++){


      for (uint64_t item_counter = 0; item_counter < num_removes; item_counter++){

         items[item_counter] = dev_tree->malloc();

         assert(items[item_counter] != veb_tree::fail());


      }



      for (uint64_t item_counter = 0; item_counter < num_removes; item_counter++){

         bool dev_removes = dev_tree->insert(items[item_counter]);

         assert(dev_removes);

      }


   }


}

// }

// __global__ void view_kernel(veb_tree * tree){

//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    if (tid != 0)return;



// }



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   // if (!test_one_thread()){
   //    printf("Test one thread: [FAIL]\n");
   // } else {
   //    printf("Test one thread: [PASS]\n");
   // }


   // uint64_t num_removes = 64;

   // veb_tree * test_allocator = veb_tree::generate_on_device(num_removes);


   // cudaDeviceSynchronize();

   // view_kernel<<<1,1>>>(test_allocator);


   // cudaDeviceSynchronize();

   // auto insert_start = std::chrono::high_resolution_clock::now();

   // test_kernel<<<(num_removes-1)/512+1,512>>>(test_allocator, num_removes,1);


   // cudaDeviceSynchronize();

   // auto insert_end= std::chrono::high_resolution_clock::now();


   // printf("Space usage: %lu\n", test_allocator->space_in_bytes());

   // std::chrono::duration<double> elapsed_seconds = insert_end - insert_start;


   // std::cout << "Inserted " <<  num_removes << " in " << elapsed_seconds.count() << "seconds, throughput: " << std::fixed << 1.0*num_removes/elapsed_seconds.count() << std::endl;
  

   //veb_tree::free_on_device(test_allocator);

   // for (int i = 0; i < 32; i++){


   //    layer * try_layer = layer::generate_on_device((1ULL << i));

   //    cudaDeviceSynchronize();

   //    view_bits<<<1,1>>>(try_layer);

   //    cudaDeviceSynchronize();

   //    layer::free_on_device(try_layer);

   //    cudaDeviceSynchronize();



   // }


   // for (int i = 28; i < 29; i++){

   //    uint64_t num_items = (1ULL << i);

   //    printf("%d shifted is %lu\n", i, num_items);

   //    uint64_t * items;

   //    CHECK_CUDA_ERROR(cudaMalloc((void **)&items, num_items*sizeof(uint64_t)));

   //    cudaMemset(items, 0, num_items*sizeof(uint64_t));


   //    uint64_t * misses;

   //    cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   //    cudaDeviceSynchronize();

   //    misses[0] = 0;

   //    cudaDeviceSynchronize();

   //    veb_tree * tree = veb_tree::generate_on_device(num_items, i);

   //    cudaDeviceSynchronize();

   //    check_insert_kernel<<<(num_items-1)/512+1,512>>>(tree, items, misses, num_items);

   //    cudaDeviceSynchronize();

   //    std::cout << "Missed " << misses[0] << "/" << (uint64_t) (num_items) << " items, fraction: " << 1.0*misses[0]/(num_items) << "\n";


   //    assert_unique<<<(num_items-1)/512+1,512>>>(items, num_items);

   //    cudaDeviceSynchronize();

   //    free_insert_kernel<<<(num_items-1)/512+1,512>>>(tree, items, num_items);

   //    cudaDeviceSynchronize();


   //    cudaFree(items);

   //    cudaFree(misses);

   //    veb_tree::free_on_device(tree);

   //    cudaDeviceSynchronize();



   // }

   for (int i = 15; i < 32; i++){


      uint64_t num_items = (1ULL << i);


      uint64_t * items;

      //cudaMalloc((void **)&items, num_items*sizeof(uint64_t));

      uint64_t * misses;

      cudaMallocManaged((void **)&misses, sizeof(uint64_t));

      cudaDeviceSynchronize();

      misses[0] = 0;

      cudaDeviceSynchronize();

      int num_rounds = 10;

      printf("Starting tree %d with %lu items\n", i, num_items);

      veb_tree * tree = veb_tree::generate_on_device(num_items, 15);



      cudaDeviceSynchronize();


      auto insert_start = std::chrono::high_resolution_clock::now();

      //peek
      //view_tree_bits<<<1,1>>>(tree);

      //remove_insert_kernel_single_thread<<<1,1>>>(tree, items, num_items, num_rounds);

      remove_insert_kernel<<<(num_items-1)/512+1,512>>>(tree, num_items,num_rounds, misses);

      //view_tree_bits<<<1,1>>>(tree);

      cudaDeviceSynchronize();


      //multi_run_remove_insert_kernel<<<(num_items/10-1)/512+1, 512>>>(tree, num_items/10, 10, misses);


      cudaDeviceSynchronize();

      auto insert_end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> elapsed_seconds = insert_end - insert_start;


      std::cout << "Inserted " <<  num_items*num_rounds << " in " << elapsed_seconds.count() << " seconds, throughput: " << std::fixed << 1.0*(num_items*num_rounds)/elapsed_seconds.count() << std::endl;
  
      std::cout << "Missed " << misses[0] << "/" << (uint64_t) (num_items*num_rounds) << " items, fraction: " << 1.0*misses[0]/(num_items*num_rounds) << "\n";

      //cudaFree(items);

      cudaFree(misses);

      veb_tree::free_on_device(tree);


   }

   // cudaDeviceSynchronize();

   // num_removes = (1ULL << 32);

   // veb_tree * test_allocator_2 = veb_tree::generate_on_device(num_removes);

   // cudaDeviceSynchronize();

   // view_kernel<<<1,1>>>(test_allocator_2);

   // cudaDeviceSynchronize();


   // insert_start = std::chrono::high_resolution_clock::now();

   // test_kernel<<<(num_removes-1)/512+1,512>>>(test_allocator_2, num_removes,1);

   // cudaDeviceSynchronize();

   // insert_end = std::chrono::high_resolution_clock::now();

   // printf("Space usage: %lu\n", test_allocator_2->space_in_bytes());

   // elapsed_seconds = insert_end - insert_start;


   // std::cout << "Inserted " <<  num_removes << " in " << elapsed_seconds.count() << "seconds, throughput: " << std::fixed << 1.0*num_removes/elapsed_seconds.count() << std::endl;
   // //printf("Aggregate inserts: %lu in %lu: %f\n", num_removes, time, 1.0*num_removes/time);

   // cudaDeviceSynchronize();

   // view_kernel<<<1,1>>>(test_allocator_2);

   // cudaDeviceSynchronize();

   // veb_tree::free_on_device(test_allocator_2);

   // cudaDeviceSynchronize();


   // num_removes = (1ULL << 34)/128;


   // veb_tree * test_allocator_3 = veb_tree::generate_on_device(num_removes);

   // cudaDeviceSynchronize();

   // insert_start = std::chrono::high_resolution_clock::now();

   // test_kernel<<<(num_removes-1)/512+1,512>>>(test_allocator_3, num_removes,1);

   // cudaDeviceSynchronize();

   // insert_end= std::chrono::high_resolution_clock::now();

   // printf("Space usage: %lu\n", test_allocator_3->space_in_bytes());

   // elapsed_seconds = insert_end - insert_start;


   // std::cout << "Inserted " <<  num_removes << " in " << elapsed_seconds.count() << "seconds, throughput: " << std::fixed << 1.0*num_removes/elapsed_seconds.count() << std::endl;
   
   // veb_tree::free_on_device(test_allocator_3);

   // cudaDeviceSynchronize();


 
   cudaDeviceReset();
   return 0;

}
