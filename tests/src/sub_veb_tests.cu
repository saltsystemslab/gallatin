/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/sub_veb.cuh>
#include <poggers/allocators/free_list.cuh>

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


using global_ptr = poggers::allocators::header;

using namespace std::chrono;


double elapsed(high_resolution_clock::time_point t1, high_resolution_clock::time_point t2) {
   return (duration_cast<duration<double> >(t2 - t1)).count();
}




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


__global__ void build_tree(sub_veb_tree ** tree_ptr, uint64_t universe, uint64_t seed, void * memory){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0 )return;

   tree_ptr[0] = sub_veb_tree::init(memory, universe, seed);

}


__global__ void get_tree_array_memory(sub_veb_tree * tree, uint64_t * request_bytes){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0 )return;


   request_bytes[0] = tree->get_num_bytes_arrays();

}

__global__ void attach_array(sub_veb_tree * tree, void * array_memory){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0 )return;

   tree->init_arrays(array_memory);


}


__global__ void malloc_free_kernel(sub_veb_tree * dev_tree, uint64_t num_inserts, uint64_t * num_misses){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_inserts) return;

   uint64_t id = dev_tree->malloc();

   if (id != sub_veb_tree::fail()){

      bool dev_removes = dev_tree->insert(id);

      if (!dev_removes){ printf("Fail!\n"); }

   } else {

      atomicAdd((unsigned long long int *) num_misses, 1ULL);

   }

}


__host__ void test_tree_constructor(uint64_t num_slots){


   uint64_t bytes_for_tree = sub_veb_tree::get_size_bytes_noarray(num_slots);

   void * memory;

   cudaMalloc((void **)&memory, bytes_for_tree);

   sub_veb_tree ** tree_ptr_capture;

   sub_veb_tree * dev_tree;

   cudaMallocManaged((void **)&tree_ptr_capture, sizeof(sub_veb_tree **));

   build_tree<<<1,1>>>(tree_ptr_capture, num_slots, 1ULL, memory);

   cudaDeviceSynchronize();

   dev_tree = tree_ptr_capture[0];

   cudaFree(tree_ptr_capture);


   uint64_t * array_memory;

   cudaMallocManaged((void **)&array_memory, sizeof(uint64_t));

   get_tree_array_memory<<<1,1>>>(dev_tree, array_memory);

   cudaDeviceSynchronize();

   void * dev_array_memory;

   printf("Requesting %llu bytes\n", array_memory[0]);

   cudaMalloc((void **)&dev_array_memory, array_memory[0]);

   array_memory[0] = 0;

   cudaDeviceSynchronize();

   attach_array<<<1,1>>>(dev_tree, dev_array_memory);

   cudaDeviceSynchronize();

   malloc_free_kernel<<<(num_slots -1)/512+1, 512>>>(dev_tree, num_slots, array_memory);

   cudaDeviceSynchronize();

   uint64_t misses = array_memory[0];

   printf("Missed %llu/%llu mallocs, %f ratio\n", misses, num_slots, 1.0*misses/num_slots);

   cudaFree(array_memory);

   cudaFree(dev_array_memory);

   cudaFree(memory);

}



__global__ void boot_veb_from_free_list(sub_veb_tree ** tree_ptr, uint64_t universe, global_ptr * free_list){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   uint64_t bytes_needed = sub_veb_tree::device_get_size_bytes_noarray(universe);


   void * memory = free_list->malloc_aligned(bytes_needed, 16, 0);

   if (memory == nullptr){
      
      printf("Failureeee\n");
      return;

   }

   tree_ptr[0] = sub_veb_tree::init(memory, universe, 15);

   sub_veb_tree * dev_tree = tree_ptr[0];


   uint64_t array_bytes = dev_tree->get_num_bytes_arrays();

   void * arr_memory = free_list->malloc_aligned(array_bytes, 16, 0);

   if (arr_memory == nullptr){
      printf("Couldn't allocate bit array\n");
   }

   dev_tree->init_arrays(arr_memory);

   return;


}



__host__ void test_free_list_constructor(uint64_t num_slots){


   high_resolution_clock::time_point kernel_start, kernel_end, boot_start, boot_end;

   global_ptr * free_list = global_ptr::init_heap(2000000000);

   sub_veb_tree ** tree_container;

   cudaMallocManaged((void **)&tree_container, sizeof(sub_veb_tree **));


   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   cudaDeviceSynchronize();

   misses[0] = 0;

   boot_start = high_resolution_clock::now();

   boot_veb_from_free_list<<<1,1>>>(tree_container, num_slots, free_list);

   cudaDeviceSynchronize();

   boot_end = high_resolution_clock::now();

   kernel_start = high_resolution_clock::now();

   malloc_free_kernel<<<(num_slots -1)/512+1, 512>>>(tree_container[0], num_slots, misses);


   cudaDeviceSynchronize();

   kernel_end = high_resolution_clock::now();

   printf("Missed %llu/%llu mallocs, %f ratio\n", misses[0], num_slots, 1.0*misses[0]/num_slots);

   std::cout << "Booted in " << elapsed(boot_start, boot_end) << ", Cycle took " << elapsed(kernel_start, kernel_end) << std::endl;
   cudaFree(misses);

   cudaFree(tree_container);

   global_ptr::free_heap(free_list);

}




//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   // test_tree_constructor(1000);
   // test_tree_constructor(1000000);
   // test_tree_constructor(4000000);
   // test_tree_constructor(1000000000);

   test_free_list_constructor(10000);
   test_free_list_constructor(1000000);
   test_free_list_constructor(10000000);
   test_free_list_constructor(100000000);
 
   cudaDeviceReset();
   return 0;

}
