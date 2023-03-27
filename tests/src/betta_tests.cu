/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/ext_veb_nosize.cuh>
#include <poggers/allocators/alloc_memory_table.cuh>
#include <poggers/allocators/betta.cuh>


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

template <uint64_t mem_segment_size, uint64_t num_bits>
__host__ void boot_ext_tree(){

   using tree_type = extending_veb_allocator_nosize<mem_segment_size, 5>;

   tree_type * tree_to_boot = tree_type::generate_on_device(num_bits, 1342);

   cudaDeviceSynchronize();

   tree_type::free_on_device(tree_to_boot);

   cudaDeviceSynchronize();

}


template <uint64_t mem_segment_size>
__host__ void boot_alloc_table(){


   using table_type = alloc_table<mem_segment_size>;

   table_type * table = table_type::generate_on_device();

   cudaDeviceSynchronize();

   table_type::free_on_device(table);

}
// }

// __global__ void view_kernel(veb_tree * tree){

//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    if (tid != 0)return;



// }

template <typename betta_type>
__global__ void register_all_segments(betta_type * betta, uint64_t num_segments){

   uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

   if (tid >= num_segments) return;

   betta->gather_new_segment(0);

}


template <typename betta_type>
__global__ void malloc_all_blocks(betta_type * betta, uint64_t num_blocks){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid > 0) return;

   uint64_t misses = 0;


   for (uint64_t i = 0; i < num_blocks; i++){

      int counter = 0;

      offset_alloc_bitarr * block = nullptr;

      while (counter < 10){

          bool needs_allocations = false;
          block = betta->sub_trees[0]->malloc_block(needs_allocations);

          if (block != nullptr) break;

          counter+=1;

      }


      if (block == nullptr){
         printf("Failed to alloc %llu/%llu\n", i, num_blocks);
         misses+=1;
      }

   }

   printf("Total alloc misses: %llu\n", misses);


}


template <typename betta_type>
__global__ void peek(betta_type * betta){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid !=0 ) return;
}


// template <typename betta_type>
// __global__ void malloc_all_bits( )

// template <typename betta_type>
// __global__ void malloc_all_segments(betta_type * betta, uint64_t num_segments){

//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    if (tid >= num_segments) return;

//    betta

// }


template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void boot_betta(uint64_t num_bytes){

   using betta_type = poggers::allocators::betta_allocator<mem_segment_size, smallest, largest>;

   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

   cudaDeviceSynchronize();

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   register_all_segments<betta_type><<<(num_segments-1)/512+1,512>>>(allocator, num_segments);

   printf("Ext sees %llu segments\n", num_segments);
   cudaDeviceSynchronize();

   poggers::utils::print_mem_in_use();


   cudaDeviceSynchronize();


   malloc_all_blocks<betta_type><<<(num_segments*128-1)/512+1,512>>>(allocator, num_segments*128);

   cudaDeviceSynchronize();

   peek<betta_type><<<1,1>>>(allocator);

   cudaDeviceSynchronize();

   betta_type::free_on_device(allocator);

}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   // boot_ext_tree<8ULL*1024*1024, 16ULL>();
 
   // boot_ext_tree<8ULL*1024*1024, 4096ULL>();


   // boot_alloc_table<8ULL*1024*1024>();


   boot_betta<8ULL*1024*1024, 16ULL, 16ULL>(10ULL*1000*1000*1000);

   cudaDeviceReset();
   return 0;

}
