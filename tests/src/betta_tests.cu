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


template <uint64_t mem_segment_size, uint64_t smallest>
__host__ void boot_alloc_table(){


   using table_type = alloc_table<mem_segment_size, smallest>;

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
__global__ void malloc_all_blocks_single_thread(betta_type * betta, uint64_t num_segments, uint64_t blocks_per_segment){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid > 0) return; 

   uint64_t misses = 0;


   for (uint64_t i =0; i < num_segments; i++){

      printf("%llu/%llu\n", i, num_segments);

      for (uint64_t j = 0; j < blocks_per_segment; j++){

         offset_alloc_bitarr * new_block = betta->table->get_block(i);

         if (new_block == nullptr){

            atomicAdd((unsigned long long int *)misses, 1);

         }

      }
   }

   printf("Total alloc misses: %llu/%llu\n", misses, num_segments*blocks_per_segment);


}

template <typename betta_type>
__global__ void malloc_all_blocks(betta_type * betta, uint64_t num_segments, uint64_t blocks_per_segment){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return; 

   uint64_t segment_id = tid/blocks_per_segment;

   bool last_block;

   offset_alloc_bitarr * new_block = betta->table->get_block(segment_id, last_block);

   if (new_block == nullptr){

      printf("Missed block %llu in section %llu\n", tid, segment_id);

   }



}

//pull all blocks using betta
template <typename betta_type>
__global__ void malloc_all_blocks_betta(betta_type * betta, uint64_t num_segments, uint64_t blocks_per_segment){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return;

   offset_alloc_bitarr * block = betta->request_new_block_from_tree(0);

   if (block == nullptr){
      printf("Failed to get block!\n");
   }

}


template <typename betta_type>
__global__ void malloc_and_save_blocks(betta_type * betta, offset_alloc_bitarr ** blocks, uint64_t num_segments, uint64_t blocks_per_segment){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return;

   offset_alloc_bitarr * block = betta->request_new_block_from_tree(0);

   if (block == nullptr){
      printf("Alloc failure\n");
   }

   blocks[tid] = block;

}


template <typename betta_type>
__global__ void malloc_and_save_blocks_tree(betta_type * betta, offset_alloc_bitarr ** blocks, uint64_t num_segments, uint64_t blocks_per_segment, int tree_id){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return;

   offset_alloc_bitarr * block = betta->request_new_block_from_tree(tree_id);

   if (block == nullptr){
      printf("Alloc failure\n");

      block = betta->request_new_block_from_tree(tree_id);
   }

   blocks[tid] = block;

}


template <typename betta_type>
__global__ void betta_free_all_blocks(betta_type * betta, offset_alloc_bitarr ** blocks, uint64_t num_segments, uint64_t blocks_per_segment){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return;

   offset_alloc_bitarr * block = blocks[tid];

   if (block == nullptr) return;

   betta->free_block(block);

}



template <typename betta_type>
__global__ void malloc_all_blocks_betta_single_thread(betta_type * betta, uint64_t num_segments, uint64_t blocks_per_segment){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   uint64_t misses = 0;

   for (uint64_t i = 0; i < num_segments*blocks_per_segment; i++){


      offset_alloc_bitarr * block = betta->request_new_block_from_tree(0);

      if (block == nullptr){
         //printf("Failed to get block!\n");
         misses+=1;
      }

   }

   printf("Missed %llu/%llu\n", misses, num_segments*blocks_per_segment);

}


template <typename betta_type>
__global__ void peek(betta_type * betta){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid !=0 ) return;
}


template <typename betta_type>
__global__ void peek_blocks(betta_type * betta, offset_alloc_bitarr ** blocks){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid !=0 ) return;
}


__global__ void assert_unique_blocks(offset_alloc_bitarr ** blocks, uint64_t num_segments, uint64_t blocks_per_segment){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return;

   uint64_t my_block = (uint64_t) blocks[tid];


   for (uint64_t i=tid+1; i < num_segments*blocks_per_segment; i++){

      uint64_t ext_block = (uint64_t) blocks[i];

      if (ext_block == my_block){
         printf("Collision on %llu and %llu: %llx\n", tid, i, ext_block);
      }

   }


}


template <typename betta_type>
__global__ void alloc_random_blocks(betta_type * betta){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   offset_alloc_bitarr * my_blocks[10];

   uint64_t num_trees = betta_type::get_num_trees();


   for (int i = 0; i < 10; i++){

      int tree = poggers::hashers::MurmurHash64A (&tid, sizeof(uint64_t), i) % num_trees;

      my_blocks[i] = betta->request_new_block_from_tree(tree);

   }


   for (int i = 0; i < 10; i++){

      if (my_blocks[i] == nullptr){
         printf("Failed to alloc\n");
      } else {
         betta->free_block(my_blocks[i]);
      }

      

   }



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

   //malloc_all_blocks_single_thread<betta_type><<<1,1>>>(allocator, num_segments, 256);
   //malloc_all_blocks<betta_type><<<(num_segments*128-1)/512+1,512>>>(allocator, num_segments*128);

   malloc_all_blocks_betta<betta_type><<<(num_segments*256-1)/512+1,512>>>(allocator, num_segments, 256);

   cudaDeviceSynchronize();

   peek<betta_type><<<1,1>>>(allocator);

   cudaDeviceSynchronize();

   betta_type::free_on_device(allocator);

}


template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void boot_betta_malloc_free(uint64_t num_bytes){

   using betta_type = poggers::allocators::betta_allocator<mem_segment_size, smallest, largest>;

   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

   cudaDeviceSynchronize();

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   register_all_segments<betta_type><<<(num_segments-1)/512+1,512>>>(allocator, num_segments);

   offset_alloc_bitarr ** blocks;

   cudaMalloc((void **)&blocks, sizeof(offset_alloc_bitarr *)*num_segments*256);

   printf("Ext sees %llu segments\n", num_segments);
   cudaDeviceSynchronize();

   poggers::utils::print_mem_in_use();


   cudaDeviceSynchronize();

   //malloc_all_blocks_single_thread<betta_type><<<1,1>>>(allocator, num_segments, 256);
   //malloc_all_blocks<betta_type><<<(num_segments*128-1)/512+1,512>>>(allocator, num_segments*128);

   malloc_and_save_blocks<betta_type><<<(num_segments*256-1)/512+1,512>>>(allocator, blocks, num_segments, 256);

   cudaDeviceSynchronize();

   allocator->print_info();

   cudaDeviceSynchronize();


   assert_unique_blocks<<<(num_segments*256 -1)/512+1, 512>>>(blocks, num_segments, 256);

   peek_blocks<betta_type><<<1,1>>>(allocator, blocks);

   cudaDeviceSynchronize();

   betta_free_all_blocks<betta_type><<<(num_segments*256-1)/512+1,512>>>(allocator, blocks, num_segments, 256);
   cudaDeviceSynchronize();


   allocator->print_info();

   cudaDeviceSynchronize();

   cudaFree(blocks);

   betta_type::free_on_device(allocator);

}


template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void boot_betta_test_all_sizes(uint64_t num_bytes){

   using betta_type = poggers::allocators::betta_allocator<mem_segment_size, smallest, largest>;

   uint64_t num_trees = betta_type::get_num_trees();

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);


   for (int i = 0; i< num_trees; i++){

      printf("Testing tree %d/%llu\n", i, num_trees);

      uint64_t blocks_per_segment = betta_type::get_blocks_per_segment(i);

      betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

      offset_alloc_bitarr ** blocks;

      cudaMalloc((void **)&blocks, sizeof(offset_alloc_bitarr *)*num_segments*blocks_per_segment);

      cudaDeviceSynchronize();

      printf("Boot done: allocator should be empty\n");
      allocator->print_info();


      cudaDeviceSynchronize();

      malloc_and_save_blocks_tree<betta_type><<<(num_segments*blocks_per_segment-1)/512+1,512>>>(allocator, blocks, num_segments, blocks_per_segment, i);


      cudaDeviceSynchronize();

      printf("Should see 0 free\n");
      allocator->print_info();

      cudaDeviceSynchronize();

      assert_unique_blocks<<<(num_segments*blocks_per_segment -1)/512+1, 512>>>(blocks, num_segments, blocks_per_segment);

      cudaDeviceSynchronize();

      betta_free_all_blocks<betta_type><<<(num_segments*blocks_per_segment-1)/512+1,512>>>(allocator, blocks, num_segments, blocks_per_segment);
   
      cudaDeviceSynchronize();

      printf("Should see all free\n");
      allocator->print_info();

      cudaDeviceSynchronize();


      cudaFree(blocks);

      betta_type::free_on_device(allocator);

   }

   cudaDeviceSynchronize();

}


template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void one_boot_betta_test_all_sizes(uint64_t num_bytes){

   using betta_type = poggers::allocators::betta_allocator<mem_segment_size, smallest, largest>;

   uint64_t num_trees = betta_type::get_num_trees();

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);


   for (int i = 0; i< num_trees; i++){

      printf("Testing tree %d/%llu\n", i, num_trees);

      uint64_t blocks_per_segment = betta_type::get_blocks_per_segment(i);

      offset_alloc_bitarr ** blocks;

      cudaMalloc((void **)&blocks, sizeof(offset_alloc_bitarr *)*num_segments*blocks_per_segment);

      cudaDeviceSynchronize();

      printf("Boot done: allocator should be empty\n");
      allocator->print_info();


      cudaDeviceSynchronize();

      malloc_and_save_blocks_tree<betta_type><<<(num_segments*blocks_per_segment-1)/512+1,512>>>(allocator, blocks, num_segments, blocks_per_segment, i);


      cudaDeviceSynchronize();

      printf("Should see 0 free\n");
      allocator->print_info();

      cudaDeviceSynchronize();

      assert_unique_blocks<<<(num_segments*blocks_per_segment -1)/512+1, 512>>>(blocks, num_segments, blocks_per_segment);

      cudaDeviceSynchronize();

      betta_free_all_blocks<betta_type><<<(num_segments*blocks_per_segment-1)/512+1,512>>>(allocator, blocks, num_segments, blocks_per_segment);
   
      cudaDeviceSynchronize();

      printf("Should see all free\n");
      allocator->print_info();

      cudaDeviceSynchronize();


      cudaFree(blocks);

     

   }

   cudaDeviceSynchronize();

   betta_type::free_on_device(allocator);

}

template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void betta_alloc_random(uint64_t num_bytes, uint64_t num_allocs){

   using betta_type = poggers::allocators::betta_allocator<mem_segment_size, smallest, largest>;

   uint64_t num_trees = betta_type::get_num_trees();

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

   alloc_random_blocks<betta_type><<<(num_allocs-1)/512+1, 512>>>(allocator);


   cudaDeviceSynchronize();

   allocator->print_info();

   cudaDeviceSynchronize();

   betta_type::free_on_device(allocator);

}







//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   // boot_ext_tree<8ULL*1024*1024, 16ULL>();
 
   // boot_ext_tree<8ULL*1024*1024, 4096ULL>();


   // boot_alloc_table<8ULL*1024*1024, 16ULL>();


   //boot_betta_malloc_free<16ULL*1024*1024, 16ULL, 64ULL>(30ULL*1000*1000*1000);

   //one_boot_betta_test_all_sizes<16ULL*1024*1024, 16ULL, 4096ULL>(2000ULL*16*1024*1024);

   betta_alloc_random<16ULL*1024*1024, 16ULL, 16ULL>(10ULL*16*1024*1024, 1000);

   cudaDeviceReset();
   return 0;

}
