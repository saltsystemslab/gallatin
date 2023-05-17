/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */





#include <poggers/counter_blocks/beta.cuh>

#include <poggers/beta/timer.cuh>


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

using namespace beta::allocators;


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

         Block * new_block = betta->table->get_block(i);

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

   Block * new_block = betta->table->get_block(segment_id, last_block);

   if (new_block == nullptr){

      printf("Missed block %llu in section %llu\n", tid, segment_id);

   }




}

//pull all blocks using betta
template <typename betta_type>
__global__ void malloc_all_blocks_betta(betta_type * betta, uint64_t num_segments, uint64_t blocks_per_segment){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return;

   Block * new_block = betta->request_new_block_from_tree(0);

   if (new_block == nullptr){
      printf("Failed to get block!\n");
   }

}


template <typename betta_type>
__global__ void malloc_and_save_blocks(betta_type * betta, Block ** blocks, uint64_t num_segments, uint64_t blocks_per_segment){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return;

   Block * new_block = betta->request_new_block_from_tree(0);

   if (new_block == nullptr){
      printf("Alloc failure in 1\n");

      new_block = betta->request_new_block_from_tree(0);
   }

   blocks[tid] = new_block;

}


template <typename betta_type>
__global__ void malloc_and_save_blocks_tree(betta_type * betta, Block ** blocks, uint64_t num_segments, uint64_t blocks_per_segment, int tree_id){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return;

   Block * new_block = betta->request_new_block_from_tree(tree_id);

   while (new_block == nullptr){
      //printf("Alloc failure in 2\n");

      new_block = betta->request_new_block_from_tree(tree_id);
   }

   blocks[tid] = new_block;

}


template <typename betta_type>
__global__ void betta_free_all_blocks(betta_type * betta, Block ** blocks, uint64_t num_segments, uint64_t blocks_per_segment){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return;

   Block * new_block = blocks[tid];

   if (new_block == nullptr) return;

   betta->free_block(new_block);

}



template <typename betta_type>
__global__ void malloc_all_blocks_betta_single_thread(betta_type * betta, uint64_t num_segments, uint64_t blocks_per_segment){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   uint64_t misses = 0;

   for (uint64_t i = 0; i < num_segments*blocks_per_segment; i++){


      Block * new_block = betta->request_new_block_from_tree(0);

      if (new_block == nullptr){
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
__global__ void peek_blocks(betta_type * betta, Block ** blocks){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid !=0 ) return;
}


__global__ void assert_unique_blocks(Block ** blocks, uint64_t num_segments, uint64_t blocks_per_segment){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_segments*blocks_per_segment) return;

   uint64_t my_block = (uint64_t) blocks[tid];

   //if (my_block == 0) return;


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

   Block * my_blocks[10];

   uint64_t num_trees = betta_type::get_num_trees();


   for (int i = 0; i < 1; i++){

      int tree = poggers::hashers::MurmurHash64A (&tid, sizeof(uint64_t), i) % num_trees;

      my_blocks[i] = betta->request_new_block_from_tree(tree);

   }


   for (int i = 0; i < 1; i++){

      if (my_blocks[i] == nullptr){
         printf("Failed to alloc\n");
      } else {
         betta->free_block(my_blocks[i]);
      }

      

   }

   //printf("Done with %llu\n", tid);



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

   using betta_type = beta::allocators::beta_allocator<mem_segment_size, smallest, largest>;

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

   using betta_type = beta::allocators::beta_allocator<mem_segment_size, smallest, largest>;

   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

   cudaDeviceSynchronize();

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   register_all_segments<betta_type><<<(num_segments-1)/512+1,512>>>(allocator, num_segments);

   Block ** blocks;

   cudaMalloc((void **)&blocks, sizeof(Block *)*num_segments*256);

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

   using betta_type = beta::allocators::beta_allocator<mem_segment_size, smallest, largest>;

   uint64_t num_trees = betta_type::get_num_trees();

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);


   for (int i = 0; i< num_trees; i++){

      printf("Testing tree %d/%llu\n", i, num_trees);

      uint64_t blocks_per_segment = betta_type::get_blocks_per_segment(i);

      betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

      Block ** blocks;

      cudaMalloc((void **)&blocks, sizeof(Block *)*num_segments*blocks_per_segment);

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

   using betta_type = beta::allocators::beta_allocator<mem_segment_size, smallest, largest>;

   uint64_t num_trees = betta_type::get_num_trees();

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);


   for (int i = 0; i< num_trees; i++){

      printf("Testing tree %d/%llu\n", i, num_trees);

      uint64_t blocks_per_segment = betta_type::get_blocks_per_segment(i);

      Block ** blocks;

      cudaMalloc((void **)&blocks, sizeof(Block *)*num_segments*blocks_per_segment);

      cudaDeviceSynchronize();

      printf("Boot done: allocator should be empty\n");
      allocator->print_info();


      cudaDeviceSynchronize();

      beta::utils::timer malloc_timing;
      malloc_and_save_blocks_tree<betta_type><<<(num_segments*blocks_per_segment-1)/512+1,512>>>(allocator, blocks, num_segments, blocks_per_segment, i);
      auto malloc_duration = malloc_timing.sync_end();

      cudaDeviceSynchronize();

      printf("Should see 0 free\n");
      allocator->print_info();

      cudaDeviceSynchronize();

      assert_unique_blocks<<<(num_segments*blocks_per_segment -1)/512+1, 512>>>(blocks, num_segments, blocks_per_segment);

      cudaDeviceSynchronize();


      beta::utils::timer free_timing;
      betta_free_all_blocks<betta_type><<<(num_segments*blocks_per_segment-1)/512+1,512>>>(allocator, blocks, num_segments, blocks_per_segment);
      auto free_duration = free_timing.sync_end();  

      cudaDeviceSynchronize();

      printf("Should see all free\n");
      allocator->print_info();

      cudaDeviceSynchronize();


      uint64_t total_num_blocks = num_segments*blocks_per_segment;

      malloc_timing.print_throughput("Alloced", total_num_blocks);
      free_timing.print_throughput("Freed", total_num_blocks);


      cudaFree(blocks);

     

   }

   cudaDeviceSynchronize();

   betta_type::free_on_device(allocator);

}

template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void betta_alloc_random(uint64_t num_bytes, uint64_t num_allocs){

   using betta_type = beta::allocators::beta_allocator<mem_segment_size, smallest, largest>;

   uint64_t num_trees = betta_type::get_num_trees();

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

   alloc_random_blocks<betta_type><<<(num_allocs-1)/512+1, 512>>>(allocator);


   cudaDeviceSynchronize();

   allocator->print_info();

   cudaDeviceSynchronize();

   betta_type::free_on_device(allocator);

}


template<typename allocator_type>
__global__ void alloc_one_size(allocator_type * allocator, uint64_t num_allocs, uint64_t size){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;


   uint64_t malloc = allocator->malloc(size);

   if (malloc == ~0ULL){
      printf("Fail\n");
   }


}


//allocate from blocks, and print failures.
template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void beta_test_allocs(uint64_t num_bytes){


   beta::utils::timer boot_timing;

   using betta_type = beta::allocators::beta_allocator<mem_segment_size, smallest, largest>;

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   uint64_t allocs_per_segment = mem_segment_size/largest;

   uint64_t num_allocs = allocs_per_segment*num_segments;

   printf("Starting test with %lu segments, %lu allocs per segment\n", num_segments, allocs_per_segment);

   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);

   //generate bitarry

   uint64_t num_bytes_bitarray = sizeof(uint64_t)*((num_allocs -1)/64+1);

   uint64_t * bits;

   cudaMalloc((void **)&bits, num_bytes_bitarray);

   cudaMemset(bits, 0, num_bytes_bitarray);




   std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;


   beta::utils::timer kernel_timing;
   alloc_one_size<betta_type><<<(num_allocs-1)/512+1,512>>>(allocator, .5*num_allocs, largest);
   kernel_timing.sync_end();


   kernel_timing.print_throughput("Malloced", .5*num_allocs);





   betta_type::free_on_device(allocator);


}

template<typename allocator_type>
__global__ void alloc_one_size_correctness(allocator_type * allocator, uint64_t num_allocs, uint64_t size, uint64_t * bitarray, uint64_t * misses){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;


   uint64_t malloc = allocator->malloc(size);


   uint64_t attempts = 0;

   while (malloc == ~0ULL && attempts < 250){
      malloc = allocator->malloc(size);
      attempts+=1;
   }

   if (malloc == ~0ULL){
      atomicAdd((unsigned long long int *)misses, 1ULL);
      return;
   }

   uint64_t high = malloc / 64;

   uint64_t low = malloc % 64;

   auto bitmask = SET_BIT_MASK(low);

   uint64_t bits = atomicOr((unsigned long long int *) &bitarray[high], (unsigned long long int) bitmask);

   if (bits & bitmask){
      printf("Double malloc bug in %llu: block %llu alloc %llu", malloc, malloc/4096, malloc % 4096);
   }

   __threadfence();


}



//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
template <uint64_t mem_segment_size, uint64_t smallest, uint64_t largest>
__host__ void beta_test_allocs_correctness(uint64_t num_bytes){


   beta::utils::timer boot_timing;

   using betta_type = beta::allocators::beta_allocator<mem_segment_size, smallest, largest>;

   uint64_t num_segments = poggers::utils::get_max_chunks<mem_segment_size>(num_bytes);

   uint64_t allocs_per_segment = mem_segment_size/largest;

   uint64_t num_allocs = allocs_per_segment*num_segments;

   printf("Starting test with %lu segments, %lu allocs per segment\n", num_segments, allocs_per_segment);


   betta_type * allocator = betta_type::generate_on_device(num_bytes, 42);



   //generate bitarry
   uint64_t num_bytes_bitarray = sizeof(uint64_t)*((num_allocs -1)/64+1);

   uint64_t * bits;

   cudaMalloc((void **)&bits, num_bytes_bitarray);

   cudaMemset(bits, 0, num_bytes_bitarray);


   uint64_t * misses;
   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   cudaDeviceSynchronize();

   misses[0] = 0;




   std::cout << "Init in " << boot_timing.sync_end() << " seconds" << std::endl;


   beta::utils::timer kernel_timing;
   alloc_one_size_correctness<betta_type><<<(num_allocs-1)/512+1,512>>>(allocator, .9*num_allocs, largest, bits, misses);
   kernel_timing.sync_end();


   kernel_timing.print_throughput("Malloced", .9*num_allocs);

   printf("Missed: %llu\n", misses[0]);

   cudaDeviceSynchronize();

   allocator->print_info();

   cudaFree(misses);

   cudaFree(bits);





   betta_type::free_on_device(allocator);


}


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   uint64_t num_segments;
   

   if (argc < 2){
      num_segments = 100;
   } else {
      num_segments = std::stoull(argv[1]);
   }




   //one_boot_betta_test_all_sizes<16ULL*1024*1024, 16ULL, 16ULL>(num_segments*16*1024*1024);  


   beta_test_allocs_correctness<16ULL*1024*1024, 16ULL, 16ULL>(num_segments*16*1024*1024);

   cudaDeviceReset();
   return 0;

}
