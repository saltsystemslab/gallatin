#ifndef GAL_BLOCK_TEST_CU
#define GAL_BLOCK_TEST_CU

/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#include <gallatin_block_tests.hpp>
#include <gallatin/allocators/config.cuh>
#include <gallatin/allocators/block.cuh>
#include <gallatin_static_funcs.hpp>



using gallatin_block_type = gallatin::internals::block;



__global__ void wipe_block_kernel(gallatin_block_type * block, uint16_t tree_id){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0 )return;

  block->reset(tree_id);

}

__host__ void wipe_block(gallatin_block_type * block, uint16_t tree_id){

  wipe_block_kernel<<<1,1>>>(block, tree_id);

  GPUErrorCheck(cudaDeviceSynchronize());

}


__global__ void readMallocCountKernel(gallatin_block_type * block, uint * count){


  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  count[0] = block->malloc_counter;

}

__global__ void readFreeCountKernel(gallatin_block_type * block, uint * count){


  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  count[0] = block->free_counter;

}


__global__ void mallocKernel(gallatin_block_type * block, uint64_t increment_amount, uint64_t * count){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  auto my_tile = cg::coalesced_threads();

  bool needs_reset = false;

  count[0] = block->block_malloc(my_tile, increment_amount, 0, needs_reset);

}

__global__ void freeKernel(gallatin_block_type * block, bool * result){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  auto my_tile = cg::coalesced_threads();

  result[0] = block->block_free(my_tile);

}

__global__ void multiMallocKernel(gallatin_block_type * block, uint64_t malloc_size, uint64_t n_mallocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_mallocs) return;

  auto my_tile = cg::coalesced_threads();

  bool needs_reset = false;

  uint64_t alloc = block->block_malloc(my_tile, malloc_size, 0, needs_reset);

  if (alloc == ~0ULL){
    printf("Failure to allocate for thread %lu/%lu size %lu\n", tid, n_mallocs, malloc_size);
  } else {
    block->block_free(my_tile);
  }

  
}

// __global__ void setBitsKernel(gallatin_block_type * block, uint64_t * bits, uint64_t malloc_size, uint64_t n_mallocs){

//   uint64_t tid = gallatin::utils::get_tid();

//   if (tid >= n_mallocs) return;

//   auto my_tile = cg::coalesced_threads();

//   uint64_t alloc = block->block_malloc(my_tile, malloc_size, 0);

//   if (alloc == ~0ULL){
//     printf("Failure to allocate for thread %lu/%lu size %lu\n", tid, n_mallocs, malloc_size);
//   } else { 

//     uint64_t high = alloc/64;
//     uint64_t low = alloc % 64;

//     if (atomicOr((unsigned long long int *)&bits[high], SET_BIT_MASK(low)) & SET_BIT_MASK(low)){
//       printf("Double map of bit %lu\n", alloc);
//     }

//   }
  
// }

// __global__ void checkBitsKernel(gallatin_block_type * block, uint64_t * bits, uint64_t malloc_size, uint64_t n_mallocs){

//   uint64_t tid = gallatin::utils::get_tid();

//   if (tid >= 4096) return;

//   uint64_t high = tid/64;
//   uint64_t low = tid % 64;

//   bool my_bit = bits[high] & SET_BIT_MASK(low);

//   if (my_bit){

//     for (uint64_t i=1; i < malloc_size; i++){

//       uint64_t high = (tid+i)/64;
//       uint64_t low = (tid+i) % 64;

//       if (bits[high] & SET_BIT_MASK(low)){
//         printf("Double malloc between bits %lu and %lu\n", tid, tid+i);
//       }

//     }

//   }
  
// }

// __global__ void freeBitsKernel(gallatin_block_type * block, uint64_t * bits, uint64_t malloc_size, uint64_t n_mallocs){

//   uint64_t tid = gallatin::utils::get_tid();

//   if (tid >= 4096) return;

//   uint64_t high = tid/64;
//   uint64_t low = tid % 64;

//   bool my_bit = bits[high] & SET_BIT_MASK(low);

//   if (my_bit){

//     auto my_tile = cg::coalesced_threads();

//     block->block_free(my_tile);

//   }



// }


__global__ void multiMallocRounds(gallatin_block_type * block, uint64_t malloc_size, uint64_t n_mallocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_mallocs) return;


  while (true){

    auto my_tile = cg::coalesced_threads();

    bool needs_reset = false;

    uint64_t alloc = block->block_malloc(my_tile, malloc_size, 0, needs_reset);

    if (needs_reset){
      block->reset(0);
    }

    if (alloc >= 4096 && alloc != ~0ULL) printf("Bad alloc\n");

    if (alloc == ~0ULL){

      //adding in time stops block from rolling over too soon.
      //this isn't a realistic issue as threads in Gallatin will replace block.
      //__nanosleep(5120);
      __threadfence();

      continue;

    } else {

      auto free_tile = cg::coalesced_threads();


      if (block->block_free(free_tile)){

        //printf("Resetting block\n");
        block->reset(0);
        __threadfence();

      }

      return;
    }

  }


  
}

struct block_wrapper::block_int {

  gallatin_block_type * block;

  gallatin_block_type * get_block(){
    return block;
  }

  void set_block(gallatin_block_type * ext_block){
    block = ext_block;
  }


};

block_wrapper::block_wrapper(){

  //printf("Starting block tests\n");

  gallatin_block_type * block = gallatin::utils::get_device_version<gallatin_block_type>();

  wipe_block(block, 0);

  pimpl = gallatin::utils::get_host_version<block_int>();

  pimpl->set_block(block);


  bits = gallatin::utils::get_device_version<uint64_t>(64);

  cudaMemset(bits, 0, sizeof(uint64_t)*64);

  //printf("setup done\n");

}

block_wrapper::~block_wrapper(){

  gallatin_block_type * block = pimpl->get_block();

  cudaFree(block);

  cudaFree(bits);

  cudaFreeHost(pimpl);

  GPUErrorCheck(cudaDeviceSynchronize());


  //printf("Block free done\n");

  return;

}


uint block_wrapper::readMallocCount(){

  gallatin_block_type * block = pimpl->get_block();

  uint * read;

  cudaMallocManaged((void **)&read, sizeof(uint));

  cudaDeviceSynchronize();

  readMallocCountKernel<<<1,1>>>(block, read);

  GPUErrorCheck(cudaDeviceSynchronize());

  uint return_val = read[0];

  cudaFree(read);

  GPUErrorCheck(cudaDeviceSynchronize());

  return return_val;


}

uint block_wrapper::readFreeCount(){

  gallatin_block_type * block = pimpl->get_block();

  uint * read;

  cudaMallocManaged((void **)&read, sizeof(uint));

  cudaDeviceSynchronize();

  readFreeCountKernel<<<1,1>>>(block, read);

  GPUErrorCheck(cudaDeviceSynchronize());

  uint return_val = read[0];

  cudaFree(read);

  GPUErrorCheck(cudaDeviceSynchronize());

  return return_val;


}


uint64_t block_wrapper::malloc(uint increment_amount){

  gallatin_block_type * block = pimpl->get_block();

  uint64_t * read;

  cudaMallocManaged((void **)&read, sizeof(uint64_t));

  cudaDeviceSynchronize();

  mallocKernel<<<1,1>>>(block, increment_amount, read);

  GPUErrorCheck(cudaDeviceSynchronize());

  uint64_t return_val = read[0];

  cudaFree(read);

  GPUErrorCheck(cudaDeviceSynchronize());

  return return_val;

}

bool block_wrapper::free(){

  gallatin_block_type * block = pimpl->get_block();

  bool * read;

  cudaMallocManaged((void **)&read, sizeof(bool));

  cudaDeviceSynchronize();

  freeKernel<<<1,1>>>(block, read);

  GPUErrorCheck(cudaDeviceSynchronize());

  bool return_val = read[0];

  cudaFree(read);

  GPUErrorCheck(cudaDeviceSynchronize());

  return return_val;

}



void block_wrapper::open_global_log(){

  gallatin::internals::init_global_error_log();
}

int block_wrapper::close_global_log(){
  return gallatin::internals::close_global_error_log();
}


//Tests

bool block_wrapper::testReset(){

  //printf("Starting reset test\n");

  open_global_log();


  //printf("Log open\n");

  gallatin_block_type * block = pimpl->get_block();

  wipe_block(block, 0);

  uint malloc_count = readMallocCount();
  uint free_count = readFreeCount();

  return (malloc_count==0 && free_count==0 && close_global_log()==0);


}



bool block_wrapper::testSingleThread(){

  open_global_log();

  gallatin_block_type * block = pimpl->get_block();


  for (uint i = 1; i < 4096; i++){


    wipe_block(block, 0);

    uint mallocResult = malloc(i);

    bool freeResult = free();

    uint read_malloc = readMallocCount();

    uint read_free = readFreeCount();

    if (mallocResult != 0 || read_malloc != i){
      
      printf("Failed malloc on %u\n", i);
      return false;

    }

    if (i != 4096) freeResult = !freeResult;

    if (!freeResult || read_free != read_malloc){


      printf("Failed free on %u\n", i); 

      if (!freeResult){
        printf("Free for %u failed\n", i);
      } else {
        printf("Should read %u, read %u instead\n", read_malloc, read_free);
      }


      return false;

    }
  
  }



  int failures = close_global_log();

  return failures == 0;
}



bool block_wrapper::testMultiThread(){

  open_global_log();

  gallatin_block_type * block = pimpl->get_block();


  for (uint64_t i = 0; i < 12; i++){

    wipe_block(block, 0);

    cudaDeviceSynchronize();

    uint64_t malloc_size = 1ULL << i;

    multiMallocKernel<<<(4096-1)/256+1,256>>>(block, malloc_size, 4096/malloc_size);

    GPUErrorCheck(cudaDeviceSynchronize());


    uint read_malloc = readMallocCount();

    uint read_free = readFreeCount();


    if (read_malloc != 4096 || read_free != 4096){
      printf("Failed test %u %u\n", read_malloc, read_free);

      return false;
    }



  }

  int failures = close_global_log();

  return failures == 0;
}


bool block_wrapper::testMultiRounds(){

  open_global_log();

  gallatin_block_type * block = pimpl->get_block();


  uint64_t n_rounds = 10;

  for (uint64_t i = 1; i <= 32; i++){

    wipe_block(block, 0);

    cudaDeviceSynchronize();

    multiMallocRounds<<<(n_rounds*4096-1)/256+1,256>>>(block, i, n_rounds*4096);

    GPUErrorCheck(cudaDeviceSynchronize());

    //printf("\033[1;32m[ PROGRESS ]\033[1;0m Done with round %lu\n", i);

  }

  int failures = close_global_log();

  return failures == 0;
}


bool block_wrapper::testInvalidOne(){

  open_global_log();

  gallatin_block_type * block = pimpl->get_block();

  wipe_block(block, 1);

  uint64_t malloc_size = 1;

  uint64_t mallocResult = malloc(malloc_size); 

  return mallocResult == ~0ULL && close_global_log() == 0;

}


bool block_wrapper::testInvalidTwo(){

  open_global_log();

  gallatin_block_type * block = pimpl->get_block();

  wipe_block(block, 0);

  uint64_t mallocResult = malloc(4096); 

  uint free_count = readFreeCount();

  if (free_count != 4095){
    printf("Bad first free count %u\n", free_count);
    return false;
  }

  bool final_free = free();

  free_count = readFreeCount();

  if (free_count != 4096){
    printf("Bad second free count %u\n", free_count);
    return false;
  }

  free();

  return final_free == true && close_global_log() == 1;

}


#endif
