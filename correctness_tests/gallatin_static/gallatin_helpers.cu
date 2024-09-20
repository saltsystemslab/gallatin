#ifndef GAL_HELPER_CU
#define GAL_HELPER_CU

/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#define GALLATIN_DEBUG 1


#include <gallatin_helpers.hpp>
#include <gallatin/allocators/gallatin.cuh>


using namespace gallatin::allocators;


//using internal_bitarry_type = internal_bitarray;

template <typename allocator>
__global__ void allocate_single_kernel(allocator * gallatin, uint64_t n_allocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint64_t * alloc = (uint64_t *) gallatin->malloc_slice(0,1);

  if (alloc == nullptr){
    write_global_log(4,0, tid);
    return;
  }

  alloc[0] = tid;
  alloc[1] = tid;


}

template <typename allocator>
__global__ void allocate_single_kernel_store(allocator * gallatin, uint64_t ** write_array, uint64_t n_allocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint64_t * alloc = (uint64_t *) gallatin->malloc_slice(0,1);

  if (alloc == nullptr){
    write_global_log(4,0, tid);
    return;
  }

  for (uint i = 0; i <= 1; i++){
    alloc[i] = tid;
  }

  write_array[tid] = alloc;


}

template <typename allocator>
__global__ void allocate_set_size(allocator * gallatin, uint64_t ** write_array, uint64_t n_allocs, uint64_t size, uint64_t alloc_size){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint64_t * alloc = (uint64_t *) gallatin->malloc_slice(size, 1);

  if (alloc == nullptr){
    write_global_log(4,0, tid);
    return;
  }



  for (uint64_t i = 0; i < alloc_size; i+=8){

    uint64_t index = i/8;

    alloc[index] = tid;

  }

  write_array[tid] = alloc;


}

template <typename allocator>
__global__ void allocate_set_size_malloc(allocator * gallatin, uint64_t ** write_array, uint64_t n_allocs, uint64_t size, uint64_t alloc_size){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint64_t * alloc = (uint64_t *) gallatin->malloc(alloc_size);

  if (alloc == nullptr){
    write_global_log(4,0, tid);
    return;
  }



  for (uint64_t i = 0; i < alloc_size; i+=8){

    uint64_t index = i/8;

    alloc[index] = tid;

  }

  write_array[tid] = alloc;


}

template <typename allocator>
__global__ void free_single_kernel_store(allocator * gallatin, uint64_t ** write_array, uint64_t n_allocs, uint64_t * segment_counts, uint64_t * block_counts){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint64_t * alloc = write_array[tid];

  if (alloc[0] != tid || alloc[1] != tid){
    write_global_log(4,1,tid);
    return;
  }

  //uint64_t alloc_as_offset = ((char *) alloc) - gallatin->memory_base;

  //doesn't trigger - why?
  // if (alloc_as_offset == 0){
  //   printf("0!\n");
  // }

  gallatin->free(alloc);


}


template <typename allocator>
__global__ void free_single_kernel_store_size(allocator * gallatin, uint64_t ** write_array, uint64_t n_allocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint64_t * alloc = write_array[tid];

  //uint64_t alloc_as_offset = ((char *) alloc) - gallatin->memory_base;

  //doesn't trigger - why?
  // if (alloc_as_offset == 0){
  //   printf("0!\n");
  // }

  gallatin->free(alloc);


}

template <typename allocator>
__global__ void free_single_kernel_store_singleton(allocator * gallatin, uint64_t ** write_array, uint64_t n_allocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;


  for (uint64_t i = 0; i < n_allocs; i++){

    if (i % 4096 == 0){
      printf("%lu\n",i/4096);
    }

    uint64_t * alloc = write_array[i];

    //uint64_t alloc_as_offset = ((char *) alloc) - gallatin->memory_base;

    gallatin->free(alloc);

  }




}


//assert no double allocs
__global__ void test_allocator_alloc_uniqueness(uint64_t ** write_array, uint64_t n_allocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint64_t * alloc = write_array[tid];

  if (alloc[0] != tid){
    write_global_log(4,3,tid, alloc[0]);
  }

  if (alloc[1] != tid){
    write_global_log(4,4,tid, alloc[1]);
  }

}


__global__ void test_allocator_alloc_uniqueness_size(uint64_t ** write_array, uint64_t n_allocs, uint64_t size){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint64_t * alloc = write_array[tid];


  uint index = 0;

  for (uint64_t i = 0; i < size; i+=8){
    if (alloc[index] != tid){
      write_global_log(4, tid, index, alloc[index]);
    }
    index+=1;
  }


}

template <typename allocator>
__global__ void allocate_singleton_kernel(allocator * gallatin, uint64_t n_allocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;


  for (uint64_t i = 0; i < n_allocs; i++){

    if (i == 4096){
      printf("Marked here\n");
    }

    uint64_t * alloc = (uint64_t *) gallatin->malloc_slice(0,1);

    if (alloc == nullptr){
      write_global_log(4,0, i);
      return;
    }

    alloc[0] = tid;
    alloc[1] = tid;


  }


}

template <typename allocator>
__global__ void count_allocations_per_segment(allocator * gallatin, uint64_t * segment_counts, uint64_t * block_counts, uint64_t ** allocs, uint64_t n_allocs){

  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_allocs) return;

  uint64_t * allocation = allocs[tid];

  uint64_t alloc_as_offset = gallatin->get_allocation_as_offset(allocation);

  uint64_t segment_num = alloc_as_offset/allocator::get_bytes_per_segment();

  uint64_t block = (alloc_as_offset % allocator::get_bytes_per_segment())/(16*4096) + segment_num*256;

  uint64_t old = atomicAdd((unsigned long long int *)&segment_counts[segment_num], 1ULL);

  if (old >= allocator::get_bytes_per_segment()/16){
    write_global_log(4, segment_num, old);
  }

  old = atomicAdd((unsigned long long int *)&block_counts[block], 1ULL);

  if (old >= 4096){
    write_global_log(4, 1, block, old);
  }


}

//kernels

bool gallatin_tests::testAllocInit(){

  open_global_log();

  using gallatin_type = Gallatin<16U*1024*1024, 16,4096, 4, 4>;


  gallatin_type * allocator = gallatin_type::generate_on_device(8ULL*1024*1024*1024, 42);


  GPUErrorCheck(cudaDeviceSynchronize());

  gallatin_type::free_on_device(allocator);


  return (close_global_log() == 0);


}


bool gallatin_tests::testSliceAllocSingle(){

  open_global_log();

  using gallatin_type = Gallatin<16U*1024*1024, 16,16, 4, 4>;


  uint64_t n_memory = 8ULL*1024*1024*1024;

  uint64_t n_allocs = (n_memory-(16U*1024*1024*14))/16;

  gallatin_type * allocator = gallatin_type::generate_on_device(n_memory, 42);


  GPUErrorCheck(cudaDeviceSynchronize());

  allocate_single_kernel<gallatin_type><<<(n_allocs-1)/256+1,256>>>(allocator, n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());

  gallatin_type::free_on_device(allocator);


  return (close_global_log() == 0);


}


void run_mini_test(uint64_t n_allocs){

  using gallatin_type = Gallatin<16U*1024*1024, 16,16, 4, 4>;


  uint64_t n_memory = 8ULL*1024*1024*1024;

  gallatin_type * allocator = gallatin_type::generate_on_device(8ULL*1024*1024*1024, 42);


  GPUErrorCheck(cudaDeviceSynchronize());

  allocate_single_kernel<gallatin_type><<<(n_allocs-1)/256+1,256>>>(allocator, n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());

  gallatin_type::free_on_device(allocator);


}



bool gallatin_tests::testSliceAllocMini(){

  open_global_log();

  run_mini_test(10);
  run_mini_test(100);
  run_mini_test(4096);
  run_mini_test(4097);
  // run_mini_test(10000);


  return (close_global_log() == 0);


}

void run_singleton_test(uint64_t n_allocs){

  using gallatin_type = Gallatin<16U*1024*1024, 16,16, 4, 4>;


  uint64_t n_memory = 8ULL*1024*1024*1024;

  gallatin_type * allocator = gallatin_type::generate_on_device(8ULL*1024*1024*1024, 42);


  GPUErrorCheck(cudaDeviceSynchronize());

  allocate_singleton_kernel<gallatin_type><<<1,1>>>(allocator, n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());

  gallatin_type::free_on_device(allocator);


}


bool gallatin_tests::testSliceAllocSingletons(){

  open_global_log();

  run_singleton_test(10);
  run_singleton_test(100);
  run_singleton_test(4096);
  run_singleton_test(4097);
  // run_mini_test(10000);


  return (close_global_log() == 0);


}


//test allocatons and frees
bool gallatin_tests::testSliceAllocFree(){

  open_global_log();

  using gallatin_type = Gallatin<16U*1024*1024, 16,16, 4, 4>;

  uint64_t n_segments = 1024;

  uint64_t n_memory = n_segments*16*1024*1024;

  uint64_t n_allocs = (n_memory-(20ULL*1024*1024*16))/16;

  printf("Testing with %lu bytes, %lu allocs of 16 bytes\n", n_memory, n_allocs);

  gallatin_type * allocator = gallatin_type::generate_on_device(n_memory, 42);

  uint64_t ** write_array = gallatin::utils::get_device_version<uint64_t *>(n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());

  allocate_single_kernel_store<gallatin_type><<<(n_allocs-1)/256+1,256>>>(allocator, write_array, n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());

  test_allocator_alloc_uniqueness<<<(n_allocs-1)/256+1,256>>>(write_array, n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());

  uint64_t * segment_counts = gallatin::utils::get_device_version<uint64_t>(n_segments);

  cudaMemset(segment_counts, 0, sizeof(uint64_t)*n_segments);

  uint64_t * block_counts = gallatin::utils::get_device_version<uint64_t>(n_segments*256);

  cudaMemset(block_counts, 0, sizeof(uint64_t)*n_segments*256);

  count_allocations_per_segment<gallatin_type><<<(n_allocs-1)/256+1,256>>>(allocator, segment_counts, block_counts, write_array, n_allocs);


  GPUErrorCheck(cudaDeviceSynchronize());

  free_single_kernel_store<gallatin_type><<<(n_allocs-1)/256+1,256>>>(allocator,write_array, n_allocs, segment_counts, block_counts);

  GPUErrorCheck(cudaDeviceSynchronize());

  cudaFree(segment_counts);
  cudaFree(block_counts);


  gallatin_type::free_on_device(allocator);

  cudaFree(write_array);


  return (close_global_log() == 0);


}


//test allocatons and frees
bool gallatin_tests::testSliceAllocFreeSingleton(){

  open_global_log();

  using gallatin_type = Gallatin<16U*1024*1024, 16,16, 4, 4>;

  uint64_t n_segments = 15;

  uint64_t n_memory = n_segments*16*1024*1024;

  uint64_t n_allocs = (n_memory-(20ULL*1024*1024*14))/16;

  printf("Testing with %lu bytes, %lu allocs of 16 bytes\n", n_memory, n_allocs);

  gallatin_type * allocator = gallatin_type::generate_on_device(n_memory, 42);

  uint64_t ** write_array = gallatin::utils::get_device_version<uint64_t *>(n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());

  allocate_single_kernel_store<gallatin_type><<<(n_allocs-1)/256+1,256>>>(allocator, write_array, n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());

  test_allocator_alloc_uniqueness<<<(n_allocs-1)/256+1,256>>>(write_array, n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());

  uint64_t * segment_counts = gallatin::utils::get_device_version<uint64_t>(n_segments);

  cudaMemset(segment_counts, 0, sizeof(uint64_t)*n_segments);

  uint64_t * block_counts = gallatin::utils::get_device_version<uint64_t>(n_segments*256);

  cudaMemset(block_counts, 0, sizeof(uint64_t)*n_segments*256);

  count_allocations_per_segment<gallatin_type><<<(n_allocs-1)/256+1,256>>>(allocator, segment_counts, block_counts, write_array, n_allocs);


  GPUErrorCheck(cudaDeviceSynchronize());

  free_single_kernel_store_singleton<gallatin_type><<<1,1>>>(allocator,write_array, n_allocs);

  GPUErrorCheck(cudaDeviceSynchronize());



  gallatin_type::free_on_device(allocator);

  cudaFree(write_array);

  cudaFree(block_counts);
  cudaFree(segment_counts);


  return (close_global_log() == 0);


}


void gallatin_tests::open_global_log(){

  gallatin::internals::init_global_error_log();
  
}

int gallatin_tests::close_global_log(){
  return gallatin::internals::close_global_error_log();
}



//full battery for allocatons and frees
bool gallatin_tests::testSliceAllocFreeAllSizes(){

  open_global_log();

  using gallatin_type = Gallatin<16U*1024*1024, 16, 4096, 4, 4>;

  uint64_t n_segments = 1024;

  uint64_t n_memory = n_segments*16*1024*1024;



  uint64_t iter = 0;
  for (uint64_t alloc_size = 16; alloc_size <= 4096; alloc_size*=2){

    uint64_t n_allocs = (n_memory-(20ULL*1024*1024*16))/alloc_size;

    printf("Test round with %lu bytes\n", alloc_size);

    gallatin_type * allocator = gallatin_type::generate_on_device(n_memory, 42);

    uint64_t ** write_array = gallatin::utils::get_device_version<uint64_t *>(n_allocs);

    GPUErrorCheck(cudaDeviceSynchronize());


    //allocate_single_kernel_store<gallatin_type><<<(n_allocs-1)/64+1,64>>>(allocator, write_array, n_allocs);
    allocate_set_size<gallatin_type><<<(n_allocs-1)/64+1,64>>>(allocator, write_array, n_allocs, iter, alloc_size);

    GPUErrorCheck(cudaDeviceSynchronize());

    printf("Done with alloc\n");

    test_allocator_alloc_uniqueness_size<<<(n_allocs-1)/128+1,128>>>(write_array, n_allocs, alloc_size);

    GPUErrorCheck(cudaDeviceSynchronize());

    free_single_kernel_store_size<gallatin_type><<<(n_allocs-1)/128+1,128>>>(allocator,write_array, n_allocs);

    GPUErrorCheck(cudaDeviceSynchronize());

    cudaFree(write_array);

    gallatin_type::free_on_device(allocator);

    iter+=1;

  }

  
  return (close_global_log() == 0);


}


//full battery for allocatons and frees
bool gallatin_tests::testSliceAllocFreeMalloc(){

  open_global_log();

  using gallatin_type = Gallatin<16U*1024*1024, 16, 4096, 4, 4>;

  uint64_t n_segments = 1024;

  uint64_t n_memory = n_segments*16*1024*1024;



  uint64_t iter = 0;
  for (uint64_t alloc_size = 16; alloc_size <= 4096; alloc_size*=2){

    uint64_t n_allocs = (n_memory-(20ULL*1024*1024*16))/alloc_size;

    printf("Test round with %lu bytes\n", alloc_size);

    gallatin_type * allocator = gallatin_type::generate_on_device(n_memory, 42);

    uint64_t ** write_array = gallatin::utils::get_device_version<uint64_t *>(n_allocs);

    GPUErrorCheck(cudaDeviceSynchronize());


    //allocate_single_kernel_store<gallatin_type><<<(n_allocs-1)/64+1,64>>>(allocator, write_array, n_allocs);
    allocate_set_size_malloc<gallatin_type><<<(n_allocs-1)/64+1,64>>>(allocator, write_array, n_allocs, iter, alloc_size);

    GPUErrorCheck(cudaDeviceSynchronize());

    printf("Done with alloc\n");

    test_allocator_alloc_uniqueness_size<<<(n_allocs-1)/128+1,128>>>(write_array, n_allocs, alloc_size);

    GPUErrorCheck(cudaDeviceSynchronize());

    free_single_kernel_store_size<gallatin_type><<<(n_allocs-1)/128+1,128>>>(allocator,write_array, n_allocs);

    GPUErrorCheck(cudaDeviceSynchronize());

    cudaFree(write_array);

    gallatin_type::free_on_device(allocator);

    iter+=1;

  }

  
  return (close_global_log() == 0);


}


#endif




