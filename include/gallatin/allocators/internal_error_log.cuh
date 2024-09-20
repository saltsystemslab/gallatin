#ifndef GALLATIN_ERROR_LOG
#define GALLATIN_ERROR_LOG

/******* ABOUT
 * The error log is a device-side global variable that records logs.
 * if the macro GALLATIN_DEBUG is set to 1 this will be initialized
 * stores a fixed number of logs - This is only intended to diagnose issues in the allocator itself.
 * If the allocator is running smoothly than it can power a dynamic logging system.
 * *******/

#include <gallatin/allocators/device_singleton.cuh>

namespace gallatin {

namespace internals {


//for now only 3 arguments supplied.
// I will update as this occurs
struct error_log_item {

  int error_code;
  uint64_t arg_0;
  uint64_t arg_1;
  uint64_t arg_2;


  __host__ void process(uint64_t error_id){


    switch(error_code){
    case 0:
      printf("\033[1;32m[ DUMMY    ]\033[1;0m %lu: Dummy Error log %lu\n", error_id, arg_0);
      break;
    case 1:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Double Free in block %lx, malloc value %lu, free value %lu\n", error_id, arg_0, arg_1, arg_2);
      break;
    case 2:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Block Malloc overflow approaching in block %lx, current count %lu\n", error_id, arg_0, arg_1);
      break;
    case 3:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  VEB segment writing index %lu, %lu bits outside of max %lu\n", error_id, arg_0, arg_0-(arg_1-1), arg_1);
      break;
    case 4:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Test Assertion Failure: %lu %lu %lu\n", error_id, arg_0, arg_1, arg_2);
      break;
    case 5:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  VEB segment writing negative index %d\n", error_id, (int) arg_0);
      break;
    case 6:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  VEB segment set/unset outside of bound: region %lu wide starting at bit %lu exceeds region %lu wide.\n", error_id, arg_0, arg_1, arg_2);
      break;
    case 7:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  VEB insert out of bounds: Setting bit %lu outside of range %lu.\n", error_id, arg_0, arg_1);
      break;
    case 8:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  VEB remove out of bounds: Setting bit %lu outside of range %lu.\n", error_id, arg_0, arg_1);
      break;
    case 9:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  VEB query out of bounds: Setting bit %lu outside of range %lu.\n", error_id, arg_0, arg_1);
      break;
    case 10:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  VEB failed to initialize segment %lu/%lu.\n", error_id, arg_0, arg_1);
      break;
    case 11:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  segment with max block %lu allocating block %lu.\n", error_id, arg_0, arg_1);
      break;
    case 12:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  segment freeing block %lu outside of range %lu\n", error_id, arg_0, arg_1);
      break;
    case 13:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  segment has allocated outside of range: Allocation %lu of size %lu is %lu bytes outside of segment size %lu\n", error_id, arg_0, arg_1, arg_0 % arg_2, arg_2);
      break;
    case 14:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Read of invalid segment size\n", error_id);
      break;
    case 15:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Segment allocation return not cleanly offset: allocation %lu not aligned to size %lu, diff %lu\n", error_id, arg_0, arg_1, arg_0 % arg_1);
      break;
    case 16:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Error allocating block %lu/%lu for size %lu\n", error_id, arg_0, arg_1, arg_2);
      break;
    case 17:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Gallatin observes multi success %lu/\n", error_id, arg_0);
      break;
    case 18:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Gallatin block tree index out of bounds %lu/%lu\n", error_id, arg_0, arg_1);
      break;
    case 19:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Gallatin primary query thread sees block tree index out of bounds %lu/%lu\n", error_id, arg_0, arg_1);
      break;
    case 20:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Gallatin Allocating block outside of range %lu >= %lu, size is %lu\n", error_id, arg_0, arg_1, arg_2);
      break;
    case 21:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Gallatin segment allocation out of bounds: Allocation at offset %lu running %lu bytes, exceeds boundary of %lu\n", error_id, arg_0, arg_1, arg_2);
      break;
    case 22:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Gallatin segment allocation does not match segment: Allocation %lu positioned in segment %lu, home segment %lu\n", error_id, arg_0, arg_1, arg_2);
      break;
    case 23:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Gallatin allocation offset at %lu bytes outside of total memory range %lu\n", error_id, arg_0, arg_1);
      break;
    case 24:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu:  Gallatin allocation offset starting at %lu bytes bleeds outside of total memory range %lu, alloc size %lu bytes\n", error_id, arg_0, arg_1, arg_2);
      break;
    case 25:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Segment return at address %lu of %lu bytes outside of range %lu\n", error_id, arg_0, arg_1, arg_2);
      break;
    case 26:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Block reset in invalid condition: Malloc %lu free %lu\n", error_id, arg_0, arg_1);
      break;
    case 27:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Block gave invalid allocation %lu\n", error_id, arg_0);
      break;
    case 28:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Free in segment %lu read invalid size %lu\n", error_id, arg_0, arg_1);
      break;
    case 29:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Size read in segment %lu disagree: %lu not previous %lu\n", error_id, arg_0, arg_1, arg_2);
      break;
    case 30:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Illegal segment read size %lu\n", error_id, arg_0);
      break;
    case 31:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Threads disagree on read size %lu != %lu\n", error_id, arg_0, arg_1);
      break;
    case 32:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Threads disagree on block ID %lu != %lu\n", error_id, arg_0, arg_1);
      break;
    case 33:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Threads disagree on block from block_free %lu != %lu\n", error_id, arg_0, arg_1);
      break;
    case 34:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Poison Out-of-bounds detected: allocation %llu was written %lu bytes ahead of allocation.\n", error_id, arg_0, arg_1+1);
      break;
    case 35:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Poison Out-of-bounds detected: allocation %lx was written %lu bytes after allocation.\n", error_id, arg_0, arg_1+1);
      break;
    case 36:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Poison Out-of-bounds [] dereference detected: allocation %lx dereference of index [%lu] writing %lu bytes outside of allocation.\n", error_id, arg_0, arg_1, arg_2);
      break;
    default:
      printf("\033[1;31m[ ERROR    ]\033[1;0m %lu: Unrecognized error code %d\n", error_id, error_code);

    }

  }


};


struct error_log {

  uint64_t current_errors;

  error_log_item logs[GALLATIN_ERROR_LOG_LENGTH];


  void init(){
    current_errors = 0;

    for (int i = 0; i < GALLATIN_ERROR_LOG_LENGTH; i++){
      logs[i] = error_log_item{0, 0ULL,0ULL, 0ULL};
    }
  }


  __device__ void add_log(int opcode, uint64_t arg_0, uint64_t arg_1, uint64_t arg_2){


    //if (this == nullptr) return;

    uint64_t log_pos = atomicAdd((unsigned long long int *)&current_errors, 1ULL);

    if (log_pos >= GALLATIN_ERROR_LOG_LENGTH) return;

    logs[log_pos] = error_log_item{opcode, arg_0, arg_1, arg_2};

  }

  __host__ static error_log * generate_on_device(){

    error_log * host_version = gallatin::utils::get_host_version<error_log>();

    host_version->init();

    return gallatin::utils::move_to_device<error_log>(host_version);

  }

  //move back and process log codes
  __host__ static int process_log(error_log * device_version){

    error_log * host_version = gallatin::utils::move_to_host<error_log>(device_version);


    uint64_t n_errors = host_version->current_errors;

    if (host_version->current_errors == 0){
      printf("\033[1;32m[ REPORT   ]\033[1;0m 0 Error logs generated by Gallatin.\n");
    } else {

      uint64_t max_errors = host_version->current_errors;

      if (host_version->current_errors > GALLATIN_ERROR_LOG_LENGTH) host_version->current_errors = GALLATIN_ERROR_LOG_LENGTH;
      printf("\033[1;31m[ REPORT   ]\033[1;0m Gallatin reporting %lu errors:\n", host_version->current_errors);

      for (uint64_t i = 0; i < host_version->current_errors; i++){

        host_version->logs[i].process(i+1);

      }

      printf("\033[1;31m[      END ]\033[1;0m %lu/%lu errors reported.\n", host_version->current_errors, max_errors);

    }


    cudaFreeHost(host_version);

    return n_errors;

  }

};



#if GALLATIN_DEBUG




//inline __managed__ error_log * global_gallatin_error_log = nullptr;

__host__ inline void init_global_error_log(){

  printf("Initialized debug log\n");

  error_log * local_copy = error_log::generate_on_device();


  singleton<error_log *>::write_instance(local_copy);

  error_log * read = singleton<error_log *>::read_instance();
  //cudaMemcpyToSymbol(global_gallatin_error_log, &local_copy, sizeof(error_log *));

  if (read != local_copy){
    printf("Singleton not set: %lx != %lx\n", (uint64_t) local_copy, (uint64_t) read);
  }

}

__host__ inline int close_global_error_log(){

  error_log * local_copy = singleton<error_log *>::read_instance();

  // cudaMallocHost((void **)&local_copy, sizeof(error_log));

  // cudaDeviceSynchronize();

  singleton<error_log *>::write_instance(nullptr);

  
  cudaDeviceSynchronize();

  if (local_copy == nullptr){
    printf("Log not initialized, value is nullptr\n");
    return -1;
  }

  int return_value = error_log::process_log(local_copy);

  //cudaFreeHost(local_copy);

  return return_value;

}


__device__ inline void write_global_log(int opcode, uint64_t arg_0=0ULL, uint64_t arg_1=0ULL, uint64_t arg_2=0ULL){


  __threadfence();
  singleton<error_log *>::instance()->add_log(opcode, arg_0, arg_1, arg_2);


}


#else

__host__ void inline init_global_error_log(){

  return;
}

__host__ int inline close_global_error_log(){
  //printf("Gallatin logs not enabled\n");
  return 0;
}

__device__ void inline write_global_log(int opcode, uint64_t arg_0=0ULL, uint64_t arg_1=0ULL, uint64_t arg_2=0ULL){
  return;
}


#endif


}  // namespace internals

}  // namespace gallatin

#endif  // End of error log